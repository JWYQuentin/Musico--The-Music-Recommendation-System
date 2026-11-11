"""The Phase 7 service: both stages run at request time, for any listener, from the bundle.

A listener the models know gets the vectors learned for them. Someone new gets vectors
folded in from the listens they hand in (bundle.vectors). From there everyone takes the
same way: their candidates from the two-tower vectors, every ranker feature, the ranker's
order for the first n_candidates, and for the tracks shown, the features that pushed each
one up the most as reasons. api.py puts these functions behind HTTP.

It imports ranker.py, so like that module it must not import torch; it imports nothing
that needs faiss or implicit either, so the service's image needs none of the three.

  tune    the fold-in weighting, binary or log(1 + plays), for the listeners the
          new-listener test held out, on val. Prints; writes nothing.
  report  the service for every known listener against the saved Phase 6 lists, the
          new-listener test, and how long a request takes. Writes reports/serving.md.
  curve   the new-listener test again with each held-out listener served from only
          their most recent tracks. Writes reports/listens_curve.md and its figure.

Usage: python -m m4a_rec.serve tune | report | curve
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xgboost as xgb

from . import bundle
from .audit import BLUE, GRID, INK, MUTED, SURFACE
from .config import load_config
from .demo import reasons
from .evaluate import PAIR, TRAIN, breakdown, paired_difference, score
from .interactions import load_train
from .ranker import load as load_ranker
from .ranker import predict, rerank, shap_values
from .reporting import stable_json, table

NEW = -1  # the user_id given to the listens of someone new
FOLD_INS = ["binary", "log"]
STAGES = ["listens", "vectors", "retrieval", "features", "ranker", "reasons"]
WARM_UP = 5  # requests run before timing starts


class Service(NamedTuple):
    bundle: bundle.Bundle
    booster: xgb.Booster
    ranker: dict  # the saved ranker's meta: features, params, trees
    titles: pl.DataFrame  # track_id, title, artist
    track_row: dict[str, int]  # track_id -> its row in the bundle's per-track arrays


def load(directory: Path) -> Service:
    b = bundle.load(directory)
    booster, meta = load_ranker(directory / "ranker")
    titles = pl.read_parquet(directory / "titles.parquet")
    return Service(b, booster, meta, titles, {t: i for i, t in enumerate(b.track_ids.to_list())})


def lap(timings: dict | None, stage: str, started: float) -> float:
    """Adds the time since `started` to `stage` in `timings`, if there is one, and returns now."""
    now = perf_counter()
    if timings is not None:
        timings[stage] = timings.get(stage, 0.0) + now - started
    return now


def recommend(
    s: Service, history: pl.DataFrame, user_ids: pl.Series, tt_user: np.ndarray, als_user: np.ndarray, timings: dict | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Both stages for a batch of listeners, from their listens and vectors.

    Returns their lists, n_recs tracks each as the Phase 6 lists have them (user_id,
    track_id, score), and the ranker's input, one row per candidate.
    """
    started = perf_counter()
    idx, scores = bundle.retrieve(s.bundle, history, user_ids, tt_user)
    started = lap(timings, "retrieval", started)
    rows = bundle.candidates(s.bundle, history, user_ids, idx, scores, als_user)
    started = lap(timings, "features", started)
    lists = rerank(rows, predict(s.booster, s.ranker, rows), s.ranker["params"]["n_candidates"])
    lap(timings, "ranker", started)
    return lists, rows


def present(s: Service, lists: pl.DataFrame, rows: pl.DataFrame, k: int, explain: bool, timings: dict | None = None) -> list[dict]:
    """One listener's first k tracks as records: track_id, rank, retrieval_rank, title, artist, reasons.

    `reasons` is None unless `explain`; then it is the features that pushed the track up the
    most, in words, as on the demo page.
    """
    started = perf_counter()
    top = lists.sort("score", descending=True).head(k).with_columns(rank=pl.int_range(1, pl.len() + 1))
    top = top.join(rows, on=PAIR, how="left", maintain_order="left").join(s.titles, on="track_id", how="left", maintain_order="left")
    said = [None] * top.height
    if explain:
        m, features = s.bundle.meta, s.ranker["features"]
        contributions = shap_values(s.booster, top, features, s.ranker["trees"])[:, :-1]
        values = top.select(features).to_numpy()
        said = [reasons(features, values[i], contributions[i], m["reasons"], m["recent_days"], m["short_days"]) for i in range(top.height)]
    out = [
        {"track_id": track, "rank": rank, "retrieval_rank": int(tt_rank), "title": title, "artist": artist, "reasons": why}
        for (track, rank, tt_rank, title, artist), why in zip(top.select("track_id", "rank", "tt_rank", "title", "artist").rows(), said)
    ]
    lap(timings, "reasons", started)
    return out


def for_user(s: Service, user_id: int, k: int, explain: bool, timings: dict | None = None) -> list[dict]:
    """A known listener's recommendations, from the vectors learned for them. KeyError if the models do not know them."""
    users = pl.Series("user_id", [user_id], dtype=pl.Int64)
    if not s.bundle.user_ids.is_in([user_id]).any():
        raise KeyError(user_id)
    started = perf_counter()
    history = bundle.listens(s.bundle, [user_id])
    started = lap(timings, "listens", started)
    tt_user, als_user = bundle.vectors(s.bundle, history, users, fold_in=None)
    lap(timings, "vectors", started)
    return present(s, *recommend(s, history, users, tt_user, als_user, timings), k, explain, timings)


def for_history(s: Service, plays: pl.DataFrame, k: int, explain: bool, timings: dict | None = None) -> tuple[list[dict], list[str]]:
    """Recommendations for someone new, from their listens: track_id and timestamp, a missing time meaning `now`.

    Returns them and the track IDs that are not in the catalogue, which are left out.
    ValueError if a listen is after `now` or no track is in the catalogue.
    """
    now = s.bundle.meta["now"]
    plays = plays.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")).fill_null(now))
    if (plays["timestamp"] > now).any():
        raise ValueError(f"a listen is after {now:%Y-%m-%d %H:%M:%S}, when the models' data ends")
    known = plays["track_id"].is_in(s.bundle.track_ids.implode())
    unknown = plays.filter(~known)["track_id"].unique(maintain_order=True).to_list()
    if not known.any():
        raise ValueError("none of the tracks is in the catalogue")
    started = perf_counter()
    history = plays.filter(known).select(pl.lit(NEW, dtype=pl.Int64).alias("user_id"), "track_id", "timestamp")
    users = pl.Series("user_id", [NEW], dtype=pl.Int64)
    tt_user, als_user = bundle.vectors(s.bundle, history, users, s.bundle.meta["fold_in"])
    lap(timings, "vectors", started)
    return present(s, *recommend(s, history, users, tt_user, als_user, timings), k, explain, timings), unknown


def similar(s: Service, track_id: str, k: int) -> list[dict]:
    """The k tracks whose two-tower vectors are closest to this one's, closest first. KeyError if it is not in the catalogue.

    The vectors have length 1, so this is what the FAISS index of Phase 6 finds.
    """
    row = s.track_row[track_id]
    scores = s.bundle.tt_track @ s.bundle.tt_track[row]
    scores[row] = -np.inf
    top = np.argsort(-scores, kind="stable")[:k]
    found = pl.DataFrame({"track_id": s.bundle.track_ids.gather(top), "similarity": scores[top]})
    found = found.join(s.titles, on="track_id", how="left", maintain_order="left")
    return [
        {"track_id": track, "rank": i + 1, "similarity": float(sim), "title": title, "artist": artist}
        for i, (track, sim, title, artist) in enumerate(found.select("track_id", "similarity", "title", "artist").rows())
    ]


def lists_for(s: Service, history: pl.DataFrame, users: list[int], fold_in: str | None, batch_users: int, stages: str = "both") -> pl.DataFrame:
    """Lists for many listeners, `batch_users` at a time: both stages, or with stages="retrieval" the candidates in retrieval order."""
    out = []
    for start in range(0, len(users), batch_users):
        chunk = pl.Series("user_id", users[start : start + batch_users], dtype=pl.Int64)
        listened = history.filter(pl.col("user_id").is_in(chunk.to_list()))
        tt_user, als_user = bundle.vectors(s.bundle, listened, chunk, fold_in)
        if stages == "retrieval":
            idx, scores = bundle.retrieve(s.bundle, listened, chunk, tt_user)
            frame = pl.DataFrame(
                {
                    "user_id": chunk.gather(np.repeat(np.arange(idx.shape[0]), idx.shape[1])),
                    "track_id": s.bundle.track_ids.gather(idx.ravel().clip(0)),
                    "score": scores.ravel(),
                }
            )
            out.append(frame.filter(pl.Series(idx.ravel() >= 0)))
        else:
            out.append(recommend(s, listened, chunk, tt_user, als_user)[0])
    return pl.concat(out)


def per_user(lists: pl.DataFrame, split: str, users: list[int], k: int) -> pl.DataFrame:
    """Each of `users`' own scores in the split, for those with a track to find there."""
    return breakdown(lists, split, k, {})["users"].filter(pl.col("user_id").is_in(users))


def same_top(a: pl.DataFrame, b: pl.DataFrame, k: int) -> float:
    """Share of the users in `a` whose first k tracks are the same, in the same order, in `b`."""
    first = lambda recs: recs.sort(["user_id", "score"], descending=[False, True]).group_by("user_id", maintain_order=True).head(k)  # noqa: E731
    together = first(a).group_by("user_id").agg(pl.col("track_id")).join(
        first(b).group_by("user_id").agg(pl.col("track_id")), on="user_id", how="left", suffix="_b"
    )
    return float((together["track_id"] == together["track_id_b"]).fill_null(False).mean())


def recent(history: pl.DataFrame, n: int) -> pl.DataFrame:
    """Each user's latest stretch of listening that holds `n` distinct tracks, repeats included.

    These are their listens after the last play of their (n+1)-th most recently played
    track. A user with n tracks or fewer keeps every listen. Where that track and the
    one after it were last played in the same second, the user is left with fewer than n.
    """
    last = history.group_by(PAIR).agg(last=pl.col("timestamp").max()).sort(["user_id", "last", "track_id"], descending=[False, True, False])
    cut = last.with_columns(place=pl.int_range(1, pl.len() + 1).over("user_id")).filter(pl.col("place") == n + 1).select("user_id", cut="last")
    kept = history.join(cut, on="user_id", how="left", maintain_order="left")
    return kept.filter(pl.col("cut").is_null() | (pl.col("timestamp") > pl.col("cut"))).drop("cut")


def known_share(lists: pl.DataFrame, history: pl.DataFrame, k: int) -> float:
    """Share of the first k tracks of every list, taken together, that the list's user plays in `history`."""
    first = lists.sort(["user_id", "score"], descending=[False, True]).group_by("user_id", maintain_order=True).head(k)
    return first.join(history.select(PAIR).unique(), on=PAIR, how="semi").height / first.height


def curve_figure(curve: dict, floor: dict, split: str, metric: str, path: Path) -> None:
    """`metric` on `split` against the number of recent tracks shown: both stages, retrieval alone, and the most popular tracks."""
    names = list(curve)
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(floor[split][metric], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(names) - 1, floor[split][metric], "most popular tracks (no listens)", ha="right", va="bottom", fontsize=8, color=MUTED)
    for stages, label, colour in (("retrieval", "retrieval alone", "#c3c2b7"), ("both", "retrieval + ranker", BLUE)):
        values = [curve[name][stages][split][metric] for name in names]
        ax.plot(x, values, color=colour, linewidth=2, marker="o", markersize=4)
        ax.text(len(names) - 1 + 0.15, values[-1], label, va="center", fontsize=8, color=INK if stages == "both" else MUTED)
    ax.set_title(f"{metric.upper().replace('@', ' at ')} on {split} by how much a new listener has played", loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel("most recent tracks the service is shown", fontsize=9, color=MUTED)
    ax.set_xticks(list(x), names)
    ax.set_xlim(-0.3, len(names) + 1.6)
    ax.set_ylim(0, None)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=MUTED, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_curve(curve: dict, floor: dict, users: dict, info: dict, ks: list[int]) -> str:
    """The listens-curve report.

    curve   tracks shown ("1", ..., "all") -> {tracks, known, retrieval: {split: metrics},
            both: {split: metrics}, gap: {split: {metric: {diff, low, high}}}}; `tracks` is the
            median number of distinct tracks a listener was served from, `gap` both stages
            minus the "all" row's
    floor   {split: metrics} for the most popular tracks
    users   {held_out, <split>: listeners scored}
    """
    main, r10, r100 = f"ndcg@{ks[0]}", f"recall@{ks[0]}", f"recall@{ks[2]}"
    L = [
        "# How many listens a new listener needs",
        "",
        "Generated by `python -m m4a_rec.serve curve`. Do not edit by hand; rerun instead.",
        "",
        f"The {users['held_out']:,} listeners of the new-listener test (`reports/serving.md`) were left out of tt_id,",
        "ALS and the population numbers. Here each is served from only their most recent tracks:",
        "the latest stretch of their train-split listening that holds N distinct tracks, repeats",
        "included. A listener with fewer than N tracks is served from all of them. The fold-in",
        f"({info['fold_in']}) and the ranker (`{info['ranker']}`) are the saved ones; nothing was tuned on these numbers.",
        "",
        "- 0: the most popular tracks, what a service can give with nothing to go on",
        "- all: their whole history, the \"fold-in, never seen\" rows of `reports/serving.md`",
        "",
        "Two things to keep in mind. A list is scored after the tracks the listener has already",
        "played are dropped, including plays the service was not shown here. `Already played` is",
        f"the share of the first {ks[0]} tracks, before that, that the listener had played: places the",
        "service would have spent on tracks they know. And the ranker was trained on a snapshot",
        "that held these listeners' earlier listens, so its columns are a little optimistic.",
        "",
        f"`Against all` is {main} of both stages minus the \"all\" row's, with a 95% interval from",
        f"resampling the held-out listeners {info['n_boot']:,} times.",
        "",
    ]
    for split in floor:
        L += [f"## {split} ({users[split]:,} held-out listeners with a track to find)", ""]
        rows = [[0, 0, "", floor[split][main], floor[split][r10], floor[split][r100], "", ""]]
        for name, row in curve.items():
            g = row["gap"][split][main]
            rows.append(
                [
                    name,
                    int(row["tracks"]),
                    row["retrieval"][split][main],
                    row["both"][split][main],
                    row["both"][split][r10],
                    row["both"][split][r100],
                    "" if name == "all" else f"{g['diff']:+.4f} ({g['low']:+.4f} to {g['high']:+.4f})",
                    f"{row['known']:.1%}",
                ]
            )
        L += table(
            ["Tracks shown", "Median tracks", f"{main}, retrieval alone", main, r10, r100, "Against all", "Already played"], rows
        )
    return "\n".join(L)


def timed(s: Service, users: list[int], history: pl.DataFrame, k: int) -> dict:
    """Seconds per stage of single requests, as the API makes them: each user as a known listener, then their listens as someone new's."""
    known, new = [], []
    for i, user in enumerate(users):
        t = {}
        started = perf_counter()
        for_user(s, user, k, explain=True, timings=t)
        t["total"] = perf_counter() - started
        plays = history.filter(pl.col("user_id") == user).select("track_id", "timestamp")
        u = {}
        started = perf_counter()
        for_history(s, plays, k, explain=True, timings=u)
        u["total"] = perf_counter() - started
        if i >= WARM_UP:
            known.append(t)
            new.append(u)
    out = {}
    for name, runs in (("known", known), ("new", new)):
        stages = [stage for stage in [*STAGES, "total"] if stage in runs[0]]
        out[name] = {stage: {q: float(np.quantile([r[stage] for r in runs], p)) for q, p in (("p50", 0.5), ("p95", 0.95))} for stage in stages}
    return out


def render(parity: dict, newcomers: dict, gaps: dict, latency: dict, info: dict, ks: list[int]) -> str:
    metrics = [f"ndcg@{ks[0]}", f"recall@{ks[0]}", *(f"recall@{k}" for k in ks[2:])]
    L = [
        "# Phase 7 serving",
        "",
        "Generated by `python -m m4a_rec.serve report`. Do not edit by hand; rerun instead.",
        "",
        "The service (`python -m m4a_rec.api`) runs both stages at request time from the bundle in",
        "`data/processed/serving/`: the two-tower vectors, ALS and the ranker fitted on both train",
        "splits. Someone the models never saw gets vectors folded in from the listens they hand in:",
        f"the {info['fold_in']}-weighted average direction of their tracks' two-tower vectors, and",
        "ALS's own least-squares solve against its track vectors.",
        "",
        "## The service gives the Phase 6 lists",
        "",
        f"Every one of the {parity['users']:,} known listeners was run through the service and compared with",
        f"the saved lists of `{info['ranker']}` (`data/processed/recs/{info['ranker']}.parquet`). Share of listeners",
        "whose first K tracks are the same, in the same order:",
        "",
    ]
    L += table(["K", "Same"], [[k, f"{share:.2%}"] for k, share in parity["same_top"].items()])
    L += [
        f"The ranker orders the first {parity['n_candidates']}; the rest keep retrieval order. The service takes the",
        "two-tower dot products in numpy, the saved lists in PyTorch, and float32 products differ in their",
        "last bits between the two (and with how many listeners go into one product), so near-ties swap.",
        "",
    ]
    L += table(
        ["", "Saved lists", "Service"],
        [[f"{split} {metric}", parity["saved"][split][metric], parity["service"][split][metric]] for split in ("val", "test") for metric in metrics],
    )
    n = newcomers["users"]
    L += [
        "## Listeners the models never saw",
        "",
        f"{n['held_out']:,} listeners ({info['fraction']:.0%} of them, seed {info['seed']}) were left out of tt_id, ALS and the",
        "population numbers (`python -m m4a_rec.newcomers`, then `python -m m4a_rec.bundle newcomers`), and then",
        "served from their train-split listens alone, as the service would serve someone new. Rows:",
        "",
        "- popularity: the most popular tracks, fitted on both train splits as in the README's table: the floor",
        "- learned vectors: the full models, with the vectors learned for these listeners (what they get as known listeners)",
        "- fold-in, full models: the full models, with vectors folded in from their listens instead",
        "- fold-in, never seen, retrieval only: the models fitted without them, the two-tower order alone",
        "- fold-in, never seen: the models fitted without them, both stages: what someone new gets",
        "",
        "The ranker is the saved one. It was trained on a snapshot that held these listeners' earlier listens",
        "(its labels come from `ranker_train`, not from val or test), so the last row is a little optimistic.",
        "",
    ]
    for split in ("val", "test"):
        L += [f"### {split} ({n[split]:,} held-out listeners with a track to find)", ""]
        L += table(["", *metrics], [[name, *(r[split][m] for m in metrics)] for name, r in newcomers["rows"].items()])
    L += ["Differences on test, with 95% intervals from resampling the held-out listeners:", ""]
    L += table(
        ["", *metrics],
        [[name, *(f"{g[m]['diff']:+.4f} ({g[m]['low']:+.4f} to {g[m]['high']:+.4f})" for m in metrics)] for name, g in gaps.items()],
    )
    L += [
        "## How long a request takes",
        "",
        f"{latency['requests']} single requests with explanations, k = {latency['k']}, after {WARM_UP} not counted; seconds,",
        f"median and 95th percentile. Measured on {info['machine']}. For someone new the listens are the",
        "same listener's, handed in; reading them from the bundle is then not needed.",
        "",
    ]
    for name, title in (("known", "A known listener (GET /users/{id}/recommendations)"), ("new", "Someone new (POST /recommendations)")):
        rows = latency[name]
        L += [title, ""]
        L += table(["", *rows], [[q, *(f"{rows[stage][q]:.3f}" for stage in rows)] for q in ("p50", "p95")])
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv not in (["tune"], ["report"], ["curve"]):
        print(__doc__)
        return 2
    cfg = load_config()
    c, n, ks, processed, reports = cfg["serve"], cfg["newcomers"], cfg["metrics"]["ks"], cfg["paths"]["processed"], cfg["paths"]["reports"]
    say = lambda message: print(message, flush=True)  # noqa: E731
    history = load_train(processed / "splits", TRAIN)
    held = np.load(processed / "models" / "tt_id_newcomers" / "held_out_users.npy").tolist()
    new = load(processed / "serving_newcomers")
    theirs = history.filter(pl.col("user_id").is_in(held))
    metrics = [f"ndcg@{ks[0]}", f"recall@{ks[0]}", *(f"recall@{k}" for k in ks[2:])]
    mean = lambda users: {m: float(users[m].mean()) for m in metrics}  # noqa: E731

    if argv == ["tune"]:
        rows = []
        for fold_in in FOLD_INS:
            say(f"fold-in {fold_in}")
            rows.append([fold_in, *mean(per_user(lists_for(new, theirs, held, fold_in, c["batch_users"]), "val", held, ks[0])).values()])
        say("\n".join(table(["fold-in", *metrics], rows)))
        return 0

    if argv == ["curve"]:
        boot, splits, fold_in = cfg["bootstrap"], ("val", "test"), new.bundle.meta["fold_in"]
        scored = {}
        for name, listens in {**{str(k): recent(theirs, k) for k in n["listens"]}, "all": theirs}.items():
            say(f"{name} tracks")
            lists = {stages: lists_for(new, listens, held, fold_in, c["batch_users"], stages) for stages in ("retrieval", "both")}
            scored[name] = {
                "tracks": listens.group_by("user_id").agg(pl.col("track_id").n_unique())["track_id"].median(),
                "known": known_share(lists["both"], theirs, ks[0]),
                **{stages: {split: per_user(recs, split, held, ks[0]) for split in splits} for stages, recs in lists.items()},
            }
        full = scored["all"]["both"]
        curve = {
            name: {
                "tracks": row["tracks"],
                "known": row["known"],
                **{stages: {split: mean(row[stages][split]) for split in splits} for stages in ("retrieval", "both")},
                "gap": {split: paired_difference(row["both"][split], full[split], metrics, boot["n"], boot["seed"]) for split in splits},
            }
            for name, row in scored.items()
        }
        popular = pl.read_parquet(processed / "recs" / "popularity.parquet").filter(pl.col("user_id").is_in(held))
        floor = {split: mean(per_user(popular, split, held, ks[0])) for split in splits}
        users = {"held_out": len(held), **{split: full[split].height for split in splits}}
        info = {"fold_in": fold_in, "ranker": c["ranker"], "n_boot": boot["n"]}
        curve_figure(curve, floor, "test", metrics[0], reports / "figures" / "listens_curve.png")
        (reports / "listens_curve.json").write_text(stable_json({"users": users, "popularity": floor, "curve": curve}))
        (reports / "listens_curve.md").write_text(render_curve(curve, floor, users, info, ks))
        say(f"wrote {reports / 'listens_curve.md'}")
        return 0

    s = load(processed / "serving")
    say("every known listener through the service")
    service = lists_for(s, history, s.bundle.user_ids.to_list(), None, c["batch_users"])
    saved = pl.read_parquet(processed / "recs" / f"{c['ranker']}.parquet")
    parity = {
        "users": len(s.bundle.user_ids),
        "n_candidates": s.ranker["params"]["n_candidates"],
        "same_top": {str(k): same_top(service, saved, k) for k in (ks[0], s.ranker["params"]["n_candidates"], s.bundle.meta["n_recs"])},
        "saved": {split: score(saved, split) for split in ("val", "test")},
        "service": {split: score(service, split) for split in ("val", "test")},
    }

    say("the held-out listeners")
    lists = {
        "popularity": pl.read_parquet(processed / "recs" / "popularity.parquet").filter(pl.col("user_id").is_in(held)),
        "learned vectors": service.filter(pl.col("user_id").is_in(held)),
        "fold-in, full models": lists_for(s, theirs, held, s.bundle.meta["fold_in"], c["batch_users"]),
        "fold-in, never seen, retrieval only": lists_for(new, theirs, held, new.bundle.meta["fold_in"], c["batch_users"], "retrieval"),
        "fold-in, never seen": lists_for(new, theirs, held, new.bundle.meta["fold_in"], c["batch_users"]),
    }
    users = {name: {split: per_user(recs, split, held, ks[0]) for split in ("val", "test")} for name, recs in lists.items()}
    newcomers = {
        "users": {"held_out": len(held), **{split: users["popularity"][split].height for split in ("val", "test")}},
        "rows": {name: {split: mean(frames[split]) for split in frames} for name, frames in users.items()},
    }
    boot = cfg["bootstrap"]
    final = users["fold-in, never seen"]["test"]
    gaps = {
        f"never seen minus {other}": paired_difference(final, users[other]["test"], metrics, boot["n"], boot["seed"])
        for other in ("learned vectors", "popularity")
    }

    say("timing single requests")
    sample = np.random.default_rng(c["seed"]).choice(s.bundle.user_ids.to_numpy(), size=c["latency_users"] + WARM_UP, replace=False)
    latency = {"requests": c["latency_users"], "k": ks[0], **timed(s, sample.tolist(), history, ks[0])}

    info = {
        "fold_in": s.bundle.meta["fold_in"],
        "ranker": c["ranker"],
        "fraction": n["fraction"],
        "seed": n["seed"],
        "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} cores",
    }
    (reports / "serving.json").write_text(stable_json({"parity": parity, "newcomers": newcomers, "gaps": gaps, "latency": latency}))
    (reports / "serving.md").write_text(render(parity, newcomers, gaps, latency, info, ks))
    say(f"wrote {reports / 'serving.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
