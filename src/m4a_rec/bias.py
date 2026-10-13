"""Who the recommendations serve: popularity bias and genre fairness, for every saved list.

Measured on each user's top K (configs/bias.yaml), once already-played tracks are dropped:

  popularity  tracks are cut into head, middle and tail, each holding a third of training
              listening. How are a model's slots spread over them, how concentrated are
              they, and how many of the head, middle and tail tracks users went on to
              discover does it find?
  listeners   users are cut into thirds by how popular their history is. Is NDCG the same
              for niche and mainstream listeners?
  genres      is a genre's share of slots in line with its share of what users went on to
              discover, and is NDCG the same whatever genre a user mostly plays?

Then one fix: a popularity penalty when the final ranker reorders candidates.

Held-out listens stay inside evaluate.py: this module gets lists without labels, per-user
scores and per-group totals from `evaluate.breakdown`. It imports ranker.py, so like that
module it must not import anything that imports torch.

  tune    the penalty's lambda swept on val. Writes reports/bias_tuning.md and a figure.
  val     every measure on val. Prints; writes nothing.
  report  every measure on test. Writes reports/bias.md and two figures.

Usage: python -m m4a_rec.bias tune | val | report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .audit import GRID, INK, MUTED, SURFACE, gini
from .config import load_config
from .evaluate import TRAIN, breakdown, difference
from .interactions import load_train
from .ranker import OPPONENTS, load, predict, rerank, shown
from .reporting import stable_json, table

RAMP = ["#104281", "#3987e5", "#86b6ef"]  # one hue, darkest for the most popular group
OVER, UNDER, QUIET = "#2a78d6", "#e34948", "#c3c2b7"
NEW = "not in training"  # a track first heard after training: no model can recommend it
SMALL = "smaller genres"
PENALISED = "{} with penalty"


def popularity_groups(listeners: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    """track_id, group and pct for a frame with track_id and listeners.

    Tracks are ranked from most to fewest listeners and cut so that each group holds the
    same share of all listeners: a track goes in the group where its own share starts.
    `pct` is the share of tracks with no more listeners than this one.
    """
    t = listeners.sort(["listeners", "track_id"], descending=[True, False])
    before = (t["listeners"].cum_sum() - t["listeners"]).to_numpy() / t["listeners"].sum()
    idx = np.minimum((before * len(names)).astype(int), len(names) - 1)
    return t.select(
        "track_id",
        group=pl.Series([names[i] for i in idx]),
        pct=pl.col("listeners").rank("max") / t.height,
    )


def listener_groups(mainstream: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    """user_id and group for a frame with user_id and mainstream: equal-sized groups, least mainstream first."""
    place = (mainstream["mainstream"].rank("ordinal") - 1).to_numpy()
    idx = np.minimum(place * len(names) // mainstream.height, len(names) - 1)
    return mainstream.select("user_id", group=pl.Series([names[i] for i in idx]))


def genre_groups(plays: pl.DataFrame, track_genre: pl.DataFrame, min_users: int) -> pl.DataFrame:
    """user_id and group: the genre with most of the user's plays. Genres of under `min_users` users are pooled.

    `plays` has user_id, track_id and plays; `track_genre` has track_id and group.
    """
    per = plays.join(track_genre, on="track_id").group_by(["user_id", "group"]).agg(pl.col("plays").sum())
    top = per.sort(["user_id", "plays", "group"], descending=[False, True, False]).unique("user_id", keep="first", maintain_order=True)
    size = pl.len().over("group")
    return top.select("user_id", group=pl.when(size >= min_users).then(pl.col("group")).otherwise(pl.lit(SMALL)))


def group_means(users: pl.DataFrame, groups: pl.DataFrame, column: str, n_boot: int, seed: int) -> dict:
    """group -> {users, mean, low, high}: the mean of `column` per group of users, with a 95%
    interval from resampling that group's users."""
    rng = np.random.default_rng(seed)
    out = {}
    for (name,), rows in users.join(groups, on="user_id").sort("group").group_by("group", maintain_order=True):
        values = rows[column].to_numpy()
        means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        out[name] = {"users": len(values), "mean": float(values.mean()), "low": float(low), "high": float(high)}
    return out


def exposure(lists: pl.DataFrame, track_groups: pl.DataFrame) -> dict:
    """group -> share of the recommended slots in `lists` that go to tracks of that group."""
    shares = lists.join(track_groups, on="track_id", how="left").group_by("group").len()
    return {group: n / lists.height for group, n in shares.sort("group").rows()}


def concentration(lists: pl.DataFrame, n_catalog: int) -> dict:
    """How the slots are spread over the catalogue: the share of tracks that appear at all,
    and the Gini index of slots per track (0 is an even spread, 1 is one track taking every slot)."""
    slots = np.zeros(n_catalog)
    counts = lists.group_by("track_id").len()["len"].to_numpy()
    slots[: len(counts)] = counts  # the tracks never recommended keep 0
    return {"coverage": len(counts) / n_catalog, "gini": gini(slots)}


def penalised_scores(top: pl.DataFrame, scores: np.ndarray, lam: float) -> np.ndarray:
    """`scores` minus `lam` times popularity, each standardised within the user's list.

    `top` has user_id and track_listeners, one row per score; a list where every value is
    the same contributes 0.
    """
    z = lambda name: ((pl.col(name) - pl.col(name).mean().over("user_id")) / pl.col(name).std().over("user_id")).fill_nan(0.0).fill_null(0.0)  # noqa: E731
    return top.select("user_id", "track_listeners", score=pl.Series(scores)).select(z("score") - lam * z("track_listeners")).to_series().to_numpy()


def pick_lambda(rows: list[dict], metric: str, max_loss: float) -> float:
    """The largest lambda whose `metric` is within `max_loss` (a share) of the unpenalised one."""
    base = next(r[metric] for r in rows if r["lambda"] == 0)
    return max(r["lambda"] for r in rows if r[metric] >= base * (1 - max_loss))


def measure(got: dict, tracks: pl.DataFrame, users: dict[str, pl.DataFrame], k: int, n_catalog: int, boot: dict) -> dict:
    """Every number the report shows for one model. `got` is evaluate.breakdown's result;
    `tracks` has track_id, popularity, pct and genre; `users` maps a grouping's name to user_id and group."""
    lists = got["lists"]
    return {
        "ndcg": float(got["users"][f"ndcg@{k}"].mean()),
        "slots": {name: exposure(lists, tracks.select("track_id", group=name)) for name in ("popularity", "genre")},
        "avg_pct": float(lists.join(tracks, on="track_id")["pct"].mean()),
        **concentration(lists, n_catalog),
        "found": {name: {g: {"relevant": n, "share": s, "found": f} for g, n, s, f in frame.rows()} for name, frame in got["found"].items()},
        "users": {name: group_means(got["users"], groups, f"ndcg@{k}", boot["n"], boot["seed"]) for name, groups in users.items()},
    }


def _style(ax, title: str, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)


def popularity_figure(rows: dict[str, dict], groups: list[str], k: int, path: Path) -> None:
    """One stacked bar per row of `rows` (a name -> group -> share), the most popular group first."""
    names = list(rows)[::-1]  # barh draws from the bottom up
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(names) + 1.5))
    fig.patch.set_facecolor(SURFACE)
    colours = dict(zip(groups, RAMP))
    if any(row.get(NEW, 0) >= 0.005 for row in rows.values()):  # shown only when there is something to see
        colours[NEW] = QUIET
    left = np.zeros(len(names))
    for group, colour in colours.items():
        widths = np.array([rows[name].get(group, 0.0) for name in names])
        ax.barh(names, widths, left=left, height=0.5, color=colour, edgecolor=SURFACE, linewidth=2, label=group)
        for y, (x, w) in enumerate(zip(left, widths)):
            if w >= 0.08:  # a label only where it fits inside the segment
                ax.text(x + w / 2, y, f"{w:.0%}", ha="center", va="center", fontsize=7, color=INK if colour in (RAMP[2], QUIET) else "white")
        left += widths
    _style(ax, f"Where each model's top-{k} slots go, by track popularity", "share of slots")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.22 * 6 / len(names)), ncol=len(colours), frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def genre_figure(ratios: dict[str, float], model: str, k: int, path: Path) -> None:
    """One bar per genre: its share of `model`'s slots divided by its share of discoveries, on a log scale around 1."""
    names = sorted(ratios, key=ratios.get)
    values = np.log2([ratios[name] for name in names])
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(names) + 1.5))
    fig.patch.set_facecolor(SURFACE)
    ax.barh(names, values, height=0.5, color=[OVER if v >= 0 else UNDER for v in values])
    for y, (v, name) in enumerate(zip(values, names)):
        ax.text(v + (0.03 if v >= 0 else -0.03), y, f"{ratios[name]:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7, color=MUTED)
    _style(ax, f"Genre exposure in the top {k} of {model}", "a genre's share of slots divided by its share of discoveries")
    reach = max(1.0, float(np.abs(values).max()) * 1.25)
    ax.set_xlim(-reach, reach)
    ticks = [t for t in (-2, -1, 0, 1, 2) if abs(t) <= reach]
    ax.set_xticks(ticks, [f"{2.0 ** t:g}×" for t in ticks])
    ax.axvline(0, color=MUTED, linewidth=0.8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (UNDER, OVER)]
    ax.legend(handles, ["under-exposed", "over-exposed"], loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def tradeoff_figure(rows: list[dict], chosen: float, k: int, max_loss: float, path: Path) -> None:
    """NDCG against the tail's share of slots as lambda grows, with the rule's floor and the chosen point."""
    x, y = [r["tail"] for r in rows], [r["ndcg"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    fig.patch.set_facecolor(SURFACE)
    ax.plot(x, y, color=OVER, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=2, solid_joinstyle="round")
    floor = y[0] * (1 - max_loss)
    ax.axhline(floor, color=MUTED, linewidth=0.8)
    ax.text(x[-1], floor, f"{max_loss:.0%} below no penalty", ha="right", va="bottom", fontsize=7, color=MUTED)
    for r in rows:
        if r["lambda"] in (rows[0]["lambda"], rows[-1]["lambda"], chosen):
            note = f"lambda {r['lambda']:g}" + (" (chosen)" if r["lambda"] == chosen and chosen else "")
            ax.annotate(note, (r["tail"], r["ndcg"]), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK)
    _style(ax, "The price of showing less popular tracks (val)", f"tail tracks' share of top-{k} slots")
    ax.set_ylabel(f"NDCG@{k}", fontsize=9, color=MUTED)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_tuning(rows: list[dict], chosen: float, model: str, k: int, max_loss: float) -> str:
    L = [
        "# Popularity penalty tuning",
        "",
        "Generated by `python -m m4a_rec.bias tune`. Do not edit by hand; rerun instead.",
        "",
        f"`{model}` on val. A candidate's score is its ranker score minus `lambda` times its popularity,",
        "both standardised within the user's list. Rule fixed in advance: the largest `lambda` that",
        f"costs under {max_loss:.0%} of NDCG@{k}. That is `lambda` = {chosen:g}.",
        "",
        "![The trade-off](figures/bias_tradeoff.png)",
        "",
    ]
    base = rows[0]["ndcg"]
    L += table(
        ["lambda", f"NDCG@{k}", "Change", "Head slots", "Middle slots", "Tail slots", f"Coverage@{k}", "Gini"],
        [[f"{r['lambda']:g}", r["ndcg"], f"{r['ndcg'] / base - 1:+.1%}", f"{r['head']:.1%}", f"{r['middle']:.1%}", f"{r['tail']:.1%}", f"{r['coverage']:.1%}", f"{r['gini']:.3f}"] for r in rows],
    )
    return "\n".join(L)


def render(results: dict, sizes: dict, gap: dict | None, final: str, split: str, k: int, cfg: dict) -> str:
    """The bias report.

    results  model -> the dict `measure` returns
    sizes    {popularity: {group: tracks}, listeners: {group: users}}
    gap      the penalised model minus the final one: {metric: {diff, low, high}}, or None
    """
    models, first = list(results), next(iter(results.values()))
    pop, pct = cfg["popularity_groups"], lambda v: f"{v:.1%}"  # noqa: E731
    L = [
        "# Popularity bias and genre fairness",
        "",
        f"Generated by `python -m m4a_rec.bias {'report' if split == 'test' else split}`. Do not edit by hand; rerun instead.",
        "",
        f"Measured on `{split}`, on each user's top {k} once already-played tracks are dropped.",
        "Every model is fitted on both train splits. A discovery is a track a user played in the",
        "period for the first time.",
        "",
        "## Popularity",
        "",
        "Tracks are ranked by listeners in training and cut so each group holds a third of all",
        "training listening: " + ", ".join(f"{sizes['popularity'][g]:,} {g} tracks" for g in pop) + ".",
        "",
        "![Slots by popularity](figures/bias_popularity.png)",
        "",
    ]
    found = first["found"]["popularity"]
    rows = [["what users went on to discover", *(pct(found[g]["share"]) for g in pop), "", "", ""]]
    rows += [[m, *(pct(r["slots"]["popularity"].get(g, 0)) for g in pop), pct(r["avg_pct"]), pct(r["coverage"]), f"{r['gini']:.3f}"] for m, r in results.items()]
    L += table(["Share of slots", *pop, "Average popularity percentile", f"Coverage@{k}", "Gini of slots"], rows)
    L += [
        f"{pct(found.get('no group', {}).get('share', 0))} of discoveries are tracks that are {NEW}, which no model can recommend.",
        "The Gini index is 0 when slots are spread evenly over the catalogue and 1 when one track takes them all.",
        "",
        f"Share of each group's discoveries found in the top {k}:",
        "",
    ]
    L += table(["Model", *pop], [[m, *(r["found"]["popularity"][g]["found"] for g in pop)] for m, r in results.items()])

    L += ["## Niche and mainstream listeners", ""]
    names = cfg["listener_groups"]
    L += [
        "Users are cut into three equal groups by how popular the tracks in their training history are.",
        f"NDCG@{k} with a 95% interval from resampling the group's users.",
        "",
    ]
    cell = lambda s: f"{s['mean']:.4f} ({s['low']:.4f} to {s['high']:.4f})"  # noqa: E731
    L += table(["Model", *(f"{g} ({sizes['listeners'][g]:,} users)" for g in names)], [[m, *(cell(r["users"]["listeners"][g]) for g in names)] for m, r in results.items()])

    L += ["## Genres", ""]
    L += [
        "A track's genre is the broad group of its first-listed label. The ratio is a genre's share of",
        "slots divided by its share of discoveries: above 1 it is shown more than users' own",
        "discoveries would suggest, below 1 less.",
        "",
        "![Genres](figures/bias_genres.png)",
        "",
    ]
    truth = {g: v["share"] for g, v in first["found"]["genre"].items()}
    genres = sorted(truth, key=lambda g: -truth[g])
    ratio = lambda r, g: f"{r['slots']['genre'].get(g, 0) / truth[g]:.2f}"  # noqa: E731
    L += table(["Genre", "Share of discoveries", *models], [[g, pct(truth[g]), *(ratio(r, g) for r in results.values())] for g in genres])
    L += [f"Share of each genre's discoveries found in the top {k}:", ""]
    L += table(["Genre", *models], [[g, *(r["found"]["genre"][g]["found"] for r in results.values())] for g in genres])
    L += [
        f"NDCG@{k} by the genre a user mostly plays. Genres with under {cfg['min_genre_users']} users are pooled as \"{SMALL}\".",
        "",
    ]
    by_user = results[final]["users"]["genre"]
    fans = sorted(by_user, key=lambda g: -by_user[g]["users"])
    L += table(["Users' main genre", "Users", *models], [[g, by_user[g]["users"], *(r["users"]["genre"][g]["mean"] for r in results.values())] for g in fans])
    L += [f"The same for `{final}` with 95% intervals:", ""]
    L += table(["Users' main genre", "Users", f"NDCG@{k}", "95% interval"], [[g, by_user[g]["users"], by_user[g]["mean"], f"{by_user[g]['low']:.4f} to {by_user[g]['high']:.4f}"] for g in fans])
    L += [
        "With this many groups one will look unusual by chance, and the intervals are not adjusted",
        "for that. Two groups whose intervals overlap are not shown to differ.",
        "",
    ]

    if gap is not None:
        name, lam = PENALISED.format(final), cfg["penalty"]["chosen"]
        a, b = results[name], results[final]
        L += ["## One fix: a popularity penalty", ""]
        L += [
            f"`{final}` with each candidate's score reduced by `lambda` = {lam:g} times its popularity, both",
            "standardised within the user's list. `lambda` was chosen on val (`reports/bias_tuning.md`).",
            "",
        ]
        d = gap[f"ndcg@{k}"]
        rows = [[f"NDCG@{k}", b["ndcg"], a["ndcg"], f"{d['diff']:+.4f} ({d['low']:+.4f} to {d['high']:+.4f})"]]
        rows += [[f"{g} tracks' share of slots", pct(b["slots"]["popularity"].get(g, 0)), pct(a["slots"]["popularity"].get(g, 0)), ""] for g in pop]
        rows += [[f"Coverage@{k}", pct(b["coverage"]), pct(a["coverage"]), ""], ["Gini of slots", f"{b['gini']:.3f}", f"{a['gini']:.3f}", ""]]
        rows += [[f"NDCG@{k}, {g} listeners", b["users"]["listeners"][g]["mean"], a["users"]["listeners"][g]["mean"], ""] for g in names]
        L += table(["", "Without", "With", "Difference (95% interval)"], rows)
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv not in (["tune"], ["val"], ["report"]):
        print(__doc__)
        return 2
    cfg = load_config()
    b, r, reports, processed = cfg["bias"], cfg["ranker"], cfg["paths"]["reports"], cfg["paths"]["processed"]
    k, final = b["k"], r["final"]
    say = lambda message: print(message, flush=True)  # noqa: E731

    # groups of tracks and of users, from training listens and the fixed track table only
    history = load_train(processed / "splits", TRAIN)
    plays = history.group_by(["user_id", "track_id"]).len(name="plays")
    popularity = popularity_groups(plays.group_by("track_id").len(name="listeners"), b["popularity_groups"])
    genre = pl.read_parquet(processed / "tracks.parquet", columns=["track_id", "genre_group"]).rename({"genre_group": "group"})
    tracks = popularity.rename({"group": "popularity"}).join(genre.rename({"group": "genre"}), on="track_id")
    mainstream = plays.join(popularity, on="track_id").group_by("user_id").agg(mainstream=pl.col("pct").mean())
    users = {"listeners": listener_groups(mainstream, b["listener_groups"]), "genre": genre_groups(plays, genre, b["min_genre_users"])}
    track_groups = {"popularity": popularity.select("track_id", "group"), "genre": genre}
    sizes = {
        "popularity": dict(popularity.group_by("group").len().rows()),
        "listeners": dict(users["listeners"].group_by("group").len().rows()),
    }

    def measured(recs: pl.DataFrame, split: str) -> dict:
        return measure(breakdown(recs, split, k, track_groups), tracks, users, k, popularity.height, cfg["bootstrap"])

    infer = pl.read_parquet(processed / "ranker" / "infer.parquet")
    booster, meta = load(processed / "models" / final)
    n = meta["params"]["n_candidates"]
    top, scores = infer.filter(pl.col("tt_rank") <= n), predict(booster, meta, infer)
    with_penalty = lambda lam: rerank(infer, penalised_scores(top, scores, lam), n)  # noqa: E731

    if argv == ["tune"]:
        rows = []
        for lam in b["penalty"]["lambdas"]:
            m = measured(with_penalty(lam), "val")
            rows.append({"lambda": lam, "ndcg": m["ndcg"], **m["slots"]["popularity"], "coverage": m["coverage"], "gini": m["gini"]})
            say(f"lambda {lam:g}: ndcg@{k} {m['ndcg']:.4f}, tail slots {m['slots']['popularity'].get('tail', 0):.1%}, coverage {m['coverage']:.1%}")
        chosen = pick_lambda(rows, "ndcg", b["penalty"]["max_ndcg_loss"])
        (reports / "bias_tuning.json").write_text(stable_json({"rows": rows, "chosen": chosen}))
        tradeoff_figure(rows, chosen, k, b["penalty"]["max_ndcg_loss"], reports / "figures" / "bias_tradeoff.png")
        (reports / "bias_tuning.md").write_text(render_tuning(rows, chosen, shown(final), k, b["penalty"]["max_ndcg_loss"]))
        say(f"the rule picks lambda {chosen:g}; wrote {reports / 'bias_tuning.md'}")
        return 0

    split = "val" if argv == ["val"] else "test"
    label = lambda name: shown(name) if name in r["models"] else name  # noqa: E731
    lists = {label(name): pl.read_parquet(processed / "recs" / f"{name}.parquet") for name in [*OPPONENTS, *r["models"]]}
    gap = None
    if b["penalty"]["chosen"] is not None:
        name = PENALISED.format(shown(final))
        lists[name] = with_penalty(b["penalty"]["chosen"])
        gap = difference(lists[name], lists[shown(final)], split)
    results = {}
    for name, recs in lists.items():
        say(f"measuring {name}")
        results[name] = measured(recs, split)
    text = render(results, sizes, gap, shown(final), split, k, b)
    if argv == ["val"]:
        say(text)
        return 0
    found = results[shown(final)]["found"]
    rows = {"what users went on to discover": {g: v["share"] for g, v in found["popularity"].items()} | {NEW: found["popularity"].get("no group", {}).get("share", 0.0)}}
    rows |= {name: res["slots"]["popularity"] for name, res in results.items()}
    popularity_figure(rows, b["popularity_groups"], k, reports / "figures" / "bias_popularity.png")
    ratios = {g: results[shown(final)]["slots"]["genre"].get(g, 0) / v["share"] for g, v in found["genre"].items()}
    genre_figure(ratios, shown(final), k, reports / "figures" / "bias_genres.png")
    (reports / "bias.json").write_text(stable_json({"split": split, "k": k, "sizes": sizes, "results": results, "gap": gap}))
    (reports / "bias.md").write_text(text)
    say(f"wrote {reports / 'bias.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
