"""Build the demo: one self-contained HTML page for exploring what the system recommends.

For a sample of listeners the page shows what they played recently, the final model's
top K with the ranker's reasons and how far it moved each track from its retrieval
position, every model's top K side by side, and for any track its nearest tracks in the
two-tower vector space. A recommendation the listener went on to play is marked.

The page holds track titles and artists from the base Music4All dataset, which is shared
for research only and must not be redistributed. It is written under data/, which git
ignores, and must stay out of the repository.

It imports ranker.py, so like that module it must not import anything that imports torch.

Usage: python -m m4a_rec.demo     # writes data/processed/demo/index.html
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

from .bias import genre_groups, listener_groups, popularity_groups
from .config import load_config
from .evaluate import PAIR, TRAIN, hits
from .interactions import load_train
from .ranker import OPPONENTS, load, shap_values, shown

TEMPLATE = Path(__file__).with_name("demo.html")
SLOT = "/*DATA*/"  # where the page's data goes in the template
USER_FEATURE = "how this listener listens in general"


def count(value: float) -> int:
    """A feature stored as log(1 + n), back as n."""
    return round(math.expm1(value))


def plural(n: int, noun: str) -> str:
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


def reason(feature: str, value: float, recent_days: int, short_days: int) -> str:
    """One feature's value as a plain statement about the listener or the track."""
    missing = value != value
    if feature.startswith("user_"):
        return USER_FEATURE
    if missing:
        return {
            "artist_days_since": "has never played this artist",
            "recent_affinity": f"has played nothing in the last {recent_days} days to compare it with",
            "sim_lyrics": "it has no lyrics on file",
        }.get(feature, f"no value for {feature}")
    return {
        "tt_rank": lambda: f"the retrieval model's number {int(value)} candidate for them",
        "tt_z": lambda: f"its retrieval score is {value:+.1f} standard deviations from their average candidate",
        "als_rank": lambda: f"a second model, ALS, puts it at number {int(value)} of their candidates",
        "als_z": lambda: f"its ALS score is {value:+.1f} standard deviations from their average candidate",
        "track_listeners": lambda: f"{plural(count(value), 'listener')} in all",
        "track_listeners_recent": lambda: f"{plural(count(value), 'listener')} in the last {recent_days} days",
        "track_listeners_short": lambda: f"{plural(count(value), 'listener')} in the last {short_days} days",
        "track_recent_share": lambda: f"{value:.0%} of its listeners played it in the last {recent_days} days",
        "track_discovery_rate": lambda: f"{value:.0%} of its recent listeners were hearing it for the first time",
        "recent_affinity": lambda: f"close to what they played in the last {recent_days} days",
        "sim_audio": lambda: "it sounds like what they have played",
        "sim_lyrics": lambda: "its lyrics are close to what they have played",
        "sim_genres": lambda: "its genres match what they have played",
        "pop_gap": lambda: ("more" if value > 0 else "less") + " popular than what they usually play",
        "artist_tracks": lambda: f"has played {plural(count(value), 'other track')} by this artist",
        "artist_play_share": lambda: f"{value:.0%} of their plays are by this artist",
        "artist_days_since": lambda: f"last played this artist {plural(round(value), 'day')} ago" if round(value) else "played this artist within the last day",
        "artist_catalog_share": lambda: f"has played {value:.0%} of this artist's tracks",
        "album_tracks": lambda: f"has played {plural(count(value), 'track')} from this album",
        "artist_listeners": lambda: f"{plural(count(value), 'listener')} play this artist",
    }[feature]()


def reasons(features: list[str], values: np.ndarray, contributions: np.ndarray, n: int, recent_days: int, short_days: int) -> list[str]:
    """The n features that push a candidate's score up the most, as statements; none if nothing pushes it up."""
    out = []
    for i in np.argsort(-contributions):
        if contributions[i] <= 0 or len(out) == n:
            break
        text = reason(features[i], float(values[i]), recent_days, short_days)
        if text not in out:  # the user features share one statement
            out.append(text)
    return out


def recent_tracks(history: pl.DataFrame, users: list[int], n: int) -> dict[int, list[tuple[str, str]]]:
    """user -> their n most recently played distinct tracks, latest first, as (track_id, date)."""
    last = history.filter(pl.col("user_id").is_in(users)).group_by(PAIR).agg(pl.col("timestamp").max())
    last = last.sort(["user_id", "timestamp", "track_id"], descending=[False, True, False]).group_by("user_id", maintain_order=True).head(n)
    out: dict[int, list] = {u: [] for u in users}
    for user, track, when in last.rows():
        out[user].append((track, when.strftime("%-d %b %Y")))
    return out


def embed(payload: dict) -> str:
    """The data as JSON that is safe inside a <script> element: no character that could end it early."""
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def page(payload: dict, template: str) -> str:
    assert template.count(SLOT) == 1
    return template.replace(SLOT, embed(payload))


def main() -> int:
    cfg = load_config()
    d, r, processed = cfg["demo"], cfg["ranker"], cfg["paths"]["processed"]
    k, split, final = d["k"], d["split"], r["final"]
    names = [*OPPONENTS, *r["models"]]
    lists = {name: pl.read_parquet(processed / "recs" / f"{name}.parquet") for name in names}

    # the listeners: a seeded sample of those the period is scored on
    scored = hits(lists[final], split, k, lists[final]["user_id"].unique().to_list())["user_id"].unique().sort().to_numpy()
    users = sorted(np.random.default_rng(d["seed"]).choice(scored, size=min(d["n_users"], len(scored)), replace=False).tolist())
    shown_lists = {name: hits(recs, split, k, users) for name, recs in lists.items()}

    history = load_train(processed / "splits", TRAIN)
    plays = history.group_by(PAIR).len(name="plays")
    popularity = popularity_groups(plays.group_by("track_id").len(name="listeners"), cfg["bias"]["popularity_groups"])
    table = pl.read_parquet(processed / "tracks.parquet")
    genre = table.select("track_id", group="genre_group")
    mainstream = plays.join(popularity, on="track_id").group_by("user_id").agg(mainstream=pl.col("pct").mean())
    listener = dict(listener_groups(mainstream, cfg["bias"]["listener_groups"]).rows())
    main_genre = dict(genre_groups(plays, genre, cfg["bias"]["min_genre_users"]).rows())
    recent = recent_tracks(history, users, d["recent"])
    mine = plays.filter(pl.col("user_id").is_in(users)).join(table.select("track_id", "artist"), on="track_id")
    totals = {u: (n, p) for u, n, p in mine.group_by("user_id").agg(pl.len(), pl.col("plays").sum()).rows()}
    by_artist = mine.group_by(["user_id", "artist"]).agg(pl.col("plays").sum()).sort(["user_id", "plays", "artist"], descending=[False, True, False])
    artists = {u: [] for u in users}
    for user, artist, n in by_artist.group_by("user_id", maintain_order=True).head(d["top_artists"]).rows():
        artists[user].append([artist, n])

    # the final model's reasons for each track in its top k
    booster, meta = load(processed / "models" / final)
    top = shown_lists[final].select(PAIR)
    rows = pl.read_parquet(processed / "ranker" / "infer.parquet").join(top, on=PAIR, how="semi")
    contributions = shap_values(booster, rows, meta["features"], meta["trees"])[:, :-1]
    values = rows.select(meta["features"]).to_numpy()
    why = {}
    for i, (user, track, tt_rank) in enumerate(rows.select(*PAIR, "tt_rank").rows()):
        said = reasons(meta["features"], values[i], contributions[i], d["reasons"], r["recent_days"], r["short_days"])
        why[(user, track)] = {"from": int(tt_rank), "reasons": said}

    # every track the page names: those in a list or a listener's recent plays, and their neighbours
    primary = sorted({t for frame in shown_lists.values() for t in frame["track_id"].to_list()} | {t for played in recent.values() for t, _ in played})
    model = processed / "models" / d["neighbours_model"]
    vec_ids = np.load(model / "track_ids.npy").tolist()
    row_of = {t: i for i, t in enumerate(vec_ids)}
    near = np.load(model / "neighbours.npy")[:, : d["neighbours"]]
    similar = {t: [vec_ids[j] for j in near[row_of[t]]] for t in primary if t in row_of}
    every = sorted(set(primary) | {t for ts in similar.values() for t in ts})
    number = {t: i for i, t in enumerate(every)}
    details = pl.DataFrame({"track_id": every}).join(table, on="track_id", how="left").join(popularity, on="track_id", how="left")
    listeners = dict(plays.group_by("track_id").len().rows())

    payload = {
        "k": k,
        "models": [shown(name) if name in r["models"] else name for name in names],
        "final": names.index(final),
        "tracks": [[title, artist, album, year, g, group or "", listeners.get(t, 0)] for t, title, artist, album, year, g, group in details.select("track_id", "title", "artist", "album", "year", "genre_group", "group").rows()],
        "similar": {str(number[t]): [number[s] for s in ts] for t, ts in similar.items()},
        "users": [],
    }
    for user in users:
        per_model = {}
        for name in names:
            rows = shown_lists[name].filter(pl.col("user_id") == user).sort("rank")
            per_model[name] = rows.select("track_id", "hit").rows()
        payload["users"].append(
            {
                "id": user,
                "tracks": totals[user][0],
                "plays": totals[user][1],
                "listener": listener[user],
                "genre": main_genre[user],
                "artists": artists[user],
                "recent": [[number[t], when] for t, when in recent[user]],
                "lists": [[[number[t], int(hit)] for t, hit in per_model[name]] for name in names],
                "why": {str(number[t]): why[(user, t)] for t, _ in per_model[final] if (user, t) in why},
            }
        )
    out = processed / "demo"
    out.mkdir(exist_ok=True)
    text = page(payload, TEMPLATE.read_text())
    (out / "index.html").write_text(text)
    print(f"wrote {out / 'index.html'}: {len(users)} listeners, {len(every):,} tracks, {len(text) / 1e6:.1f} MB. Keep it out of git.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
