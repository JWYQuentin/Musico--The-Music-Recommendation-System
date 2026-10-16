"""Synthetic stand-ins for the Music4All-Onion files, in the same .tsv.bz2 layout."""
from __future__ import annotations

import bz2
from datetime import datetime, timedelta

import numpy as np
import pytest


def make_events(n_users=60, n_tracks=200, seed=0):
    """Rows of (user_id, track_id, datetime) over 2018-01..2020-03, heavy-tailed."""
    rng = np.random.default_rng(seed)
    start = datetime(2018, 1, 1)
    span = int((datetime(2020, 3, 20) - start).total_seconds())
    tracks = [f"{i:016x}" for i in range(n_tracks)]
    weights = 1 / np.arange(1, n_tracks + 1)
    weights /= weights.sum()
    rows = []
    for u in range(n_users):
        k = 5 if u < 10 else int(rng.integers(150, 600))  # users 0-9 are light listeners
        for s, t in zip(rng.integers(0, span, k), rng.choice(n_tracks, k, p=weights)):
            rows.append((u, tracks[t], start + timedelta(seconds=int(s))))
    return rows


def write_events(path, rows, header=True, fmt="%Y-%m-%d %H:%M:%S"):
    with bz2.open(path, "wt") as f:
        if header:
            f.write("user_id\ttrack_id\ttimestamp\n")
        for u, t, ts in rows:
            stamp = str(int(ts.timestamp())) if fmt == "epoch" else ts.strftime(fmt)
            f.write(f"{u}\t{t}\t{stamp}\n")


@pytest.fixture(scope="session")
def rows():
    return make_events()


@pytest.fixture(scope="session")
def full_parquet(tmp_path_factory, rows):
    from m4a_rec.prepare import convert

    d = tmp_path_factory.mktemp("data")
    write_events(d / "events.tsv.bz2", rows)
    convert(d / "events.tsv.bz2", d / "full.parquet")
    return d / "full.parquet"


SPLIT_DAYS = dict(test_days=14, val_days=14, ranker_train_days=28)


@pytest.fixture(scope="session")
def events(tmp_path_factory, full_parquet):
    """The synthetic subsample, as prepare.subsample writes it."""
    import polars as pl

    from m4a_rec.prepare import subsample

    dest = tmp_path_factory.mktemp("processed") / "events.parquet"
    subsample(full_parquet, dest, window_months=13, n_users=50, min_user_listens=50, seed=42)
    return pl.read_parquet(dest)


@pytest.fixture(scope="session")
def parts(events):
    from m4a_rec.split import boundaries, split

    return split(events, boundaries(events["timestamp"].max(), **SPLIT_DAYS))


@pytest.fixture(scope="session")
def served(tmp_path_factory, parts):
    """What a bundle of both synthetic train splits is made from. Random vectors stand in
    for the two-tower model and ALS, saved in their model directories' layout."""
    import polars as pl

    from m4a_rec.baselines import save_als
    from m4a_rec.interactions import build
    from m4a_rec.rank_features import unit

    history = pl.concat([parts["retrieval_train"], parts["ranker_train"]])
    inter = build(history, "log")
    n_users, n_tracks = inter.matrix.shape
    rng = np.random.default_rng(0)
    vecs = lambda rows, dim: rng.standard_normal((rows, dim)).astype(np.float32)  # noqa: E731
    models = tmp_path_factory.mktemp("models")
    out = {"history": history, "inter": inter, "dirs": {}}
    for name, dim in (("tt", 8), ("als", 3)):
        out[name] = (vecs(n_users, dim), vecs(n_tracks, dim))
        if name == "tt":
            out[name] = tuple(unit(v) for v in out[name])  # the towers' vectors have length 1
        out["dirs"][name] = models / name
        save_als(models / name, inter, *out[name], {"weighting": "log", "regularization": 0.1, "alpha": 10})
    lyrics = vecs(n_tracks, 5)
    lyrics[::7] = 0  # every seventh track has no lyrics
    out["blocks"] = {
        "audio": vecs(n_tracks, 4),
        "lyrics": lyrics,
        "has_lyrics": (lyrics != 0).any(axis=1, keepdims=True).astype(np.float32),
        "genres": np.abs(vecs(n_tracks, 6)),
    }
    out["artist"], out["album"] = np.arange(n_tracks) // 4, np.arange(n_tracks) // 2  # four tracks an artist, two an album
    out["titles"] = pl.DataFrame(
        {"track_id": inter.track_ids, "title": [f"Song {i}" for i in range(n_tracks)], "artist": [f"Band {i // 4}" for i in range(n_tracks)]}
    )
    out["settings"] = {"n_recs": 20, "recent_days": 28, "short_days": 7, "fold_in": "log", "reasons": 3}
    return out


@pytest.fixture(scope="session")
def write_bundle(served):
    """Writes the synthetic bundle to a directory, with the ranker saved in `ranker`, and loads it."""
    from m4a_rec.bundle import load, write

    def write_to(directory, ranker):
        s = served
        write(directory, s["history"], s["dirs"]["tt"], s["dirs"]["als"], ranker, s["blocks"], s["artist"], s["album"], s["titles"], s["settings"])
        return load(directory)

    return write_to
