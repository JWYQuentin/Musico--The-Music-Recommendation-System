"""Everything the Phase 7 service needs in one directory, and the steps that take a listener to the ranker.

The bundle holds the two-tower and ALS vectors fitted on the listens it is built from,
the saved ranker, the content features, artist and album codes, the population numbers
of rank_features.population, the listens themselves (for the listeners the models know)
and titles. The service reads nothing else, so the directory can be mounted into a
container that holds code only. It is written under data/, which git ignores: the titles
come from Music4All and must stay on this machine.

The functions take listeners to the ranker's input, as candidates.py does for the infer
snapshot. Each listener gets two-tower and ALS vectors: the learned ones, or for someone
the models never saw, vectors folded in from their listens. Their candidates are the
unplayed tracks with the highest two-tower scores, and each candidate gets every ranker
feature. tests/test_bundle.py checks the rows against candidates.snapshot.

Nothing here imports torch, faiss, implicit or xgboost, so the service can load it beside
XGBoost and its image needs none of them.

  (no argument)  writes data/processed/serving/ from the models named in configs/serve.yaml
  newcomers      writes data/processed/serving_newcomers/ from the models fitted without the
                 listeners `python -m m4a_rec.newcomers` held out, and without their listens

Usage: python -m m4a_rec.bundle [newcomers]
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import scipy.sparse as sp

from .config import load_config
from .evaluate import PAIR, TRAIN
from .features import BLOCKS, load_blocks
from .interactions import load_train
from .metadata import codes
from .metadata import load as load_tracks
from .rank_features import build as build_features
from .rank_features import mean_direction, pair_matrix, population, scaled

BATCH = 1000  # users scored at a time in top_unseen
ROW_GROUP = 100_000  # listens per row group of history.parquet, so one listener's are found without reading the rest


class Bundle(NamedTuple):
    meta: dict  # now, n_recs, n_candidates, recent_days, short_days, als settings, fold_in, reasons, sources
    track_ids: pl.Series  # sorted; the row order of every per-track array
    user_ids: pl.Series  # sorted; the listeners with learned vectors
    tt_track: np.ndarray
    tt_user: np.ndarray
    als_track: np.ndarray
    als_user: np.ndarray
    blocks: dict[str, np.ndarray]
    artist: np.ndarray
    album: np.ndarray
    pop: dict  # rank_features.population of the listens
    side: dict  # rank_features.scaled of the two-tower track vectors and the blocks
    history: Path  # the listens, sorted by user then time; read with `listens`


def write(
    directory: Path,
    history: pl.DataFrame,
    retrieval_model: Path,
    als_model: Path,
    ranker: Path,
    blocks: dict[str, np.ndarray],
    artist: np.ndarray,
    album: np.ndarray,
    titles: pl.DataFrame,
    settings: dict,
) -> None:
    """The bundle for `history`. Both models must have been fitted on exactly these listens.

    `blocks`, `artist`, `album` and `titles` (track_id, title, artist) are in sorted track
    order. `settings` has n_recs, recent_days, short_days, fold_in and reasons (how many
    the service gives per track); the rest of the meta comes from the model directories.
    """
    track_ids = history["track_id"].unique().sort()
    user_ids = history["user_id"].unique().sort()
    vecs = {}
    for name, model in (("tt", retrieval_model), ("als", als_model)):
        same_users = np.array_equal(np.load(model / "user_ids.npy"), user_ids.to_numpy())
        same_tracks = np.load(model / "track_ids.npy").tolist() == track_ids.to_list()
        if not (same_users and same_tracks):
            raise ValueError(f"{model.name} was not fitted on these listens")
        vecs[name] = (np.load(model / "user_vecs.npy"), np.load(model / "track_vecs.npy"))
    pop = population(history, track_ids, artist, settings["recent_days"], settings["short_days"])
    als = json.loads((als_model / "meta.json").read_text())["params"]
    ranker_meta = json.loads((ranker / "meta.json").read_text())
    meta = {
        **settings,
        "now": pop["now"].isoformat(),
        "n_candidates": ranker_meta["params"]["n_candidates"],
        "als": {key: als[key] for key in ("weighting", "regularization", "alpha")},
        "sources": {"retrieval_model": retrieval_model.name, "als_model": als_model.name, "ranker": ranker.name},
        "listeners": len(user_ids),
        "tracks": len(track_ids),
    }

    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / "tracks.npz",
        track_ids=np.array(track_ids.to_list()),
        tt=vecs["tt"][1],
        als=vecs["als"][1],
        artist=artist,
        album=album,
        artist_users=pop["artist_users"],
        **{f"block_{name}": values for name, values in blocks.items()},
        **pop["track"],  # every key starts with track_
    )
    np.savez(directory / "users.npz", user_ids=user_ids.to_numpy(), tt=vecs["tt"][0], als=vecs["als"][0])
    history.sort(["user_id", "timestamp"]).write_parquet(directory / "history.parquet", row_group_size=ROW_GROUP)
    titles.write_parquet(directory / "titles.parquet")
    (directory / "ranker").mkdir(exist_ok=True)
    for name in ("model.json", "meta.json"):
        shutil.copy(ranker / name, directory / "ranker" / name)
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def load(directory: Path) -> Bundle:
    meta = json.loads((directory / "meta.json").read_text())
    meta["now"] = datetime.fromisoformat(meta["now"])
    with np.load(directory / "tracks.npz") as saved:
        t = {name: saved[name] for name in saved.files}
    with np.load(directory / "users.npz") as saved:
        u = {name: saved[name] for name in saved.files}
    track = {name: values for name, values in t.items() if name.startswith("track_") and name != "track_ids"}
    return Bundle(
        meta=meta,
        track_ids=pl.Series("track_id", t["track_ids"].tolist()),
        user_ids=pl.Series("user_id", u["user_ids"]),
        tt_track=t["tt"],
        tt_user=u["tt"],
        als_track=t["als"],
        als_user=u["als"],
        blocks={name: t[f"block_{name}"] for name in BLOCKS},
        artist=t["artist"],
        album=t["album"],
        pop={"now": meta["now"], "track": track, "artist_users": t["artist_users"]},
        side=scaled(t["tt"], {name: t[f"block_{name}"] for name in BLOCKS}),
        history=directory / "history.parquet",
    )


def listens(b: Bundle, user_ids: list[int]) -> pl.DataFrame:
    """The bundle's listens of these listeners."""
    return pl.scan_parquet(b.history).filter(pl.col("user_id").is_in(user_ids)).collect()


def plays_matrix(history: pl.DataFrame, user_ids: pl.Series, track_ids: pl.Series, weighting: str) -> sp.csr_matrix:
    """users x tracks: 1 for a played track ("binary") or log(1 + plays) ("log"). Tracks not in `track_ids` are left out."""
    plays = history.group_by(PAIR).len(name="plays")
    weight = pl.col("plays").log1p() if weighting == "log" else pl.lit(1.0)
    return pair_matrix(plays.with_columns(w=weight), user_ids, track_ids, values="w")


def fold_in_als(played: sp.csr_matrix, track_vecs: np.ndarray, regularization: float, alpha: float) -> np.ndarray:
    """ALS vectors for listeners the fit never saw: each one's exact least-squares solve against the fitted track vectors.

    `played` holds the values the fit was given; a cell's confidence is alpha times its
    value. This is the solve implicit's AlternatingLeastSquares.recalculate_user does.
    """
    Y = track_vecs.astype(np.float64)
    base = Y.T @ Y + regularization * np.eye(Y.shape[1])
    out = np.zeros((played.shape[0], Y.shape[1]), dtype=np.float32)
    for i in range(played.shape[0]):
        cols = played.indices[played.indptr[i] : played.indptr[i + 1]]
        confidence = alpha * played.data[played.indptr[i] : played.indptr[i + 1]].astype(np.float64)
        y = Y[cols]
        out[i] = np.linalg.solve(base + (y * (confidence - 1)[:, None]).T @ y, y.T @ confidence)
    return out


def vectors(b: Bundle, history: pl.DataFrame, user_ids: pl.Series, fold_in: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Each listener's two-tower and ALS vectors.

    With `fold_in` None, the learned ones (every listener must be in the bundle). With
    "binary" or "log", folded in from their listens in `history`: the two-tower vector is
    the direction of the weighted average of their tracks' vectors, the ALS vector the
    solve in fold_in_als.
    """
    if fold_in is None:
        rows = np.searchsorted(b.user_ids.to_numpy(), user_ids.to_numpy())
        if not np.array_equal(b.user_ids.to_numpy()[rows.clip(max=len(b.user_ids) - 1)], user_ids.to_numpy()):
            raise ValueError("a listener has no learned vectors")
        return b.tt_user[rows], b.als_user[rows]
    als = b.meta["als"]
    tt = mean_direction(plays_matrix(history, user_ids, b.track_ids, fold_in), b.tt_track)
    played = plays_matrix(history, user_ids, b.track_ids, als["weighting"])
    return tt, fold_in_als(played, b.als_track, als["regularization"], als["alpha"])


def top_unseen(user_vecs: np.ndarray, track_vecs: np.ndarray, played: sp.csr_matrix, n: int) -> tuple[np.ndarray, np.ndarray]:
    """For each user, the n best-scoring tracks they have not played, best first: (users, n) indices and scores.

    retrieval.top_unseen without torch. A user with fewer than n tracks left gets index -1
    in the spare places.
    """
    n = min(n, track_vecs.shape[0])
    idx, scores = [], []
    for start in range(0, user_vecs.shape[0], BATCH):
        s = user_vecs[start : start + BATCH] @ track_vecs.T
        rows, cols = played[start : start + BATCH].nonzero()
        s[rows, cols] = -np.inf
        top = np.argpartition(-s, n - 1, axis=1)[:, :n]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(s, top, axis=1), axis=1, kind="stable"), axis=1)
        top_scores = np.take_along_axis(s, top, axis=1)
        idx.append(np.where(np.isneginf(top_scores), -1, top))
        scores.append(top_scores)
    return np.concatenate(idx), np.concatenate(scores)


def retrieve(b: Bundle, history: pl.DataFrame, user_ids: pl.Series, tt_user: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The first stage: each listener's n_recs candidates, as top_unseen gives them."""
    return top_unseen(tt_user, b.tt_track, plays_matrix(history, user_ids, b.track_ids, "binary"), b.meta["n_recs"])


def candidates(
    b: Bundle, history: pl.DataFrame, user_ids: pl.Series, idx: np.ndarray, scores: np.ndarray, als_user: np.ndarray
) -> pl.DataFrame:
    """One row per (listener, candidate) with every feature: what candidates.snapshot gives for these listeners.

    `history` holds their listens only; what needs everyone's comes from the bundle.
    """
    m = b.meta
    features = build_features(
        history, user_ids, b.track_ids, idx, scores, b.tt_track, als_user, b.als_track, b.blocks, b.artist, b.album,
        m["recent_days"], m["short_days"], pop=b.pop, side=b.side,
    )
    return pl.DataFrame(
        {
            "user_id": user_ids.gather(np.repeat(np.arange(idx.shape[0]), idx.shape[1])),
            "track_id": b.track_ids.gather(idx.ravel().clip(0)),
            **{name: values.ravel() for name, values in features.items()},
        }
    ).filter(pl.Series(idx.ravel() >= 0))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv not in ([], ["newcomers"]):
        print(__doc__)
        return 2
    cfg = load_config()
    s, r, processed = cfg["serve"], cfg["ranker"], cfg["paths"]["processed"]
    models = processed / "models"
    history = load_train(processed / "splits", TRAIN)
    tt, als, out = models / s["retrieval_model"], models / s["als_model"], processed / "serving"
    if argv == ["newcomers"]:
        held = np.load(models / "tt_id_newcomers" / "held_out_users.npy")
        history = history.filter(~pl.col("user_id").is_in(held.tolist()))
        tt, als, out = models / "tt_id_newcomers", models / "als_newcomers", processed / "serving_newcomers"
    track_ids = history["track_id"].unique().sort().to_list()
    tracks = load_tracks(processed / "tracks.parquet", track_ids)
    settings = {
        "n_recs": r["n_recs"], "recent_days": r["recent_days"], "short_days": r["short_days"], "fold_in": s["fold_in"], "reasons": s["reasons"]
    }
    write(
        out, history, tt, als, models / s["ranker"], load_blocks(processed / "features.npz", track_ids), *codes(tracks),
        tracks.select("track_id", "title", "artist"), settings,
    )
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out}: {history['user_id'].n_unique():,} listeners, {len(track_ids):,} tracks, {size / 1e6:,.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
