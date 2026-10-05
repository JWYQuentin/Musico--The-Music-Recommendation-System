"""Training listens as a user-by-track matrix, and model output back as a recommendations frame.

Row i of the matrix is user_ids[i] and column j is track_ids[j]. Both are sorted, so the
mapping depends only on which users and tracks are in the training listens.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import scipy.sparse as sp

from .evaluate import TRAIN


class Interactions(NamedTuple):
    user_ids: pl.Series
    track_ids: pl.Series
    matrix: sp.csr_matrix  # users x tracks, float32


def load_train(splits_dir: Path, names: list[str]) -> pl.DataFrame:
    """The named train splits, concatenated. Refuses any name that is not a train split."""
    if bad := set(names) - set(TRAIN):
        raise ValueError(f"not a train split: {sorted(bad)}")
    return pl.concat([pl.read_parquet(splits_dir / f"{name}.parquet") for name in names])


def build(events: pl.DataFrame, weighting: str) -> Interactions:
    """One cell per (user, track) pair: 1 for "binary", log(1 + plays) for "log"."""
    plays = events.group_by(["user_id", "track_id"]).len(name="plays")
    users = plays.select(pl.col("user_id").unique().sort()).with_row_index("u")
    tracks = plays.select(pl.col("track_id").unique().sort()).with_row_index("t")
    plays = plays.join(users, on="user_id").join(tracks, on="track_id")
    counts = plays["plays"].to_numpy()
    values = {"binary": np.ones(len(counts)), "log": np.log1p(counts)}[weighting]
    matrix = sp.csr_matrix(
        (values, (plays["u"].to_numpy(), plays["t"].to_numpy())),
        shape=(users.height, tracks.height),
        dtype=np.float32,
    )
    return Interactions(users["user_id"], tracks["track_id"], matrix)


def to_frame(inter: Interactions, item_idx: np.ndarray, scores: np.ndarray) -> pl.DataFrame:
    """Recommendations frame from (users, N) arrays of column indices and scores.

    Row i of the arrays belongs to user_ids[i]. Drops padding (index -1) and any track the
    user played in training.
    """
    rows = np.repeat(np.arange(item_idx.shape[0]), item_idx.shape[1])
    cols, scores = item_idx.ravel(), scores.ravel()
    keep = cols >= 0
    keep[keep] = np.asarray(inter.matrix[rows[keep], cols[keep]]).ravel() == 0
    return pl.DataFrame(
        {
            "user_id": inter.user_ids.gather(rows[keep]),
            "track_id": inter.track_ids.gather(cols[keep]),
            "score": scores[keep].astype(np.float32),
        }
    )
