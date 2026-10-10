"""Build the two snapshots the ranker works from.

A snapshot is one moment in time: the listens up to it, a retrieval model and an ALS model
that saw nothing later, each user's candidates from that retrieval model, and the ranker's
features for every candidate.

  train  history is retrieval_train; a candidate's label is 1 if the user plays it in ranker_train
  infer  history is both train splits; no labels, because the period after it is val and test

Both are defined under `snapshots` in configs/ranker.yaml and built by the same code.

Usage: python -m m4a_rec.candidates     # writes data/processed/ranker/{train,infer}.parquet
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

from .baselines import als_vectors
from .config import load_config
from .evaluate import PAIR
from .features import load_blocks
from .interactions import Interactions, build, load_train
from .rank_features import FEATURES
from .rank_features import build as build_features
from .reporting import table
from .retrieval import top_unseen


def snapshot(
    history: pl.DataFrame,
    inter: Interactions,
    tt_user: np.ndarray,
    tt_track: np.ndarray,
    als_user: np.ndarray,
    als_track: np.ndarray,
    blocks: dict[str, np.ndarray],
    n: int,
    recent_days: int,
    short_days: int,
    labels: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per (user, candidate): user_id, track_id, every feature, and `label` if `labels` is given.

    `inter` is `history` as a matrix, and the vectors and `blocks` are in its user and track
    order. Candidates are the `n` tracks the retrieval vectors score highest among those the
    user has not played in `history`. `labels` is the listens of the period being predicted;
    it decides the label and nothing else.
    """
    idx, scores = top_unseen(tt_user, tt_track, inter.matrix, n)
    features = build_features(
        history, inter.user_ids, inter.track_ids, idx, scores, tt_track, als_user, als_track, blocks, recent_days, short_days
    )
    frame = pl.DataFrame(
        {
            "user_id": inter.user_ids.gather(np.repeat(np.arange(idx.shape[0]), idx.shape[1])),
            "track_id": inter.track_ids.gather(idx.ravel().clip(0)),
            **{name: values.ravel() for name, values in features.items()},
        }
    ).filter(pl.Series(idx.ravel() >= 0))  # a user with fewer than n unplayed tracks has padding
    if labels is not None:
        played = labels.select(PAIR).unique().with_columns(label=pl.lit(1, dtype=pl.Int8))
        frame = frame.join(played, on=PAIR, how="left", maintain_order="left").with_columns(pl.col("label").fill_null(0))
    return frame


def main() -> int:
    cfg = load_config()
    r, b, processed = cfg["ranker"], cfg["baselines"], cfg["paths"]["processed"]
    out = processed / "ranker"
    out.mkdir(exist_ok=True)
    means = {}
    for name, snap in r["snapshots"].items():
        history = load_train(processed / "splits", snap["history"])
        inter = build(history, b["als"]["weighting"])
        model = processed / "models" / snap["retrieval_model"]
        same_users = np.array_equal(np.load(model / "user_ids.npy"), inter.user_ids.to_numpy())
        same_tracks = np.load(model / "track_ids.npy").tolist() == inter.track_ids.to_list()
        if not (same_users and same_tracks):
            raise ValueError(f"{model.name} was not trained on {snap['history']}")
        als_user, als_track = als_vectors(inter, iterations=b["als"]["iterations"], seed=b["seed"], **b["als"]["chosen"])
        frame = snapshot(
            history,
            inter,
            np.load(model / "user_vecs.npy"),
            np.load(model / "track_vecs.npy"),
            als_user,
            als_track,
            load_blocks(processed / "features.npz", inter.track_ids.to_list()),
            r["n_recs"],
            r["recent_days"],
            r["short_days"],
            labels=load_train(processed / "splits", [snap["labels"]]) if "labels" in snap else None,
        )
        frame.write_parquet(out / f"{name}.parquet", compression="zstd")
        means[name] = frame.select(pl.col(FEATURES).fill_nan(None).mean()).row(0)
        line = f"{name}: {frame.height:,} rows, {frame['user_id'].n_unique():,} users, history to {history['timestamp'].max()}"
        if "label" in frame.columns:
            with_positive = frame.filter(pl.col("label") == 1)["user_id"].n_unique()
            line += f", {frame['label'].sum():,} positives ({frame['label'].mean():.2%}) over {with_positive:,} users"
        print(line, flush=True)
    # A feature whose average moves a lot between the snapshots would mean something different in use.
    print("\n".join(table(["feature", *means], [[f, *(m[i] for m in means.values())] for i, f in enumerate(FEATURES)])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
