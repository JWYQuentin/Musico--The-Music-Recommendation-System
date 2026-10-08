"""Score recommendations against held-out listens. The only module that reads val and test.

A track is relevant for a user in a period if the user plays it in that period and did
not play it earlier in the window. Recommended tracks the user has already played are
dropped before the list is cut at K, so every model is filtered the same way.

Recommendations are a Polars frame with user_id, track_id and score (higher is better).
Hand in more than K tracks per user: some may be dropped as already played.

Usage: python -m m4a_rec.evaluate     # prints the ground-truth summary for val and test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from .config import load_config

PAIR = ["user_id", "track_id"]
TRAIN = ["retrieval_train", "ranker_train"]
HISTORY = {"val": TRAIN, "test": [*TRAIN, "val"]}  # what a user has played before each period


def relevant(history: pl.DataFrame, holdout: pl.DataFrame) -> pl.DataFrame:
    """Unique (user, track) pairs in `holdout` that are absent from `history`."""
    return holdout.select(PAIR).unique().join(history.select(PAIR).unique(), on=PAIR, how="anti")


def metrics(
    recs: pl.DataFrame, relevant: pl.DataFrame, seen: pl.DataFrame, ks: list[int], n_catalog: int
) -> dict:
    """Recall@K, NDCG@K, coverage@K and short-list share, over users with a relevant track."""
    n_rel = relevant.group_by("user_id").len(name="n_rel")
    ranked = (
        recs.join(n_rel, on="user_id", how="semi")
        .join(seen, on=PAIR, how="anti")
        .sort(["user_id", "score", "track_id"], descending=[False, True, False])
        .unique(subset=PAIR, keep="first", maintain_order=True)
        .with_columns(rank=pl.int_range(1, pl.len() + 1).over("user_id"))
        .filter(pl.col("rank") <= max(ks))
        .join(relevant.with_columns(hit=pl.lit(1.0)), on=PAIR, how="left")
        .with_columns(pl.col("hit").fill_null(0.0))
    )
    # ideal[m - 1] is the DCG of m hits in the top m places
    ideal = np.cumsum(1 / np.log2(np.arange(2, max(ks) + 2)))
    out = {"users": n_rel.height}
    for k in ks:
        top = ranked.filter(pl.col("rank") <= k)
        per_user = top.group_by("user_id").agg(
            hits=pl.col("hit").sum(),
            dcg=(pl.col("hit") / (pl.col("rank") + 1).log(2)).sum(),
            n=pl.len(),
        )
        u = n_rel.join(per_user, on="user_id", how="left").fill_null(0)  # no recs scores 0
        nrel = u["n_rel"].to_numpy()
        out[f"recall@{k}"] = float((u["hits"].to_numpy() / nrel).mean())
        out[f"ndcg@{k}"] = float((u["dcg"].to_numpy() / ideal[np.minimum(k, nrel) - 1]).mean())
        out[f"coverage@{k}"] = top["track_id"].n_unique() / n_catalog
        out[f"short@{k}"] = float((u["n"].to_numpy() < k).mean())
    return out


def truth(splits_dir: Path, split: str) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Relevant pairs, already-played pairs, and training-catalogue size for val or test."""
    load = lambda name: pl.read_parquet(splits_dir / f"{name}.parquet", columns=PAIR)  # noqa: E731
    history = pl.concat([load(name) for name in HISTORY[split]])
    n_catalog = pl.concat([load(name) for name in TRAIN])["track_id"].n_unique()
    return relevant(history, load(split)), history.unique(), n_catalog


def only(recs: pl.DataFrame, rel: pl.DataFrame, tracks: list[str]) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Recommendations, relevant pairs and catalogue size when only `tracks` count."""
    keep = pl.col("track_id").is_in(tracks)
    return recs.filter(keep), rel.filter(keep), len(tracks)


def score(recs: pl.DataFrame, split: str, only_tracks: list[str] | None = None) -> dict:
    """Metrics on val or test. With `only_tracks`, as if those were the only tracks in the catalogue."""
    cfg = load_config()
    rel, seen, n_catalog = truth(cfg["paths"]["processed"] / "splits", split)
    if only_tracks is not None:
        recs, rel, n_catalog = only(recs, rel, only_tracks)
    return metrics(recs, rel, seen, cfg["metrics"]["ks"], n_catalog)


def main() -> int:
    cfg = load_config()
    summary = {}
    for split in HISTORY:
        rel, _, n_catalog = truth(cfg["paths"]["processed"] / "splits", split)
        per_user = rel.group_by("user_id").len()["len"]
        summary[split] = {
            "users_evaluated": per_user.len(),
            "relevant_pairs": rel.height,
            "relevant_per_user_median": per_user.median(),
            "relevant_tracks": rel["track_id"].n_unique(),
            "training_catalogue": n_catalog,
        }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
