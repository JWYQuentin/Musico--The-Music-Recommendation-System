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


def per_user(
    recs: pl.DataFrame, relevant: pl.DataFrame, seen: pl.DataFrame, ks: list[int]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Each user's own recall@K, ndcg@K and list length n@K, and the ranked lists behind them.

    The first frame has one row per user with a relevant track, sorted by user_id. A user
    with no recommendations scores 0.
    """
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
    users = n_rel.sort("user_id")
    for k in ks:
        top = ranked.filter(pl.col("rank") <= k).group_by("user_id").agg(
            hits=pl.col("hit").sum(),
            dcg=(pl.col("hit") / (pl.col("rank") + 1).log(2)).sum(),
            n=pl.len(),
        )
        u = users.select("user_id", "n_rel").join(top, on="user_id", how="left", maintain_order="left").fill_null(0)
        nrel = u["n_rel"].to_numpy()
        users = users.with_columns(
            pl.Series(f"recall@{k}", u["hits"].to_numpy() / nrel),
            pl.Series(f"ndcg@{k}", u["dcg"].to_numpy() / ideal[np.minimum(k, nrel) - 1]),
            pl.Series(f"n@{k}", u["n"].to_numpy()),
        )
    return users, ranked


def metrics(
    recs: pl.DataFrame, relevant: pl.DataFrame, seen: pl.DataFrame, ks: list[int], n_catalog: int
) -> dict:
    """Recall@K, NDCG@K, coverage@K and short-list share, over users with a relevant track."""
    users, ranked = per_user(recs, relevant, seen, ks)
    out = {"users": users.height}
    for k in ks:
        out[f"recall@{k}"] = float(users[f"recall@{k}"].to_numpy().mean())
        out[f"ndcg@{k}"] = float(users[f"ndcg@{k}"].to_numpy().mean())
        out[f"coverage@{k}"] = ranked.filter(pl.col("rank") <= k)["track_id"].n_unique() / n_catalog
        out[f"short@{k}"] = float((users[f"n@{k}"].to_numpy() < k).mean())
    return out


def paired_difference(a: pl.DataFrame, b: pl.DataFrame, columns: list[str], n_boot: int, seed: int) -> dict:
    """Mean of a minus b over users for each column, with a 95% interval from resampling users.

    `a` and `b` are per-user frames for the same users. Each resample draws users with
    replacement and keeps both models' values for a drawn user together.
    """
    both = a.join(b, on="user_id", suffix="_b")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, both.height, size=(n_boot, both.height))
    out = {}
    for column in columns:
        diff = (both[column] - both[f"{column}_b"]).to_numpy()
        low, high = np.quantile(diff[draws].mean(axis=1), [0.025, 0.975])
        out[column] = {"diff": float(diff.mean()), "low": float(low), "high": float(high)}
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


def difference(recs_a: pl.DataFrame, recs_b: pl.DataFrame, split: str) -> dict:
    """recs_a minus recs_b on val or test, per recall@K and ndcg@K, with a 95% interval each."""
    cfg = load_config()
    rel, seen, _ = truth(cfg["paths"]["processed"] / "splits", split)
    ks, boot = cfg["metrics"]["ks"], cfg["bootstrap"]
    a, _ = per_user(recs_a, rel, seen, ks)
    b, _ = per_user(recs_b, rel, seen, ks)
    columns = [f"{metric}@{k}" for metric in ("recall", "ndcg") for k in ks]
    return paired_difference(a, b, columns, boot["n"], boot["seed"])


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
