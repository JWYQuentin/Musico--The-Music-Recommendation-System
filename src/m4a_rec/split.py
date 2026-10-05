"""Cut the working subsample into four temporal splits, oldest to newest:

  retrieval_train | ranker_train | val | test

Boundaries are counted back from the last timestamp, in days. Each split is half-open
(start, end]: an event exactly on a boundary belongs to the earlier split, the same
convention prepare.subsample uses for the window.

This is the only module that writes val and test. Only evaluate.py reads them.

Usage: python -m m4a_rec.split
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta

import polars as pl

from .config import load_config

NAMES = ["retrieval_train", "ranker_train", "val", "test"]


def boundaries(end: datetime, test_days: int, val_days: int, ranker_train_days: int) -> dict:
    """Exclusive start of ranker_train, val and test, counted back from `end`."""
    test = end - timedelta(days=test_days)
    val = test - timedelta(days=val_days)
    return {"ranker_train": val - timedelta(days=ranker_train_days), "val": val, "test": test}


def split(events: pl.DataFrame, bounds: dict) -> dict[str, pl.DataFrame]:
    t = pl.col("timestamp")
    r, v, s = bounds["ranker_train"], bounds["val"], bounds["test"]
    return {
        "retrieval_train": events.filter(t <= r),
        "ranker_train": events.filter((t > r) & (t <= v)),
        "val": events.filter((t > v) & (t <= s)),
        "test": events.filter(t > s),
    }


def main() -> int:
    cfg = load_config()
    events = pl.read_parquet(cfg["paths"]["processed"] / "events.parquet")
    end = events["timestamp"].max()
    bounds = boundaries(end, **cfg["split"])
    parts = split(events, bounds)
    out = cfg["paths"]["processed"] / "splits"
    out.mkdir(exist_ok=True)
    for name in NAMES:
        parts[name].write_parquet(out / f"{name}.parquet", compression="zstd")
    meta = {
        "end_inclusive": end.isoformat(),
        "start_exclusive": {k: v.isoformat() for k, v in bounds.items()},
        **cfg["split"],
        "splits": {
            name: {"listens": parts[name].height, "users": parts[name]["user_id"].n_unique()}
            for name in NAMES
        },
    }
    (out / "splits.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
