"""Turn the raw listening-events file into a laptop-sized Parquet subsample.

Two steps, so the slow one runs once:

  convert    data/raw/<events>.tsv.bz2 -> data/interim/events_full.parquet
             Streams the bz2 in bounded memory. Slow (bz2 is single-threaded).
  subsample  events_full.parquet -> data/processed/events.parquet
             Keeps the last `window_months`, drops light users, then takes a
             seeded random sample of `n_users`.

Usage: python -m m4a_rec.prepare convert | subsample
"""
from __future__ import annotations

import bz2
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .config import load_config

COLUMNS = ["user_id", "track_id", "timestamp"]
TS_FORMATS = [pacsv.ISO8601, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]


def sniff(path: Path) -> dict:
    """Inspect the first lines: is there a header, and are timestamps epoch numbers?"""
    with bz2.open(path, "rt") as f:
        first = f.readline().rstrip("\n").split("\t")
        second = f.readline().rstrip("\n").split("\t")
    if len(first) != 3:
        raise ValueError(f"expected 3 tab-separated columns, got {len(first)}: {first}")
    has_header = not first[2][:1].isdigit()
    sample = second if has_header else first
    return {"has_header": has_header, "epoch": sample[2].isdigit(), "sample": sample}


def convert(src: Path, dest: Path, block_size: int = 64 << 20) -> int:
    """Stream src (.tsv.bz2) into a zstd Parquet file. Returns rows written."""
    info = sniff(src)
    print(f"sniffed {src.name}: {info}")
    ts_type = pa.int64() if info["epoch"] else pa.timestamp("s")
    reader = pacsv.open_csv(
        pa.CompressedInputStream(pa.OSFile(str(src)), "bz2"),
        read_options=pacsv.ReadOptions(
            column_names=COLUMNS, skip_rows=1 if info["has_header"] else 0, block_size=block_size
        ),
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(
            column_types={"track_id": pa.string(), "timestamp": ts_type},
            timestamp_parsers=TS_FORMATS,
        ),
    )
    rows, writer = 0, None
    tmp = dest.with_suffix(".parquet.tmp")
    try:
        for batch in reader:
            table = pa.Table.from_batches([batch])
            if info["epoch"]:
                ts = table["timestamp"]
                unit = "ms" if info["sample"][2] and len(info["sample"][2]) > 11 else "s"
                table = table.set_column(2, "timestamp", ts.cast(pa.timestamp(unit)).cast(pa.timestamp("s")))
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            rows += table.num_rows
            print(f"\r  {rows:,} rows", end="")
    finally:
        if writer is not None:
            writer.close()
    print()
    tmp.rename(dest)
    return rows


def months_before(end: datetime, months: int) -> datetime:
    """Calendar-month subtraction, clamping the day (31 Mar - 1 month = 28/29 Feb)."""
    total = end.year * 12 + (end.month - 1) - months
    year, month = divmod(total, 12)
    month += 1
    for day in range(end.day, 27, -1):
        try:
            return end.replace(year=year, month=month, day=day)
        except ValueError:
            continue
    return end.replace(year=year, month=month, day=min(end.day, 28))


def subsample(
    src: Path,
    dest: Path,
    window_months: int,
    n_users: int,
    min_user_listens: int,
    seed: int,
    window_end: str | None = None,
) -> dict:
    """Window, filter, and sample users. Returns (and writes) the metadata."""
    lf = pl.scan_parquet(src)
    data_max = lf.select(pl.col("timestamp").max()).collect().item()
    end = datetime.fromisoformat(window_end) if window_end else data_max
    start = months_before(end, window_months)

    # Half-open window (start, end]: an event at exactly `start` is excluded.
    window = lf.filter((pl.col("timestamp") > start) & (pl.col("timestamp") <= end))

    counts = window.group_by("user_id").len().collect()
    eligible = counts.filter(pl.col("len") >= min_user_listens)["user_id"].sort().to_numpy()
    if len(eligible) == 0:
        raise ValueError("no users meet min_user_listens inside the window")
    take = min(n_users, len(eligible))
    # Sorting first makes the draw depend only on the seed, not on row order.
    chosen = np.sort(np.random.default_rng(seed).choice(eligible, size=take, replace=False))

    out = (
        window.filter(pl.col("user_id").is_in(chosen.tolist()))
        .sort(["user_id", "timestamp"])
        .collect(engine="streaming")
    )
    out.write_parquet(dest, compression="zstd")

    meta = {
        "window_start_exclusive": start.isoformat(),
        "window_end_inclusive": end.isoformat(),
        "data_max_timestamp": data_max.isoformat(),
        "window_months": window_months,
        "min_user_listens": min_user_listens,
        "seed": seed,
        "users_in_window": counts.height,
        "users_eligible": int(len(eligible)),
        "users_sampled": int(take),
        "listens": out.height,
        "tracks": out["track_id"].n_unique(),
    }
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] not in {"convert", "subsample"}:
        print(__doc__)
        return 2
    cfg = load_config()
    full = cfg["paths"]["interim"] / "events_full.parquet"
    if argv[0] == "convert":
        rows = convert(cfg["paths"]["raw"] / cfg["events_file"], full)
        print(f"wrote {full} ({rows:,} rows)")
    else:
        meta = subsample(full, cfg["paths"]["processed"] / "events.parquet", **cfg["subsample"])
        print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
