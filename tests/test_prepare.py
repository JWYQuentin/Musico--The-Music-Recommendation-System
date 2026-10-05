from datetime import datetime

import polars as pl
import pytest

from m4a_rec.prepare import convert, months_before, sniff, subsample

from .conftest import write_events

KW = dict(window_months=13, n_users=20, min_user_listens=50, seed=42)


def test_convert_keeps_every_row(full_parquet, rows):
    df = pl.read_parquet(full_parquet)
    assert df.height == len(rows)
    assert df.columns == ["user_id", "track_id", "timestamp"]
    assert df["timestamp"].dtype == pl.Datetime
    assert df["track_id"].dtype == pl.String
    assert df["timestamp"].max() == max(r[2] for r in rows)


@pytest.mark.parametrize(
    "header,fmt",
    [(False, "%Y-%m-%d %H:%M:%S"), (True, "%Y-%m-%d %H:%M"), (True, "epoch"), (False, "epoch")],
)
def test_convert_handles_format_variants(tmp_path, rows, header, fmt):
    src = tmp_path / "e.tsv.bz2"
    write_events(src, rows[:500], header=header, fmt=fmt)
    info = sniff(src)
    assert info["has_header"] is header
    assert info["epoch"] is (fmt == "epoch")
    assert convert(src, tmp_path / "e.parquet") == 500
    df = pl.read_parquet(tmp_path / "e.parquet")
    assert df["timestamp"].dtype == pl.Datetime
    assert df["timestamp"].min().year >= 2018


def test_sniff_rejects_wrong_shape(tmp_path):
    import bz2

    with bz2.open(tmp_path / "bad.tsv.bz2", "wt") as f:
        f.write("a\tb\n1\t2\n")
    with pytest.raises(ValueError):
        sniff(tmp_path / "bad.tsv.bz2")


def test_months_before():
    assert months_before(datetime(2020, 3, 20), 13) == datetime(2019, 2, 20)
    assert months_before(datetime(2020, 3, 31), 1) == datetime(2020, 2, 29)  # leap-year clamp
    assert months_before(datetime(2020, 1, 15), 1) == datetime(2019, 12, 15)


def test_subsample_respects_window_and_user_rules(full_parquet, tmp_path):
    meta = subsample(full_parquet, tmp_path / "s.parquet", **KW)
    df = pl.read_parquet(tmp_path / "s.parquet")
    start = datetime.fromisoformat(meta["window_start_exclusive"])
    end = datetime.fromisoformat(meta["window_end_inclusive"])

    assert df["timestamp"].min() > start
    assert df["timestamp"].max() <= end
    assert df["user_id"].n_unique() == 20 == meta["users_sampled"]
    assert df.group_by("user_id").len()["len"].min() >= 50
    assert not set(df["user_id"].to_list()) & set(range(10))  # light users never sampled
    assert meta["listens"] == df.height
    # sorted by user, then time: what the sequence models in Phase 4 rely on
    assert df.equals(df.sort(["user_id", "timestamp"]))


def test_subsample_keeps_all_of_a_users_window_listens(full_parquet, tmp_path):
    meta = subsample(full_parquet, tmp_path / "s.parquet", **KW)
    df = pl.read_parquet(tmp_path / "s.parquet")
    start = datetime.fromisoformat(meta["window_start_exclusive"])
    end = datetime.fromisoformat(meta["window_end_inclusive"])
    full = pl.read_parquet(full_parquet).filter(
        (pl.col("timestamp") > start) & (pl.col("timestamp") <= end)
    )
    uid = df["user_id"][0]
    assert df.filter(pl.col("user_id") == uid).height == full.filter(pl.col("user_id") == uid).height


def test_subsample_is_deterministic_and_seed_sensitive(full_parquet, tmp_path):
    subsample(full_parquet, tmp_path / "a.parquet", **KW)
    subsample(full_parquet, tmp_path / "b.parquet", **KW)
    subsample(full_parquet, tmp_path / "c.parquet", **{**KW, "seed": 7})
    a, b, c = (pl.read_parquet(tmp_path / f"{x}.parquet") for x in "abc")
    assert a.equals(b)
    assert set(a["user_id"].to_list()) != set(c["user_id"].to_list())


def test_subsample_explicit_window_end(full_parquet, tmp_path):
    meta = subsample(full_parquet, tmp_path / "s.parquet", **KW, window_end="2019-06-30")
    df = pl.read_parquet(tmp_path / "s.parquet")
    assert df["timestamp"].max() <= datetime(2019, 6, 30)
    assert meta["window_start_exclusive"].startswith("2018-05-30")


def test_subsample_caps_at_eligible_users(full_parquet, tmp_path):
    meta = subsample(full_parquet, tmp_path / "s.parquet", **{**KW, "n_users": 10_000})
    assert meta["users_sampled"] == meta["users_eligible"] <= 50
