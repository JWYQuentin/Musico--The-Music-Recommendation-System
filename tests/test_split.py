from datetime import datetime, timedelta

import polars as pl

from m4a_rec.split import NAMES, boundaries, split

from .conftest import SPLIT_DAYS

END = datetime(2020, 3, 20, 12, 59, 51)
BOUNDS = {
    "ranker_train": datetime(2020, 1, 24, 12, 59, 51),
    "val": datetime(2020, 2, 21, 12, 59, 51),
    "test": datetime(2020, 3, 6, 12, 59, 51),
}


def test_boundaries_count_back_from_end():
    assert boundaries(END, **SPLIT_DAYS) == BOUNDS  # 14, 14 and 28 days; February 2020 has 29


def test_boundary_event_goes_to_earlier_split():
    sec = timedelta(seconds=1)
    r, v, s = BOUNDS["ranker_train"], BOUNDS["val"], BOUNDS["test"]
    events = pl.DataFrame(
        {
            "user_id": [1] * 6,
            "track_id": ["on_r", "after_r", "on_v", "after_v", "on_s", "after_s"],
            "timestamp": [r, r + sec, v, v + sec, s, s + sec],
        }
    )
    got = {name: part["track_id"].to_list() for name, part in split(events, BOUNDS).items()}
    assert got == {
        "retrieval_train": ["on_r"],
        "ranker_train": ["after_r", "on_v"],
        "val": ["after_v", "on_s"],
        "test": ["after_s"],
    }


def test_splits_partition_events_in_time_order(events, parts):
    assert list(parts) == NAMES
    assert all(parts[name].height > 0 for name in NAMES)
    assert sum(parts[name].height for name in NAMES) == events.height  # nothing lost or doubled
    for earlier, later in zip(NAMES, NAMES[1:]):
        assert parts[earlier]["timestamp"].max() < parts[later]["timestamp"].min()
    days = lambda name: parts[name]["timestamp"].max() - parts[name]["timestamp"].min()  # noqa: E731
    assert days("test") <= timedelta(days=14) and days("val") <= timedelta(days=14)
    assert days("ranker_train") <= timedelta(days=28)
