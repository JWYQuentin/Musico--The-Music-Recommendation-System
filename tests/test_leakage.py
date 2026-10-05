from pathlib import Path

import polars as pl

import m4a_rec
from m4a_rec.evaluate import PAIR, truth

# Modules allowed to name the full event files or the held-out splits. Model code must
# read the two train splits only and get its scores from evaluate.score.
MAY_READ_EVERYTHING = {"prepare.py", "audit.py", "split.py", "evaluate.py"}
FORBIDDEN = ["events.parquet", "events_full", "val.parquet", "test.parquet"]


def test_only_listed_modules_name_heldout_files():
    for path in Path(m4a_rec.__file__).parent.glob("*.py"):
        if path.name in MAY_READ_EVERYTHING:
            continue
        text = path.read_text()
        assert not [s for s in FORBIDDEN if s in text], f"{path.name} names a held-out file"


def test_history_stops_where_the_period_starts(tmp_path):
    played = {
        "retrieval_train": [(1, "a")],
        "ranker_train": [(1, "b")],
        "val": [(1, "a"), (1, "c")],
        "test": [(1, "c"), (1, "d")],
    }
    for name, rows in played.items():
        pl.DataFrame(rows, schema=PAIR, orient="row").write_parquet(tmp_path / f"{name}.parquet")

    rel, seen, n_catalog = truth(tmp_path, "val")
    assert rel.rows() == [(1, "c")]  # "a" was played in training
    assert sorted(seen.rows()) == [(1, "a"), (1, "b")]  # val's own plays are not history
    assert n_catalog == 2

    rel, seen, n_catalog = truth(tmp_path, "test")
    assert rel.rows() == [(1, "d")]  # "c" was first played in val, so it is not new in test
    assert sorted(seen.rows()) == [(1, "a"), (1, "b"), (1, "c")]
    assert n_catalog == 2  # catalogue is training tracks only, in both periods
