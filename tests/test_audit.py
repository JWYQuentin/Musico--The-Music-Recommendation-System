import bz2
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from m4a_rec.audit import compute, coverage, feature_ids, figures, gini, render


def tiny():
    return pl.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3],
            "track_id": ["a", "a", "b", "a", "c", "a"],
            "timestamp": [datetime(2020, m, 1) for m in (1, 2, 2, 1, 3, 3)],
        }
    )


def test_gini_bounds():
    assert gini(np.array([5, 5, 5, 5])) == pytest.approx(0.0)
    assert gini(np.array([0, 0, 0, 10])) == pytest.approx(0.75)
    assert gini(np.array([])) == 0.0


def test_compute_hand_checked():
    s = compute(tiny())
    assert (s["listens"], s["users"], s["tracks"]) == (6, 3, 3)
    assert s["unique_pairs"] == 5
    assert s["repeat_share"] == pytest.approx(1 / 6)  # only user 1's second play of "a"
    assert s["top1pct_share"] == pytest.approx(4 / 6)  # "a" has 4 of 6 listens
    assert s["density"] == pytest.approx(5 / 9)  # unique pairs, not listens, over 3 x 3
    assert s["_monthly"]["listens"].to_list() == [2, 2, 2]


def test_feature_ids_and_coverage(tmp_path):
    with bz2.open(tmp_path / "f.tsv.bz2", "wt") as f:
        f.write("id\tx\ty\na\t0.1\t0.2\nc\t0.3\t0.4\nzzz\t0\t0\n")
    assert feature_ids(tmp_path / "f.tsv.bz2") == {"a", "c", "zzz"}
    ev = tiny()
    s = compute(ev)
    cov = coverage(ev, s["_per_track"], {"f": tmp_path / "f.tsv.bz2", "g": tmp_path / "nope.bz2"})
    assert cov[0]["track_cov"] == pytest.approx(2 / 3)
    assert cov[0]["listen_cov"] == pytest.approx(5 / 6)
    assert cov[1]["missing"] is True


def test_report_and_figures_render(tmp_path):
    ev = tiny()
    s = compute(ev)
    figures(s, tmp_path)
    assert {p.name for p in tmp_path.glob("*.png")} == {
        "listens_per_user.png",
        "track_popularity.png",
        "listens_per_month.png",
    }
    md = render(s, coverage(ev, s["_per_track"], {}), None)
    assert "| Listens | 6 |" in md
