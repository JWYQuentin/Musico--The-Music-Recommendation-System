"""Tests for bias.py. It imports ranker.py, so they run in their own process: see tests/test_ranker.py."""
import numpy as np
import polars as pl
import pytest

from m4a_rec.bias import (
    SMALL,
    concentration,
    exposure,
    genre_figure,
    genre_groups,
    group_means,
    listener_groups,
    measure,
    penalised_scores,
    pick_lambda,
    popularity_figure,
    popularity_groups,
    render,
    render_tuning,
    tradeoff_figure,
)

CFG = {
    "popularity_groups": ["head", "middle", "tail"],
    "listener_groups": ["niche", "mainstream"],
    "min_genre_users": 2,
    "penalty": {"chosen": 0.2},
}


def test_popularity_groups_hand_checked():
    listeners = pl.DataFrame({"track_id": ["a", "b", "c", "d"], "listeners": [6, 3, 2, 1]})  # 12 in all
    got = popularity_groups(listeners, ["head", "middle", "tail"])
    # a's share starts at 0, b's at 6/12, c's at 9/12, d's at 11/12: thirds are cut at 4/12 and 8/12
    assert got.rows() == [("a", "head", 1.0), ("b", "middle", 0.75), ("c", "tail", 0.5), ("d", "tail", 0.25)]


def test_listener_groups_are_equal_sized_and_ordered():
    m = pl.DataFrame({"user_id": [1, 2, 3, 4, 5, 6], "mainstream": [0.1, 0.9, 0.5, 0.3, 0.7, 0.2]})
    got = dict(listener_groups(m, ["niche", "middle", "mainstream"]).rows())
    assert got == {1: "niche", 6: "niche", 4: "middle", 3: "middle", 5: "mainstream", 2: "mainstream"}


def test_genre_groups_take_the_most_played_genre_and_pool_small_ones():
    plays = pl.DataFrame({"user_id": [1, 1, 2, 3, 4], "track_id": ["r", "p", "r", "r", "j"], "plays": [5, 2, 1, 1, 9]})
    genre = pl.DataFrame({"track_id": ["r", "p", "j"], "group": ["rock", "pop", "jazz"]})
    assert dict(genre_groups(plays, genre, min_users=2).rows()) == {1: "rock", 2: "rock", 3: "rock", 4: SMALL}  # one jazz fan
    assert dict(genre_groups(plays, genre, min_users=1).rows())[4] == "jazz"


def test_group_means_hand_checked():
    users = pl.DataFrame({"user_id": [1, 2, 3, 4], "ndcg@10": [0.0, 1.0, 0.5, 0.5]})
    groups = pl.DataFrame({"user_id": [1, 2, 3, 4], "group": ["a", "a", "b", "b"]})
    got = group_means(users, groups, "ndcg@10", n_boot=500, seed=0)
    assert got["a"]["users"] == 2 and got["a"]["mean"] == 0.5 and (got["a"]["low"], got["a"]["high"]) == (0.0, 1.0)
    assert got["b"] == {"users": 2, "mean": 0.5, "low": 0.5, "high": 0.5}  # both users score the same


def test_exposure_and_concentration_hand_checked():
    lists = pl.DataFrame({"user_id": [1, 1, 2, 2], "track_id": ["a", "b", "a", "c"], "rank": [1, 2, 1, 2]})
    groups = pl.DataFrame({"track_id": ["a", "b", "c", "d"], "group": ["head", "tail", "tail", "tail"]})
    assert exposure(lists, groups) == {"head": 0.5, "tail": 0.5}  # a takes two of the four slots
    c = concentration(lists, n_catalog=4)
    assert c["coverage"] == 0.75  # a, b and c appear; d never does
    # slots per track 0, 1, 1, 2: Gini = sum of |x_i - x_j| over all pairs / (2 n^2 mean) = 12 / 32
    assert c["gini"] == pytest.approx(0.375)
    assert concentration(lists.filter(pl.col("track_id") == "a"), 4)["gini"] == pytest.approx(0.75)  # one track takes all


def test_penalised_scores_hand_checked():
    top = pl.DataFrame({"user_id": [1, 1, 1, 2], "track_listeners": [3.0, 2.0, 1.0, 5.0]})
    scores = np.array([3.0, 2.0, 1.0, 7.0])  # user 1: the more popular, the higher the score
    assert penalised_scores(top, scores, 0.0).tolist() == [1, 0, -1, 0]  # standardised; user 2's single candidate is 0
    assert penalised_scores(top, scores, 1.0).tolist() == [0, 0, 0, 0]  # the penalty cancels the score exactly
    assert penalised_scores(top, scores, 2.0).tolist() == [-1, 0, 1, 0]  # and now the least popular comes first


def test_pick_lambda_takes_the_largest_within_the_allowed_loss():
    rows = [{"lambda": 0.0, "ndcg": 0.100}, {"lambda": 0.1, "ndcg": 0.099}, {"lambda": 0.2, "ndcg": 0.0981}, {"lambda": 0.5, "ndcg": 0.090}]
    assert pick_lambda(rows, "ndcg", max_loss=0.02) == 0.2  # 0.0981 is within 2% of 0.100, 0.090 is not
    assert pick_lambda(rows, "ndcg", max_loss=0.0) == 0.0


def breakdown_result():
    """Two users; user 1 is shown a then b, user 2 is shown a then c."""
    found = lambda rows: pl.DataFrame(rows, schema=["group", "relevant", "share", "found"], orient="row")  # noqa: E731
    return {
        "lists": pl.DataFrame({"user_id": [1, 1, 2, 2], "track_id": ["a", "b", "a", "c"], "rank": [1, 2, 1, 2]}),
        "users": pl.DataFrame({"user_id": [1, 2], "ndcg@2": [1.0, 0.5]}),
        "found": {
            "popularity": found([("head", 1, 0.25, 1.0), ("middle", 1, 0.25, 0.0), ("no group", 1, 0.25, 0.0), ("tail", 1, 0.25, 1.0)]),
            "genre": found([("pop", 2, 0.5, 0.5), ("rock", 2, 0.5, 0.5)]),
        },
    }


def measured():
    tracks = pl.DataFrame({"track_id": ["a", "b", "c", "d"], "popularity": ["head", "middle", "tail", "tail"], "pct": [1.0, 0.75, 0.5, 0.25], "genre": ["rock", "rock", "pop", "pop"]})
    users = {"listeners": pl.DataFrame({"user_id": [1, 2], "group": ["niche", "mainstream"]}), "genre": pl.DataFrame({"user_id": [1, 2], "group": ["rock", "rock"]})}
    return measure(breakdown_result(), tracks, users, k=2, n_catalog=4, boot={"n": 50, "seed": 0})


def test_measure_hand_checked():
    m = measured()
    assert m["ndcg"] == 0.75
    assert m["slots"]["popularity"] == {"head": 0.5, "middle": 0.25, "tail": 0.25}
    assert m["slots"]["genre"] == {"pop": 0.25, "rock": 0.75}
    assert m["avg_pct"] == pytest.approx((1.0 + 0.75 + 1.0 + 0.5) / 4)
    assert m["coverage"] == 0.75 and m["found"]["popularity"]["tail"] == {"relevant": 1, "share": 0.25, "found": 1.0}
    assert m["users"]["listeners"]["niche"]["mean"] == 1.0 and m["users"]["genre"]["rock"]["users"] == 2


def test_report_and_figures_render(tmp_path):
    m = measured()
    worse = {**m, "ndcg": 0.7}
    results = {"tt_id": m, "tt_id + ranker_artist": m, "tt_id + ranker_artist with penalty": worse}
    sizes = {"popularity": {"head": 1, "middle": 1, "tail": 2}, "listeners": {"niche": 1, "mainstream": 1}}
    gap = {"ndcg@2": {"diff": -0.05, "low": -0.08, "high": -0.02}}
    md = render(results, sizes, gap, "tt_id + ranker_artist", "test", 2, CFG)
    assert "1 head tracks, 1 middle tracks, 2 tail tracks" in md
    assert "| what users went on to discover | 25.0% | 25.0% | 25.0% |" in md
    assert "| tt_id | 50.0% | 25.0% | 25.0% | 81.2% | 75.0% | 0.375 |" in md  # slots, average percentile, coverage, Gini
    assert "25.0% of discoveries are tracks that are not in training" in md
    assert "| tt_id | 1.0000 (1.0000 to 1.0000) | 0.5000 (0.5000 to 0.5000) |" in md  # niche, then mainstream
    assert "| rock | 50.0% | 1.50 | 1.50 | 1.50 |" in md  # 75% of slots against 50% of discoveries
    assert "| NDCG@2 | 0.7500 | 0.7000 | -0.0500 (-0.0800 to -0.0200) |" in md
    assert "One fix" not in render({"tt_id": m, "tt_id + ranker_artist": m}, sizes, None, "tt_id + ranker_artist", "val", 2, CFG)

    rows = [{"lambda": 0.0, "ndcg": 0.04, "head": 0.5, "middle": 0.3, "tail": 0.2, "coverage": 0.3, "gini": 0.9}, {"lambda": 0.5, "ndcg": 0.03, "head": 0.3, "middle": 0.3, "tail": 0.4, "coverage": 0.5, "gini": 0.8}]
    assert "| 0.5 | 0.0300 | -25.0% | 30.0% | 30.0% | 40.0% | 50.0% | 0.800 |" in render_tuning(rows, 0.0, "tt_id + ranker_artist", 10, 0.02)
    tradeoff_figure(rows, 0.0, 10, 0.02, tmp_path / "t.png")
    popularity_figure({"discovered": {"head": 0.3, "middle": 0.3, "tail": 0.3, "not in training": 0.1}, "tt_id": m["slots"]["popularity"]}, CFG["popularity_groups"], 10, tmp_path / "p.png")
    genre_figure({"rock": 1.5, "pop": 0.5}, "tt_id + ranker_artist", 10, tmp_path / "g.png")
    assert all((tmp_path / name).stat().st_size > 0 for name in ("t.png", "p.png", "g.png"))
