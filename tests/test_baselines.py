from datetime import datetime

import polars as pl
import pytest

from m4a_rec.baselines import (
    MODELS,
    als,
    fit_recommend,
    grid_points,
    item_knn,
    popularity,
    render_report,
    render_tuning,
)
from m4a_rec.evaluate import PAIR, metrics, relevant
from m4a_rec.interactions import build


def ranking(recs, user):
    """(track, score) for one user, best first."""
    mine = recs.filter(pl.col("user_id") == user).sort("score", descending=True)
    return list(zip(mine["track_id"].to_list(), mine["score"].to_list()))


def pop_events():
    # a: users 1, 2, 3 once each, in January.  b: user 1 five times.  c: users 2 and 3.
    # d: user 4, on the last day. So a has the most listeners and b the most plays.
    rows = [(u, "a", datetime(2020, 1, 1)) for u in (1, 2, 3)]
    rows += [(1, "b", datetime(2020, 3, day)) for day in range(1, 6)]
    rows += [(u, "c", datetime(2020, 3, 10)) for u in (2, 3)]
    rows += [(4, "d", datetime(2020, 3, 20))]
    return pl.DataFrame(rows, schema=["user_id", "track_id", "timestamp"], orient="row")


def test_popularity_hand_checked():
    ev = pop_events()
    inter = build(ev, "binary")
    by_users = popularity(ev, inter, n=3, count="users", window_days=None)
    assert ranking(by_users, 4) == [("a", 3), ("c", 2), ("b", 1)]  # user 4 has only heard d
    assert ranking(by_users, 1) == [("c", 2), ("d", 1)]  # a and b are already played
    by_listens = popularity(ev, inter, n=3, count="listens", window_days=None)
    assert ranking(by_listens, 4) == [("b", 5), ("a", 3), ("c", 2)]
    assert ranking(popularity(ev, inter, n=2, count="users", window_days=None), 4) == [("a", 3), ("c", 2)]


def test_popularity_window_counts_recent_plays_only():
    ev = pop_events()
    recent = popularity(ev, build(ev, "binary"), n=3, count="users", window_days=30)
    assert ranking(recent, 4) == [("c", 2), ("b", 1)]  # a was last played in January
    assert sorted(ranking(recent, 2)) == [("b", 1), ("d", 1)]  # tied, so either order


def knn_events():
    # a: users 0, 1, 3.  b: users 0, 1, 2.  c: user 2.  d: user 4.
    played = {0: "ab", 1: "ab", 2: "bc", 3: "a", 4: "d"}
    return pl.DataFrame(
        [(u, t) for u, tracks in played.items() for t in tracks], schema=PAIR, orient="row"
    )


def test_item_knn_hand_checked():
    recs = item_knn(build(knn_events(), "binary"), n=3, k=4)
    # cos(a, b) = 2 / sqrt(3 * 3), cos(b, c) = 1 / sqrt(3 * 1), cos(a, c) = 0
    assert ranking(recs, 3) == [("b", pytest.approx(2 / 3))]
    assert ranking(recs, 0) == [("c", pytest.approx(3**-0.5))]
    assert ranking(recs, 2) == [("a", pytest.approx(2 / 3))]
    assert ranking(recs, 4) == []  # nobody else played d, so there is nothing to recommend


def als_events():
    # Users 0-9 play five of the six A tracks, users 10-19 five of the six B tracks.
    rows = []
    for u in range(20):
        group = "A" if u < 10 else "B"
        rows += [(u, f"{group}{t}") for t in range(6) if t != u % 6]
    return pl.DataFrame(rows, schema=PAIR, orient="row")


def test_als_recommends_within_the_users_group_and_is_seeded():
    kw = dict(n=1, factors=4, regularization=0.01, alpha=10, iterations=15, seed=0)
    inter = build(als_events(), "binary")
    recs = als(inter, **kw)
    for u in range(20):
        group = "A" if u < 10 else "B"
        assert ranking(recs, u)[0][0] == f"{group}{u % 6}"  # the one track of their group they skipped
    assert recs.equals(als(inter, **kw))


CFG = {"n_recs": 20, "seed": 42, "als": {"weighting": "log", "iterations": 5}}
PARAMS = {
    "popularity": {"count": "users", "window_days": 90},
    "item_knn": {"weighting": "binary", "k": 20},
    "als": {"factors": 8, "regularization": 0.01, "alpha": 10},
}


@pytest.mark.parametrize("name", MODELS)
def test_every_model_hands_the_evaluator_what_it_expects(parts, name):
    train = parts["retrieval_train"]
    recs = fit_recommend(name, train, CFG, PARAMS[name])
    assert recs.schema == {"user_id": pl.Int64, "track_id": pl.String, "score": pl.Float32}
    assert recs["score"].is_finite().all()
    assert recs.group_by("user_id").len()["len"].max() <= 20
    assert recs.select(PAIR).is_duplicated().sum() == 0
    assert recs.join(train, on=PAIR, how="semi").height == 0  # nothing the user played in training
    assert set(recs["user_id"].to_list()) <= set(train["user_id"].to_list())

    history = pl.concat([parts["retrieval_train"], parts["ranker_train"]]).select(PAIR)
    m = metrics(recs, relevant(history, parts["val"]), history.unique(), ks=[10], n_catalog=200)
    assert 0 <= m["recall@10"] <= 1 and 0 <= m["ndcg@10"] <= 1
    assert 0 < m["coverage@10"] <= 1


def test_grid_points():
    assert grid_points({"a": [1, 2], "b": [None, 3]}) == [
        {"a": 1, "b": None},
        {"a": 1, "b": 3},
        {"a": 2, "b": None},
        {"a": 2, "b": 3},
    ]


def test_tuning_table_puts_the_best_setting_first():
    m = lambda ndcg: {"ndcg@10": ndcg, "recall@10": 0.1, "recall@20": 0.2, "coverage@10": 0.5, "short@20": 0.0}  # noqa: E731
    rows = [
        {"model": "item_knn", "params": {"k": 50}, "metrics": m(0.01)},
        {"model": "item_knn", "params": {"k": 200}, "metrics": m(0.03)},
    ]
    md = render_tuning(rows, "ndcg@10", ks=[10, 20])
    assert "| k | ndcg@10 | recall@10 | recall@20 | coverage@10 | short@20 |" in md
    assert md.index("| 200 | 0.0300 |") < md.index("| 50 | 0.0100 |")


def test_report_renders():
    one = {"users": 1234, "recall@10": 0.25, "ndcg@10": 0.125, "coverage@10": 0.5, "short@10": 0.0}
    results = {"popularity": {"params": {"count": "users"}, "val": one, "test": one}}
    md = render_report(results, ["retrieval_train"], ks=[10])
    assert "## val (1,234 users scored)" in md
    assert "| popularity | 0.2500 |" in md
    assert '| popularity | {"count": "users"} |' in md
