import numpy as np
import polars as pl
import pytest

from m4a_rec.evaluate import PAIR, metrics, only, paired_difference, per_user, relevant

NO_SEEN = pl.DataFrame({"user_id": [], "track_id": []}, schema={"user_id": pl.Int64, "track_id": pl.String})


def pairs(rows):
    return pl.DataFrame(rows, schema=PAIR, orient="row")


def recs(rows):
    return pl.DataFrame(rows, schema=[*PAIR, "score"], orient="row")


# u1 wants {a, b} and is shown a, x, b; u2 wants {c} and is shown y, c. Catalogue of 6.
RELEVANT = pairs([(1, "a"), (1, "b"), (2, "c")])
RECS = recs([(1, "a", 0.9), (1, "x", 0.8), (1, "b", 0.7), (2, "y", 0.9), (2, "c", 0.8)])


def test_relevant_is_new_to_the_user():
    history = pairs([(1, "a"), (2, "b")])
    holdout = pairs([(1, "a"), (1, "b"), (1, "b")])  # "a" is a repeat; "b" is played twice
    assert relevant(history, holdout).rows() == [(1, "b")]


def test_metrics_hand_checked():
    m = metrics(RECS, RELEVANT, NO_SEEN, ks=[2], n_catalog=6)
    ideal_2 = 1 + 1 / np.log2(3)  # two hits in the top two places = 1.6309
    assert m["users"] == 2
    assert m["recall@2"] == pytest.approx((1 / 2 + 1) / 2)  # u1 finds 1 of 2, u2 finds 1 of 1
    assert m["ndcg@2"] == pytest.approx((1 / ideal_2 + 1 / np.log2(3)) / 2)  # 0.6131 and 0.6309
    assert m["ndcg@2"] == pytest.approx(0.6220, abs=1e-4)
    assert m["coverage@2"] == pytest.approx(4 / 6)  # a, x, y, c
    assert m["short@2"] == 0.0


def test_already_played_tracks_are_dropped_before_the_cut():
    seen = pairs([(1, "s")])
    with_seen_first = pl.concat([recs([(1, "s", 1.0)]), RECS])
    assert metrics(with_seen_first, RELEVANT, seen, [2], 6) == metrics(RECS, RELEVANT, NO_SEEN, [2], 6)
    # without the filter, "s" would take first place and push "a" to rank 2
    assert metrics(with_seen_first, RELEVANT, NO_SEEN, [2], 6)["ndcg@2"] < 0.6


def test_user_without_recommendations_scores_zero():
    m = metrics(RECS.filter(pl.col("user_id") == 2), RELEVANT, NO_SEEN, ks=[2], n_catalog=6)
    assert m["users"] == 2
    assert m["recall@2"] == pytest.approx(1 / 2)  # u1 scores 0, u2 scores 1
    assert m["short@2"] == pytest.approx(1 / 2)  # u1 has fewer than 2 recommendations


def test_duplicate_recommendations_count_once():
    doubled = pl.concat([RECS, recs([(2, "c", 0.95)])])  # "c" twice for u2
    m = metrics(doubled, RELEVANT, NO_SEEN, ks=[2], n_catalog=6)
    assert m["recall@2"] == pytest.approx((1 / 2 + 1) / 2)


def test_perfect_scores_one_and_random_scores_near_zero():
    rng = np.random.default_rng(0)
    catalog = np.array([f"t{i:05d}" for i in range(10_000)])
    rel = pairs([(u, t) for u in range(50) for t in rng.choice(catalog, 5, replace=False)])
    perfect = metrics(rel.with_columns(score=pl.lit(1.0)), rel, NO_SEEN, ks=[10], n_catalog=10_000)
    assert perfect["recall@10"] == pytest.approx(1.0)
    assert perfect["ndcg@10"] == pytest.approx(1.0)
    random = recs(
        [(u, t, float(s)) for u in range(50) for t, s in zip(rng.choice(catalog, 10, replace=False), rng.random(10))]
    )
    m = metrics(random, rel, NO_SEEN, ks=[10], n_catalog=10_000)
    assert m["recall@10"] < 0.02  # expected 10 / 10,000 = 0.001


def test_only_some_tracks_count():
    # Keep b and c. u1 now wants {b} and is shown b; u2 wants {c} and is shown c.
    recs_, rel, n_catalog = only(RECS, RELEVANT, ["b", "c", "z"])
    assert recs_["track_id"].to_list() == ["b", "c"]
    assert rel.rows() == [(1, "b"), (2, "c")]
    m = metrics(recs_, rel, NO_SEEN, ks=[2], n_catalog=n_catalog)
    assert m["recall@2"] == pytest.approx(1.0)
    assert m["ndcg@2"] == pytest.approx(1.0)  # each is now first in its list
    assert m["coverage@2"] == pytest.approx(2 / 3)  # b and c out of b, c, z


def test_per_user_hand_checked():
    users, ranked = per_user(RECS, RELEVANT, NO_SEEN, ks=[2])
    assert users["user_id"].to_list() == [1, 2]
    assert users["recall@2"].to_list() == pytest.approx([1 / 2, 1])  # u1 finds a but not b; u2 finds c
    ideal_2 = 1 + 1 / np.log2(3)
    assert users["ndcg@2"].to_list() == pytest.approx([1 / ideal_2, 1 / np.log2(3)])  # a is first; c is second
    assert users["n@2"].to_list() == [2, 2]
    assert ranked.filter(pl.col("rank") <= 2).height == 4
    # the means of these columns are what metrics() reports
    m = metrics(RECS, RELEVANT, NO_SEEN, ks=[2], n_catalog=6)
    assert m["recall@2"] == pytest.approx(users["recall@2"].mean())
    assert m["ndcg@2"] == pytest.approx(users["ndcg@2"].mean())


def frame(values):
    return pl.DataFrame({"user_id": list(range(len(values))), "ndcg@10": values})


def test_paired_difference_hand_checked():
    # a beats b by exactly 0.5 for every user, so every resample gives 0.5
    same_gap = paired_difference(frame([1.0, 0.5, 0.75]), frame([0.5, 0.0, 0.25]), ["ndcg@10"], n_boot=200, seed=0)
    assert same_gap["ndcg@10"] == pytest.approx({"diff": 0.5, "low": 0.5, "high": 0.5})
    # a wins by 1 for half the users and loses by 1 for the other half: no difference on average
    mixed = paired_difference(frame([1.0, 0.0] * 50), frame([0.0, 1.0] * 50), ["ndcg@10"], n_boot=2000, seed=0)
    assert mixed["ndcg@10"]["diff"] == pytest.approx(0.0)
    assert mixed["ndcg@10"]["low"] < 0 < mixed["ndcg@10"]["high"]
    # 100 users with standard deviation 1: the 95% interval is about 1.96 / sqrt(100) = 0.196 either side
    assert mixed["ndcg@10"]["high"] == pytest.approx(0.196, abs=0.04)
    again = paired_difference(frame([1.0, 0.0] * 50), frame([0.0, 1.0] * 50), ["ndcg@10"], n_boot=2000, seed=0)
    assert again == mixed  # seeded
