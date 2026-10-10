from datetime import datetime

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from m4a_rec.rank_features import (
    FEATURES,
    GROUPS,
    build,
    first_and_last,
    mean_direction,
    pair_matrix,
    pair_scores,
    rank_rows,
    standardise_rows,
    track_stats,
    user_stats,
)

NOW = datetime(2020, 3, 1)  # the last listen; "recent" is after 2 February, "this week" after 23 February
USERS, TRACKS = pl.Series("user_id", [1, 2, 3]), pl.Series("track_id", ["a", "b", "c"])


def history():
    rows = [
        (1, "a", datetime(2020, 1, 1)),
        (1, "a", datetime(2020, 2, 25)),  # user 1 comes back to a this week
        (1, "b", datetime(2020, 2, 10)),  # user 1 first plays b recently, but not this week
        (2, "a", datetime(2020, 1, 5)),
        (2, "c", datetime(2020, 3, 1)),  # user 2 first plays c today
        (3, "b", datetime(2020, 1, 10)),
    ]
    return pl.DataFrame(rows, schema=["user_id", "track_id", "timestamp"], orient="row")


def close(got, expected):
    return np.allclose(got, np.array(expected, dtype=float), equal_nan=True, atol=1e-6)


def test_row_helpers_hand_checked():
    x = np.array([[1.0, 3.0], [5.0, 5.0]])
    assert standardise_rows(x).tolist() == [[-1, 1], [0, 0]]  # mean 2 and sd 1; a constant row becomes 0
    assert rank_rows(np.array([[0.2, 0.9, 0.5]])).tolist() == [[3, 1, 2]]


def test_pair_scores_hand_checked():
    users = np.array([[1.0, 0.0], [0.0, 2.0]])
    tracks = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
    idx = np.array([[2, 0], [1, 2]])  # user 0's candidates are tracks 2 and 0; user 1's are 1 and 2
    assert pair_scores(users, tracks, idx).tolist() == [[1, 3], [8, 2]]


def test_pair_matrix_and_mean_direction_hand_checked():
    pairs = pl.DataFrame({"user_id": [1, 1, 3], "track_id": ["a", "b", "b"]})
    played = pair_matrix(pairs, USERS, TRACKS)
    assert played.toarray().tolist() == [[1, 1, 0], [0, 0, 0], [0, 1, 0]]
    vecs = np.array([[2.0, 0.0], [0.0, 5.0], [1.0, 0.0]])  # a and c point the same way
    # user 1 played a and b: unit vectors [1, 0] and [0, 1], whose average points along [1, 1]
    assert close(mean_direction(played, vecs), [[0.5**0.5, 0.5**0.5], [0, 0], [0, 1]])


def test_track_stats_hand_checked():
    t = track_stats(first_and_last(history()), TRACKS, NOW, recent_days=28, short_days=7)
    #                                             a          b          c
    assert close(t["track_listeners"], np.log1p([2, 2, 1]))  # a: users 1, 2; b: users 1, 3; c: user 2
    assert close(t["track_listeners_recent"], np.log1p([1, 1, 1]))  # a: user 1; b: user 1; c: user 2
    assert close(t["track_listeners_short"], np.log1p([1, 0, 1]))  # b was last played on 10 February
    assert close(t["track_recent_share"], [1 / 2, 1 / 2, 1])
    assert close(t["track_discovery_rate"], [0, 1, 1])  # a's one recent listener had played it before
    assert close(t["track_pct"], [1, 1, 1 / 3])  # no track has more listeners than a or b


def test_user_stats_hand_checked():
    pct = pl.DataFrame({"track_id": TRACKS, "pct": [1.0, 1.0, 1 / 3]})
    u = user_stats(history(), first_and_last(history()), USERS, pct, NOW, recent_days=28)
    #                                       user 1     user 2     user 3
    assert close(u["user_tracks"], np.log1p([2, 2, 1]))
    assert close(u["user_plays_recent"], np.log1p([2, 1, 0]))
    assert close(u["user_new_share"], [1 / 2, 1, np.nan])  # user 1: b is new, a is not; user 3: nothing recent
    assert close(u["user_mainstream"], [1, 2 / 3, 1])  # user 2 plays a (1) and c (1/3)
    assert close(u["user_days_since_last"], [5, 0, 51])


def build_tiny(hist):
    """One candidate per user: user 1 is offered c, user 2 is offered b, user 3 is offered a."""
    direction = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])  # a and c alike, b different
    blocks = {
        "audio": direction,
        "lyrics": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),  # c has no lyrics
        "genres": direction,
    }
    return build(
        hist,
        USERS,
        TRACKS,
        idx=np.array([[2], [1], [0]]),
        tt_scores=np.array([[0.5], [0.4], [0.3]]),
        tt_track=direction,
        als_user=np.array([[1.0], [2.0], [3.0]]),
        als_track=np.array([[1.0], [1.0], [1.0]]),
        blocks=blocks,
        recent_days=28,
        short_days=7,
    )


def test_build_hand_checked():
    f = build_tiny(history())
    assert list(f) == FEATURES and sorted(FEATURES) == sorted(n for names in GROUPS.values() for n in names)
    assert all(v.shape == (3, 1) and v.dtype == np.float32 for v in f.values())
    col = lambda name: f[name].ravel()  # noqa: E731
    assert close(col("track_listeners"), np.log1p([1, 2, 2]))  # of c, b, a
    assert close(col("user_days_since_last"), [5, 0, 51])
    assert close(col("pop_gap"), [1 / 3 - 1, 1 - 2 / 3, 1 - 1])  # the candidate's share minus the user's usual
    # user 1's recent tracks a and b average to [1, 1] / sqrt(2), and c is [1, 0]; user 2's only
    # recent track is c, at right angles to b; user 3 has played nothing recently
    assert close(col("recent_affinity"), [0.5**0.5, 0, np.nan])
    assert close(col("sim_audio"), [0.5**0.5, 0, 0])  # user 3 has only played b, and a is unlike b
    assert close(col("sim_lyrics"), [np.nan, 0, 0])  # c has no lyrics to compare
    assert close(col("tt_rank"), [1, 1, 1])
    assert close(col("tt_z"), [0, 0, 0]) and close(col("als_z"), [0, 0, 0])  # one candidate each


def test_features_ignore_everything_after_the_history():
    """The same history gives the same features whatever happens next, and a later history changes them."""
    before = build_tiny(history())
    later = pl.concat([history(), pl.DataFrame({"user_id": [3], "track_id": ["a"], "timestamp": [datetime(2020, 3, 9)]})])
    assert all(np.array_equal(before[n], build_tiny(history())[n], equal_nan=True) for n in FEATURES)
    after = build_tiny(later)
    assert not np.array_equal(before["track_listeners"], after["track_listeners"])  # a now has three listeners
