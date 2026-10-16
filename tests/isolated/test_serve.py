"""Tests for serve.py. It imports ranker.py, so they run in their own process: see tests/test_ranker.py."""
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from m4a_rec import bundle
from m4a_rec.rank_features import unit
from m4a_rec.serve import NEW, for_history, for_user, lists_for, recommend, same_top, similar


def plays_of(s, user):
    return bundle.listens(s.bundle, [user]).select("track_id", "timestamp")


def test_a_known_listener_gets_the_first_k_of_their_reordered_list(service):
    s, user = service, int(service.bundle.user_ids[4])
    users = pl.Series("user_id", [user])
    history = bundle.listens(s.bundle, [user])
    lists, rows = recommend(s, history, users, *bundle.vectors(s.bundle, history, users, fold_in=None))
    got = for_user(s, user, k=5, explain=False)
    first = lists.sort("score", descending=True).head(5)["track_id"].to_list()
    assert [r["track_id"] for r in got] == first and [r["rank"] for r in got] == [1, 2, 3, 4, 5]
    tt_rank = dict(rows.select("track_id", "tt_rank").rows())
    assert all(r["retrieval_rank"] == tt_rank[r["track_id"]] for r in got)
    assert all(r["title"] == f"Song {s.track_row[r['track_id']]}" and r["reasons"] is None for r in got)
    assert not set(first) & set(history["track_id"].to_list())  # nothing they have played


def test_explain_gives_up_to_three_reasons_in_words(service):
    got = for_user(service, int(service.bundle.user_ids[4]), k=5, explain=True)
    assert all(len(r["reasons"]) <= 3 and all(isinstance(text, str) for text in r["reasons"]) for r in got)
    assert any(r["reasons"] for r in got)


def test_an_unknown_listener_is_a_key_error(service):
    with pytest.raises(KeyError):
        for_user(service, -5, k=5, explain=False)


def test_someone_new_gets_lists_from_their_folded_in_vectors(service):
    s, user = service, int(service.bundle.user_ids[4])
    plays = plays_of(s, user)
    got, unknown = for_history(s, plays, k=5, explain=False)
    history = plays.select(pl.lit(NEW, dtype=pl.Int64).alias("user_id"), "track_id", "timestamp")
    users = pl.Series("user_id", [NEW])
    lists, _ = recommend(s, history, users, *bundle.vectors(s.bundle, history, users, fold_in="log"))
    assert [r["track_id"] for r in got] == lists.sort("score", descending=True).head(5)["track_id"].to_list()
    assert unknown == [] and not {r["track_id"] for r in got} & set(plays["track_id"].to_list())


def test_tracks_not_in_the_catalogue_are_reported_and_left_out(service):
    s = service
    plays = plays_of(s, int(s.bundle.user_ids[4]))
    extra = pl.DataFrame({"track_id": ["nope", "nope", "gone"], "timestamp": [plays["timestamp"][0]] * 3}, schema=plays.schema)
    got, unknown = for_history(s, pl.concat([plays, extra]), k=5, explain=False)
    assert unknown == ["nope", "gone"] and got == for_history(s, plays, k=5, explain=False)[0]
    with pytest.raises(ValueError, match="none of the tracks"):
        for_history(s, extra, k=5, explain=False)


def test_a_listen_without_a_time_is_at_now_and_none_may_be_later(service):
    s, now = service, service.bundle.meta["now"]
    tracks = s.bundle.track_ids.head(3).to_list()
    undated = pl.DataFrame({"track_id": tracks, "timestamp": [None] * 3}, schema={"track_id": pl.String, "timestamp": pl.Datetime("ms")})
    at_now = undated.with_columns(timestamp=pl.lit(now, dtype=pl.Datetime("ms")))
    assert for_history(s, undated, k=5, explain=False) == for_history(s, at_now, k=5, explain=False)
    with pytest.raises(ValueError, match="after"):
        for_history(s, at_now.with_columns(pl.col("timestamp") + timedelta(seconds=1)), k=5, explain=False)


def test_similar_tracks_are_the_nearest_other_vectors(service):
    s = service
    track = s.bundle.track_ids[0]
    got = similar(s, track, k=4)
    vecs = unit(s.bundle.tt_track)
    order = [i for i in np.argsort(-(vecs @ vecs[0])) if i != 0][:4]
    assert [r["track_id"] for r in got] == s.bundle.track_ids.gather(order).to_list()
    assert [r["rank"] for r in got] == [1, 2, 3, 4] and got[0]["title"] == f"Song {order[0]}"
    with pytest.raises(KeyError):
        similar(s, "nope", k=4)


def test_lists_for_many_listeners(service):
    s = service
    users = s.bundle.user_ids.to_list()
    history = bundle.listens(s.bundle, users)
    series = pl.Series("user_id", users)
    tt_user, als_user = bundle.vectors(s.bundle, history, series, fold_in=None)
    assert lists_for(s, history, users, None, batch_users=1000).equals(recommend(s, history, series, tt_user, als_user)[0])
    idx, scores = bundle.retrieve(s.bundle, history, series, tt_user)
    first = lists_for(s, history, users, None, batch_users=1000, stages="retrieval")
    assert first.filter(pl.col("user_id") == users[0])["track_id"].to_list() == s.bundle.track_ids.gather(idx[0]).to_list()
    # in batches of 3 every listener still gets a full list; the order can differ, see test_bundle
    batched = lists_for(s, history, users, "log", batch_users=3)
    assert batched.group_by("user_id").len()["len"].unique().to_list() == [s.bundle.meta["n_recs"]]


def test_same_top_hand_checked():
    a = pl.DataFrame({"user_id": [1, 1, 1, 2, 2], "track_id": ["x", "y", "z", "x", "y"], "score": [3.0, 2.0, 1.0, 2.0, 1.0]})
    b = pl.DataFrame({"user_id": [1, 1, 2, 2], "track_id": ["x", "y", "y", "x"], "score": [2.0, 1.0, 2.0, 1.0]})
    assert same_top(a, b, 2) == 0.5  # user 1's first two agree; user 2's are the same tracks in the other order
    assert same_top(a, b, 1) == 0.5
