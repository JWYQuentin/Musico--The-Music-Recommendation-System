"""Tests for serve.py. It imports ranker.py, so they run in their own process: see tests/test_ranker.py."""
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from m4a_rec import bundle
from m4a_rec.rank_features import unit
from m4a_rec.serve import NEW, curve_figure, for_history, for_user, known_share, lists_for, recent, recommend, render_curve, same_top, similar


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


def test_recent_hand_checked():
    at = lambda *seconds: [datetime(2020, 1, 1, 0, 0, sec) for sec in seconds]  # noqa: E731
    history = pl.DataFrame(
        {
            # user 1 plays a, b, a, c, b: most recently played first, that is b (5), c (4), a (3)
            "user_id": [1, 1, 1, 1, 1, 2, 2],
            "track_id": ["a", "b", "a", "c", "b", "x", "y"],
            "timestamp": at(1, 2, 3, 4, 5, 1, 2),
        }
    )
    kept = lambda n, user: recent(history, n).filter(pl.col("user_id") == user).select("track_id", "timestamp").rows()  # noqa: E731
    assert kept(1, 1) == [("b", at(5)[0])]  # after c's last play
    assert kept(2, 1) == [("c", at(4)[0]), ("b", at(5)[0])]  # after a's last play: b's earlier play is not part of the stretch
    assert kept(3, 1) == history.filter(pl.col("user_id") == 1).select("track_id", "timestamp").rows() == kept(9, 1)
    assert kept(1, 2) == [("y", at(2)[0])] and len(kept(2, 2)) == 2  # each user is cut on their own
    assert recent(history, 2).schema == history.schema


def test_known_share_hand_checked():
    lists = pl.DataFrame(
        {"user_id": [1, 1, 1, 2, 2, 2], "track_id": ["a", "b", "c", "a", "b", "c"], "score": [3.0, 2.0, 1.0, 1.0, 2.0, 3.0]}
    )
    history = pl.DataFrame({"user_id": [1, 1, 2, 2], "track_id": ["a", "a", "a", "c"]})
    assert known_share(lists, history, 1) == 1.0  # user 1's first is a, user 2's is c: both played
    assert known_share(lists, history, 2) == 0.5  # then b for each, which neither has played
    assert known_share(lists, history, 3) == 0.5  # 3 of 6: user 2's a is their third


def test_curve_report_and_figure_render(tmp_path):
    m = lambda v: {"ndcg@10": v, "recall@10": 0.02, "recall@100": 0.1, "recall@500": 0.3}  # noqa: E731
    gap = lambda d: {name: {"diff": d, "low": d - 0.004, "high": d + 0.004} for name in m(0)}  # noqa: E731
    row = lambda tracks, both, known, d: {  # noqa: E731
        "tracks": float(tracks), "known": known, "retrieval": {"test": m(both - 0.005)}, "both": {"test": m(both)}, "gap": {"test": gap(d)}
    }
    curve = {"1": row(1, 0.02, 0.4, -0.013), "20": row(20, 0.031, 0.25, -0.002), "all": row(270, 0.033, 0.0, 0.0)}
    floor = {"test": m(0.006)}
    md = render_curve(curve, floor, {"held_out": 1435, "test": 1014}, {"fold_in": "binary", "ranker": "ranker_artist", "n_boot": 2000}, [10, 20, 100, 500])
    assert "python -m m4a_rec.serve curve" in md and "## test (1,014 held-out listeners with a track to find)" in md
    assert "| Tracks shown | Median tracks | ndcg@10, retrieval alone | ndcg@10 | recall@10 | recall@100 | Against all | Already played |" in md
    assert "| 0 | 0 |  | 0.0060 | 0.0200 | 0.1000 |  |  |" in md  # the most popular tracks
    assert "| 1 | 1 | 0.0150 | 0.0200 | 0.0200 | 0.1000 | -0.0130 (-0.0170 to -0.0090) | 40.0% |" in md
    assert "| all | 270 | 0.0280 | 0.0330 | 0.0200 | 0.1000 |  | 0.0% |" in md
    curve_figure(curve, floor, "test", "ndcg@10", tmp_path / "curve.png")
    assert (tmp_path / "curve.png").stat().st_size > 0
