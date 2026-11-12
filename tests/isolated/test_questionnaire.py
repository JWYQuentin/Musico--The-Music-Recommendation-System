"""Tests for questionnaire.py. It imports serve.py, so they run in their own process: see tests/test_ranker.py."""
from datetime import datetime

import numpy as np
import polars as pl
import scipy.sparse as sp

from m4a_rec import bundle
from m4a_rec.evaluate import PAIR
from m4a_rec.questionnaire import (
    favourites,
    form,
    genre_seeds,
    look_alikes,
    most_listened,
    named,
    read_sheet,
    render,
    seed_tracks,
    strongest,
    taste,
    thirds,
    vector_lists,
)
from m4a_rec.rank_features import unit

NOW = datetime(2020, 1, 1)


def test_taste_hand_checked():
    played = sp.csr_matrix(np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32))  # users x tracks
    genres = np.array([[1, 0], [0.5, 0.5], [0, 1]], dtype=np.float32)  # tracks x genres
    profile, mainstream = taste(played, genres, np.array([0.2, 0.4, 1.0]))
    assert profile.tolist() == [[1.5, 0.5], [0, 1], [0, 0]]  # user 0: tracks 0 and 1 added up
    assert np.allclose(mainstream, [0.3, 1.0, 0.0])  # the mean of 0.2 and 0.4; a user with no track gets 0


def test_strongest_hand_checked():
    profile = np.array([[0.5, 2.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0]])
    assert strongest(profile, 1).tolist() == [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]  # the second listener's tie goes to the lower column
    assert strongest(profile, 2).tolist() == [[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]]  # the third has one genre and names one
    assert strongest(profile, 9).tolist() == (profile > 0).astype(float).tolist()


def test_thirds_hand_checked():
    known = np.arange(1.0, 10.0)  # 1 to 9: the cut points are 3.67 and 6.33
    assert thirds(np.array([1, 3.6, 3.7, 6.3, 6.4, 9]), known).tolist() == [0, 0, 1, 1, 2, 2]
    assert np.bincount(thirds(known, known)).tolist() == [3, 3, 3]
    assert thirds(np.array([3.0, 6.0]), np.array([0.0, 3.0, 6.0, 9.0])).tolist() == [0, 1]  # on a cut point: the lower third


def test_look_alikes_hand_checked():
    known_profiles = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    known_vectors = np.eye(3, dtype=np.float32)  # known listener i's vector points along axis i
    answers = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    assert look_alikes(answers, known_profiles, known_vectors, 1).tolist() == np.eye(3).tolist()  # each answer's one closest listener
    h = 0.5**0.5
    # two closest: answer 0 gets listeners 0 and 2, answer 1 gets 1 and 2, and answer 2 gets 2 and,
    # of the tied 0 and 1, the earlier; each result is the average, at length 1
    assert np.allclose(look_alikes(answers, known_profiles, known_vectors, 2), [[h, 0, h], [0, h, h], [h, 0, h]])
    # answers 0 and 1 may only use listeners 1 and 2, answer 2 only listener 0
    bands = (np.array([1, 1, 0]), np.array([0, 1, 1]))
    assert look_alikes(answers, known_profiles, known_vectors, 1, bands).tolist() == [[0, 0, 1], [0, 1, 0], [1, 0, 0]]


def test_answers_become_seed_tracks_hand_checked():
    track_ids, artist = pl.Series("track_id", ["a", "b", "c", "d", "e"]), np.array([0, 0, 1, 1, 2])
    history = pl.DataFrame(
        {
            # listener 1: artist 0 three times, artist 1 once, artist 2 three times. Listener 2: artist 1, and a track not in the catalogue
            "user_id": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
            "track_id": ["a", "a", "b", "c", "e", "e", "e", "d", "d", "gone"],
        }
    )
    assert favourites(history, track_ids, artist, 1).rows() == [(1, 0), (2, 1)]  # the tie of artists 0 and 2 goes to the lower code
    picks = favourites(history, track_ids, artist, 2)
    assert picks.rows() == [(1, 0), (1, 2), (2, 1)]

    # tracks a to d by listeners; e is in no group
    ranked = most_listened(np.array([0, 0, 1, 1, -1]), np.array([5.0, 9.0, 2.0, 2.0, 7.0]), track_ids)
    assert ranked.rows() == [(0, "b", 1), (0, "a", 2), (1, "c", 1), (1, "d", 2)]  # c and d tie: the lower track_id first

    one = seed_tracks(picks, ranked, 1, NOW)
    assert one.rows() == [(1, "b", NOW), (2, "c", NOW)]  # listener 1's second artist has no ranked track
    assert seed_tracks(picks, ranked, 2, NOW).select(PAIR).rows() == [(1, "b"), (1, "a"), (2, "c"), (2, "d")]
    assert one.schema == {"user_id": pl.Int64, "track_id": pl.String, "timestamp": pl.Datetime("ms")}

    said = named(np.array([[0, 1, 1], [0, 0, 0], [1, 0, 0]]), pl.Series("user_id", [7, 8, 9]))
    assert said.rows() == [(7, 1), (7, 2), (9, 0)] and said.schema == picks.schema


def test_report_renders():
    m = lambda v: {"ndcg@10": v, "recall@10": 0.02, "recall@100": 0.1, "recall@500": 0.3}  # noqa: E731
    gap = lambda d: {"test": {name: {"diff": d, "low": d - 0.004, "high": d + 0.004} for name in m(0)}}  # noqa: E731
    rows = {
        "3 genres + mainstream": {"questions": 4, "stages": "retrieval", "retrieval": {"test": m(0.012)}, "gap": gap(-0.004)},
        "3 artists": {"questions": 3, "stages": "both", "retrieval": {"test": m(0.014)}, "both": {"test": m(0.018)}, "gap": gap(0.002)},
        "1 recent track": {"questions": None, "stages": "both", "retrieval": {"test": m(0.013)}, "both": {"test": m(0.016)}, "gap": gap(0.0)},
    }
    info = {
        "fold_in": "binary", "ranker": "ranker_artist", "n_boot": 2000, "against": "1 recent track", "genres": 685,
        "chosen": {"neighbours": 20, "tracks_per_artist": 1, "tracks_per_genre": 3},
    }
    md = render(rows, {"test": m(0.006)}, {"held_out": 1435, "test": 1014}, info, [10, 20, 100, 500])
    assert "python -m m4a_rec.questionnaire report" in md and "## test (1,014 held-out listeners with a track to find)" in md
    assert "the 20 known listeners" in md and "each named artist's 1 most-listened track," in md and "each genre's 3 most-listened tracks." in md
    assert "| Form | Questions | Stages | ndcg@10 | recall@10 | recall@100 | Against 1 recent track | ndcg@10, retrieval alone |" in md
    assert "| nothing: the most popular tracks | 0 |  | 0.0060 | 0.0200 | 0.1000 |  |  |" in md
    assert "| 3 genres + mainstream | 4 | retrieval | 0.0120 | 0.0200 | 0.1000 | -0.0040 (-0.0080 to +0.0000) | 0.0120 |" in md
    assert "| 3 artists | 3 | both | 0.0180 | 0.0200 | 0.1000 | +0.0020 (-0.0020 to +0.0060) | 0.0140 |" in md
    assert "| 1 recent track |  | both | 0.0160 | 0.0200 | 0.1000 |  | 0.0130 |" in md  # the row the others are compared with


def test_every_form_gives_every_listener_a_full_list(service):
    s, b = service, service.bundle
    users = b.user_ids.head(6)  # asked as if new; the bundle also knows them
    sheet = read_sheet(b, bundle.listens(b, users.to_list()), users)
    n = b.meta["n_recs"]
    full = lambda made: all(recs.group_by("user_id").len()["len"].to_list() == [n] * 6 for recs in made.values())  # noqa: E731
    assert sheet.profile.shape == (6, 6) and set(sheet.band.tolist()) <= {0, 1, 2} and len(sheet.known_band) == len(b.user_ids)

    # asked for their whole profile, each is their own closest known listener and gets the vector learned for them
    assert np.allclose(look_alikes(sheet.profile, sheet.known_profile, b.tt_user, 1), unit(b.tt_user[:6]), atol=1e-6)
    genres = form(s, sheet, genres=2, mainstream=True, neighbours=3)
    assert list(genres) == ["retrieval"] and full(genres)  # no listens, so no ranker
    look = look_alikes(strongest(sheet.profile, 2), sheet.known_profile, b.tt_user, 3, (sheet.band, sheet.known_band))
    first = genres["retrieval"].filter(pl.col("user_id") == users[0])["score"].to_numpy()
    # the n best scores over every track, so nothing is left out as played (scores, not tracks: float32 near-ties swap)
    assert np.allclose(first, np.sort(b.tt_track @ look[0])[::-1][:n], atol=1e-6)

    artists = form(s, sheet, artists=2, per_artist=1)
    seeds = seed_tracks(favourites(sheet.listens, b.track_ids, b.artist, 2), sheet.by_artist, 1, b.meta["now"])
    assert list(artists) == ["retrieval", "both"] and full(artists)
    assert seeds.group_by("user_id").len()["len"].to_list() == [2] * 6  # one track for each of two artists
    assert all(recs.join(seeds, on=PAIR, how="semi").height == 0 for recs in artists.values())  # the seed tracks are not recommended
    assert artists["retrieval"].equals(vector_lists(b, users, bundle.vectors(b, seeds, users, b.meta["fold_in"])[0], seeds))

    asked_both = form(s, sheet, genres=2, mainstream=True, artists=2, neighbours=3, per_artist=1)
    assert list(asked_both) == ["retrieval", "both"] and full(asked_both)
    mixed = unit(bundle.vectors(b, seeds, users, b.meta["fold_in"])[0] + look)
    assert asked_both["retrieval"].equals(vector_lists(b, users, mixed, seeds))

    as_seeds = genre_seeds(s, sheet, genres=2, per=3)
    assert list(as_seeds) == ["retrieval", "both"] and full(as_seeds)
