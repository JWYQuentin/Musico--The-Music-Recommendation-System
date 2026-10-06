import numpy as np
import pytest
import scipy.sparse as sp

from m4a_rec.coldstart import content_sim_vecs, held_out, random_vecs, render, unit, without
from m4a_rec.retrieval import fit, recommend

from .test_retrieval import CFG, PARAMS, SPECS, two_groups


def test_held_out_is_seeded_and_the_right_size():
    a = held_out(1000, 0.05, seed=42)
    assert len(a) == 50 and len(set(a.tolist())) == 50
    assert a.tolist() == sorted(a.tolist()) and a.min() >= 0 and a.max() < 1000
    assert np.array_equal(a, held_out(1000, 0.05, seed=42))
    assert not np.array_equal(a, held_out(1000, 0.05, seed=7))


def test_without_empties_the_columns():
    matrix = sp.csr_matrix(np.array([[1, 2, 0], [0, 3, 4]], dtype=np.float32))
    out = without(matrix, np.array([1]))
    assert out.toarray().tolist() == [[1, 0, 0], [0, 0, 4]]
    assert out.nnz == 2
    assert matrix.toarray().tolist() == [[1, 2, 0], [0, 3, 4]]  # the original is untouched


def test_unit_hand_checked():
    assert unit(np.array([[3.0, 4.0], [0.0, 0.0]])) == pytest.approx(np.array([[0.6, 0.8], [0, 0]]))


def test_content_sim_hand_checked():
    # Three tracks. Tracks 0 and 2 share an audio direction; track 1 points the other way.
    # Lyrics and genres are the same for all three, so audio alone decides the order.
    blocks = {
        "audio": np.array([[2.0, 0.0], [0.0, 5.0], [1.0, 0.0]]),
        "lyrics": np.array([[1.0], [1.0], [1.0]]),
        "has_lyrics": np.array([[1.0], [1.0], [1.0]]),
        "genres": np.array([[1.0], [1.0], [1.0]]),
    }
    played = sp.csr_matrix(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))  # user 0: track 0, user 1: track 1
    users, tracks = content_sim_vecs(played, blocks)
    # each track: [audio direction, 1, 1] / sqrt(3), so same-audio tracks score 3/3 and the others 2/3
    assert tracks @ tracks.T == pytest.approx(np.array([[1, 2 / 3, 1], [2 / 3, 1, 2 / 3], [1, 2 / 3, 1]]))
    assert users @ tracks.T == pytest.approx(np.array([[1, 2 / 3, 1], [2 / 3, 1, 2 / 3]]))
    # a user who played tracks 0 and 1 sits between them: closer to each than they are to one another
    both, _ = content_sim_vecs(sp.csr_matrix(np.array([[1, 1, 0]], dtype=np.float32)), blocks)
    assert (both @ tracks.T).ravel() == pytest.approx(np.full(3, (1 + 2 / 3) / np.sqrt(2 + 4 / 3)))


def test_random_vecs_are_seeded():
    u, v = random_vecs(5, 7, seed=1)
    assert u.shape == (5, 32) and v.shape == (7, 32) and u.dtype == np.float32
    assert np.array_equal(u, random_vecs(5, 7, seed=1)[0])


def test_render():
    one = {"users": 321, "recall@10": 0.25, "ndcg@10": 0.125, "coverage@10": 0.5, "short@10": 0.0}
    md = render(
        {"tt_content": {"val": one}, "random": {"val": one}},
        {"tt_content": {"val": one}},
        n_held=2808,
        c={"fraction": 0.05, "seed": 42},
        splits=("val",),
        ks=[10],
    )
    assert "2,808 tracks (5% of the training tracks, seed 42) were held out" in md
    assert "## Held-out tracks only: val (321 users scored)" in md
    assert "## All tracks: val (321 users scored)" in md
    assert "| random | 0.2500 |" in md


# ---- The test below trains a model, so it needs the owner's three functions in twotower.py.


def test_content_model_recommends_a_track_it_never_trained_on():
    inter, features = two_groups()
    names = inter.track_ids.to_list()
    # Two tracks of each group get no training pairs at all.
    held = np.array([names.index(t) for t in ("A0", "A1", "B0", "B1")])
    # Users who skipped A0 or A1 (0, 1, 6, 7) or B0 or B1 (12, 13, 18, 19) still have a held-out
    # track of their own group to be offered. Everyone else has played both already.
    judged = [0, 1, 6, 7, 12, 13, 18, 19]

    def evaluate(user_vecs, track_vecs):
        recs = recommend(inter, user_vecs, track_vecs, n=1, candidates=held)
        first = dict(zip(recs["user_id"], recs["track_id"]))
        return {"own_group": sum(first[u][0] == ("A" if u < 10 else "B") for u in judged) / len(judged)}

    result = fit(inter, features, SPECS["tt_content"], PARAMS, CFG, evaluate, exclude_tracks=held, log=lambda m: None)
    # Each of them is offered their own group's unplayed held-out track ahead of the other group's two.
    assert result["metrics"]["own_group"] == 1.0
