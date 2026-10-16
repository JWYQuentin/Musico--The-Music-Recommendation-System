import json
from datetime import datetime

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from m4a_rec.bundle import candidates, fold_in_als, listens, plays_matrix, retrieve, top_unseen, vectors, write
from m4a_rec.candidates import snapshot
from m4a_rec.evaluate import PAIR
from m4a_rec.rank_features import FEATURES, unit


def close(got, expected):
    return np.allclose(got, np.array(expected, dtype=float), equal_nan=True, atol=1e-5)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, write_bundle):
    ranker = tmp_path_factory.mktemp("ranker")  # this module needs only the ranker's meta
    (ranker / "meta.json").write_text(json.dumps({"features": FEATURES, "params": {"n_candidates": 10}, "trees": 1}))
    (ranker / "model.json").write_text("{}")
    return write_bundle(tmp_path_factory.mktemp("bundle"), ranker)


def test_fold_in_als_hand_checked():
    played = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))  # listener 0 played track 0; listener 1 nothing
    got = fold_in_als(played, np.array([[2.0], [1.0]]), regularization=1, alpha=3)
    # confidence 3 on track 0: (2*2 + 1*1 + 1 + (3 - 1) * 2*2) x = 3 * 2, so x = 6 / 14
    assert close(got, [[6 / 14], [0]])


def test_fold_in_als_is_implicits_own_solve():
    from implicit.als import AlternatingLeastSquares

    rng = np.random.default_rng(0)
    m = sp.random(40, 60, density=0.1, random_state=1, format="csr", dtype=np.float32)
    m.data = np.log1p(rng.integers(1, 20, m.nnz)).astype(np.float32)
    model = AlternatingLeastSquares(factors=8, regularization=0.1, alpha=10, iterations=3, random_state=0)
    model.fit(m, show_progress=False)
    expected = model.recalculate_user(np.arange(5), m[:5])
    assert np.allclose(fold_in_als(m[:5], model.item_factors, 0.1, 10), expected, atol=1e-4)


def test_plays_matrix_hand_checked():
    when = datetime(2020, 1, 1)
    history = pl.DataFrame({"user_id": [7, 7, 7, 7], "track_id": ["a", "a", "b", "x"], "timestamp": [when] * 4})
    users, tracks = pl.Series("user_id", [7]), pl.Series("track_id", ["a", "b", "c"])
    assert plays_matrix(history, users, tracks, "binary").toarray().tolist() == [[1, 1, 0]]  # x is not in the catalogue
    assert close(plays_matrix(history, users, tracks, "log").toarray(), [[np.log(3), np.log(2), 0]])


def test_top_unseen_is_retrieval_top_unseen():
    from m4a_rec.retrieval import top_unseen as with_torch

    rng = np.random.default_rng(0)
    users, tracks = rng.standard_normal((3, 4)).astype(np.float32), rng.standard_normal((30, 4)).astype(np.float32)
    played = sp.lil_matrix((3, 30), dtype=np.float32)
    played[0, :10] = 1  # listener 0 has 20 tracks left, listener 1 all 30, listener 2 only 2
    played[2, :28] = 1
    played = played.tocsr()
    idx, scores = top_unseen(users, tracks, played, 25)
    expected_idx, expected_scores = with_torch(users, tracks, played, 25)
    assert np.array_equal(idx, expected_idx) and np.allclose(scores, expected_scores)
    assert (idx[0] == -1).sum() == 5 and (idx[1] == -1).sum() == 0 and (idx[2] == -1).sum() == 23


def test_write_and_load_round_trip(bundle, served):
    s = served
    assert bundle.meta["now"] == s["history"]["timestamp"].max()
    assert bundle.meta["n_recs"] == 20 and bundle.meta["n_candidates"] == 10
    assert bundle.meta["als"] == {"weighting": "log", "regularization": 0.1, "alpha": 10}
    assert bundle.track_ids.to_list() == s["inter"].track_ids.to_list()
    assert np.array_equal(bundle.user_ids.to_numpy(), s["inter"].user_ids.to_numpy())
    assert np.array_equal(bundle.tt_track, s["tt"][1]) and np.array_equal(bundle.als_user, s["als"][0])
    assert all(np.array_equal(bundle.blocks[name], values) for name, values in s["blocks"].items())
    user = int(bundle.user_ids[3])
    assert listens(bundle, [user]).sort(["timestamp", "track_id"]).equals(
        s["history"].filter(pl.col("user_id") == user).sort(["timestamp", "track_id"])
    )


def test_write_refuses_models_fitted_on_other_listens(tmp_path, served):
    s = served
    fewer = s["history"].filter(pl.col("user_id") != s["history"]["user_id"][0])
    with pytest.raises(ValueError, match="not fitted on these listens"):
        write(tmp_path, fewer, s["dirs"]["tt"], s["dirs"]["als"], tmp_path, s["blocks"], s["artist"], s["album"], s["titles"], s["settings"])


def test_a_known_listener_gets_their_snapshot_rows(bundle, served):
    """The service's way to the ranker's input gives the rows candidates.py gives for the same listeners."""
    s = served
    expected = snapshot(
        s["history"], s["inter"], *s["tt"], *s["als"], s["blocks"], s["artist"], s["album"], n=20, recent_days=28, short_days=7
    )
    users = bundle.user_ids.gather([0, 3, 11, 20])
    history = listens(bundle, users.to_list())
    tt_user, als_user = vectors(bundle, history, users, fold_in=None)
    got = candidates(bundle, history, users, *retrieve(bundle, history, users, tt_user), als_user)
    expected = expected.filter(pl.col("user_id").is_in(users.to_list()))
    assert got.columns == expected.columns
    assert got.select(PAIR).equals(expected.select(PAIR))
    assert all(close(got[name].to_numpy(), expected[name].to_numpy()) for name in FEATURES)


def test_vectors_learned_or_folded_in(bundle, served):
    users = bundle.user_ids.gather([2, 5])
    tt_user, als_user = vectors(bundle, listens(bundle, users.to_list()), users, fold_in=None)
    assert np.array_equal(tt_user, served["tt"][0][[2, 5]]) and np.array_equal(als_user, served["als"][0][[2, 5]])
    with pytest.raises(ValueError, match="no learned vectors"):
        vectors(bundle, listens(bundle, []), pl.Series("user_id", [-1]), fold_in=None)
    # someone new who has played one track, twice: their two-tower vector points the way that track's does
    new = pl.Series("user_id", [-1])
    history = pl.DataFrame({"user_id": [-1, -1], "track_id": [bundle.track_ids[4]] * 2, "timestamp": [bundle.meta["now"]] * 2})
    for fold_in in ("binary", "log"):
        tt_user, als_user = vectors(bundle, history, new, fold_in=fold_in)
        assert close(tt_user, unit(bundle.tt_track[[4]]))
        played = plays_matrix(history, new, bundle.track_ids, "log")
        assert close(als_user, fold_in_als(played, bundle.als_track, 0.1, 10))


def test_features_do_not_depend_on_who_is_in_the_batch(bundle):
    """Up to float rounding: a float32 matrix product can differ in its last bits with the
    number of rows in it, so the scores behind six features move by about 1e-6."""
    users = bundle.user_ids.to_list()
    history = listens(bundle, users)

    def rows_for(chunk, fold_in):
        series = pl.Series("user_id", chunk)
        mine = history.filter(pl.col("user_id").is_in(chunk))
        tt_user, als_user = vectors(bundle, mine, series, fold_in)
        return candidates(bundle, mine, series, *retrieve(bundle, mine, series, tt_user), als_user)

    for fold_in in (None, "log"):
        whole = rows_for(users, fold_in).sort(PAIR)
        batched = pl.concat([rows_for(users[i : i + 3], fold_in) for i in range(0, len(users), 3)]).sort(PAIR)
        assert whole.select(PAIR).equals(batched.select(PAIR))
        assert all(close(batched[name].to_numpy(), whole[name].to_numpy()) for name in FEATURES)
