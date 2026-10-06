import numpy as np
import pytest
import scipy.sparse as sp

from m4a_rec.interactions import build
from m4a_rec.retrieval import (
    fit,
    recommend,
    render_report,
    render_tuning,
    staged_search,
    top_unseen,
    training_pairs,
)

from .test_baselines import als_events

LOG2, LOG3 = np.log(2), np.log(3)


def test_top_unseen_hand_checked():
    users = np.array([[1, 0], [0, 1]], dtype=np.float32)
    tracks = np.array([[1, 0], [0.8, 0.6], [0, 1], [-1, -0.5]], dtype=np.float32)
    # scores   track:  0    1    2     3
    #   user 0:        1   0.8   0    -1
    #   user 1:        0   0.6   1   -0.5
    played = sp.csr_matrix(np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32))
    idx, scores = top_unseen(users, tracks, played, n=2)
    assert idx.tolist() == [[1, 2], [1, 0]]  # each user's own played track is skipped
    assert scores == pytest.approx(np.array([[0.8, 0], [0.6, 0]]))


def test_top_unseen_candidates_and_padding():
    users = np.array([[1, 0], [0, 1]], dtype=np.float32)
    tracks = np.array([[1, 0], [0.8, 0.6], [0, 1], [-1, -0.5]], dtype=np.float32)
    played = sp.csr_matrix(np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32))
    idx, _ = top_unseen(users, tracks, played, n=2, candidates=np.array([2, 3]))
    assert idx.tolist() == [[2, 3], [3, -1]]  # user 1 has played track 2, so one place stays empty
    idx, _ = top_unseen(users, tracks, played, n=10)  # more than there are tracks
    assert idx.tolist() == [[1, 2, 3, -1], [1, 0, 3, -1]]


def test_training_pairs_hand_checked():
    # user 0 played track 0 once and track 1 twice; user 1 played track 1 once. Cells are log(1 + plays).
    matrix = sp.csr_matrix(np.array([[LOG2, LOG3, 0], [0, LOG2, 0]], dtype=np.float32))
    users, tracks, weights, log_q = training_pairs(matrix, "pairs")
    assert sorted(zip(users.tolist(), tracks.tolist())) == [(0, 0), (0, 1), (1, 1)]
    assert weights.tolist() == [1, 1, 1]
    assert np.exp(log_q[:2]) == pytest.approx([1 / 3, 2 / 3])  # track 1 is in two of the three pairs

    _, _, weights, log_q = training_pairs(matrix, "log_plays")
    assert sorted(weights.tolist()) == pytest.approx([LOG2, LOG2, LOG3])
    total = 2 * LOG2 + LOG3
    assert np.exp(log_q[:2]) == pytest.approx([LOG2 / total, (LOG2 + LOG3) / total])

    users, tracks, _, log_q = training_pairs(matrix, "pairs", exclude_tracks=np.array([1]))
    assert (users.tolist(), tracks.tolist()) == ([0], [0])  # only the pair on track 0 is left
    assert np.exp(log_q[0]) == pytest.approx(1.0)


def test_staged_search_hand_checked():
    calls = []

    def run(params):
        calls.append(dict(params))
        return -abs(params["a"] - 2) - abs(params["b"] - 5)  # best at a = 2, b = 5

    best, tried = staged_search({"a": 0, "b": 0}, [{"a": [1, 2, 3]}, {"b": [4, 5]}], run)
    assert best == {"a": 2, "b": 5}
    # stage 1 varies a with b at its start value; stage 2 varies b with a at the stage-1 winner
    assert calls == [{"a": 1, "b": 0}, {"a": 2, "b": 0}, {"a": 3, "b": 0}, {"a": 2, "b": 4}, {"a": 2, "b": 5}]
    assert [metric for _, metric in tried] == [-6, -5, -6, -1, 0]


def test_staged_search_does_not_repeat_a_setting():
    calls = []
    staged_search({"a": 0}, [{"a": [0, 2]}, {"a": [2]}], lambda p: calls.append(dict(p)) or p["a"])
    assert calls == [{"a": 0}, {"a": 2}]


def test_tuning_table_puts_the_best_run_first():
    m = lambda r: {"recall@100": r, "recall@10": 0.1, "ndcg@10": 0.2, "coverage@10": 0.3}  # noqa: E731
    runs = {"tt_id": [{"params": {"dim": 64}, "epoch": 4, "metrics": m(0.11)}, {"params": {"dim": 128}, "epoch": 7, "metrics": m(0.13)}]}
    md = render_tuning(runs, "recall@100", ks=[10, 100])
    assert "| dim | epoch | recall@100 | recall@10 | ndcg@10 | coverage@10 |" in md
    assert md.index("| 128 | 7 | 0.1300 |") < md.index("| 64 | 4 | 0.1100 |")


def test_report_renders():
    one = {"users": 1234, "recall@10": 0.25, "ndcg@10": 0.125, "coverage@10": 0.5, "short@10": 0.0}
    results = {"als": {"params": {"alpha": 10}, "val": one, "test": one}, "tt_id": {"params": {"dim": 128}, "val": one, "test": one}}
    md = render_report(results, ["retrieval_train"], ks=[10])
    assert "## test (1,234 users scored)" in md
    assert "| tt_id | 0.2500 |" in md and "| als | 0.2500 |" in md
    assert '| tt_id | {"dim": 128} |' in md


# ---- The tests below train a model, so they need the owner's three functions in twotower.py.

# One batch holds all 100 pairs, and patience equals max_epochs so training never stops early.
CFG = {"seed": 0, "device": "cpu", "batch_size": 128, "max_epochs": 60, "patience": 60, "select_on": "own_group"}
PARAMS = {
    "dim": 8,
    "temperature": 0.2,
    "lr": 0.1,
    "log_q": False,
    "mask_duplicates": False,
    "sampling": "pairs",
    "hidden": 16,
    "dropout": 0.0,
}
SPECS = {
    "tt_id": {"use_id": True, "use_content": False},
    "tt_hybrid": {"use_id": True, "use_content": True},
    "tt_content": {"use_id": False, "use_content": True},
}


def two_groups():
    """Users 0-9 play five of the six A tracks, users 10-19 five of the six B tracks."""
    inter = build(als_events(), "log")
    # One feature per group: A tracks are [1, 0], B tracks are [0, 1].
    features = np.array([[1, 0] if t.startswith("A") else [0, 1] for t in inter.track_ids], dtype=np.float32)
    return inter, features


def own_group(inter):
    """Share of users whose best unplayed track is the one track of their group they skipped."""

    def evaluate(user_vecs, track_vecs):
        recs = recommend(inter, user_vecs, track_vecs, n=1)
        right = sum(t == f"{'A' if u < 10 else 'B'}{u % 6}" for u, t in zip(recs["user_id"], recs["track_id"]))
        return {"own_group": right / 20}

    return evaluate


@pytest.mark.parametrize("name", SPECS)
def test_fit_learns_which_group_a_user_belongs_to(name):
    inter, features = two_groups()
    result = fit(inter, features, SPECS[name], PARAMS, CFG, own_group(inter), log=lambda m: None)
    assert result["metrics"]["own_group"] == 1.0
    assert result["user_vecs"].shape == (20, 8) and result["track_vecs"].shape == (12, 8)
    assert np.linalg.norm(result["track_vecs"], axis=1) == pytest.approx(np.ones(12), abs=1e-5)
    assert result["history"][0]["loss"] > result["history"][-1]["loss"]


def test_fit_keeps_the_best_epoch_and_stops_after_patience():
    inter, features = two_groups()
    values = iter([0.1, 0.3, 0.2, 0.2, 0.2, 0.9])  # the last one is never reached with patience 3
    cfg = {**CFG, "patience": 3}
    result = fit(inter, features, SPECS["tt_id"], PARAMS, cfg, lambda u, v: {"own_group": next(values)}, log=lambda m: None)
    assert result["epoch"] == 2 and result["metrics"] == {"own_group": 0.3}
    assert [h["epoch"] for h in result["history"]] == [1, 2, 3, 4, 5]


def test_fit_is_repeatable_on_the_cpu():
    inter, features = two_groups()
    run = lambda: fit(inter, features, SPECS["tt_hybrid"], PARAMS, {**CFG, "max_epochs": 3}, own_group(inter), log=lambda m: None)  # noqa: E731
    assert np.array_equal(run()["track_vecs"], run()["track_vecs"])


@pytest.mark.parametrize("option", [{"log_q": True, "mask_duplicates": True}, {"sampling": "log_plays"}])
def test_fit_learns_with_each_option_switched_on(option):
    inter, features = two_groups()
    result = fit(inter, features, SPECS["tt_id"], {**PARAMS, **option}, CFG, own_group(inter), log=lambda m: None)
    assert result["metrics"]["own_group"] == 1.0
