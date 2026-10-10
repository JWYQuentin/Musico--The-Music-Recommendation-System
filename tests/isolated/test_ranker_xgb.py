"""Tests for ranker.py. They import XGBoost, so they run in their own process: see tests/test_ranker.py."""
import numpy as np
import polars as pl
import pytest

from m4a_rec.evaluate import PAIR
from m4a_rec.ranker import (
    TWO_STAGE,
    explain,
    render_report,
    render_tuning,
    rerank,
    run,
    shap_summary,
    shap_values,
    to_dmatrix,
    training_rows,
)

FEATURES = ["signal", "noise"]
CFG = {
    "fixed": {"tree_method": "hist", "min_child_weight": 1},
    "seed": 0,
    "max_rounds": 30,
    "check_every": 10,
    "select_on": "top1",
}
PARAMS = {"n_candidates": 10, "max_depth": 3, "eta": 0.3, "lambdarank_pair_method": "topk", "objective": "rank:ndcg"}


def candidates(seed, n_users=300, n=10, shuffle_labels=False):
    """Each user has `n` candidates in random retrieval order. The one with the highest `signal` is the positive."""
    rng = np.random.default_rng(seed)
    signal = rng.random((n_users, n))
    label = (signal == signal.max(axis=1, keepdims=True)).astype(np.int8)
    if shuffle_labels:
        label = rng.permuted(label, axis=1)
    return pl.DataFrame(
        {
            "user_id": np.repeat(np.arange(n_users), n),
            "track_id": [f"t{i}" for i in np.tile(np.arange(n), n_users)],
            "tt_rank": np.tile(np.arange(1, n + 1), n_users),
            "signal": signal.ravel().astype(np.float32),
            "noise": rng.random(n_users * n).astype(np.float32),
            "label": label.ravel(),
        }
    )


def top1(truth):
    """Share of users whose first-placed track is their positive."""
    positives = truth.filter(pl.col("label") == 1).select(PAIR)

    def evaluate(recs):
        first = recs.sort(["user_id", "score"], descending=[False, True]).group_by("user_id", maintain_order=True).first()
        return {"top1": first.join(positives, on=PAIR, how="semi").height / first.height}

    return evaluate


def test_training_rows_hand_checked():
    train = pl.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "track_id": list("abcabcabc"),
            "tt_rank": [1, 2, 3] * 3,
            "label": [0, 1, 0, 0, 0, 1, 0, 0, 0],  # user 2's positive is their third candidate; user 3 has none
        }
    )
    assert training_rows(train, n_candidates=3)["user_id"].to_list() == [1, 1, 1, 2, 2, 2]
    assert training_rows(train, n_candidates=2)["user_id"].to_list() == [1, 1]  # user 2's positive is cut off


def test_rerank_hand_checked():
    infer = pl.DataFrame({"user_id": [1, 1, 1, 1], "track_id": list("abcd"), "tt_rank": [1, 2, 3, 4]})
    recs = rerank(infer, scores=np.array([0.1, 0.9]), n_candidates=2)  # the ranker prefers b to a
    # the score is minus the place in the final list; c and d keep their retrieval order, below both
    assert dict(zip(recs["track_id"], recs["score"])) == {"b": -1, "a": -2, "c": -3, "d": -4}
    assert recs.schema == {"user_id": pl.Int64, "track_id": pl.String, "score": pl.Float32}


def test_rerank_keeps_retrieval_order_between_equal_scores():
    infer = pl.DataFrame({"user_id": [1, 1, 1, 1], "track_id": list("dcba"), "tt_rank": [1, 2, 3, 4]})
    recs = rerank(infer, scores=np.array([0.5, 0.5, 0.9]), n_candidates=3)  # d and c tie; by track ID c would come first
    assert dict(zip(recs["track_id"], recs["score"])) == {"b": -1, "d": -2, "c": -3, "a": -4}


def test_ranker_learns_the_feature_that_decides_the_label():
    train, infer = candidates(seed=0), candidates(seed=1)
    evaluate = top1(infer)
    retrieval_order = infer.select(PAIR).with_columns(score=-infer["tt_rank"].cast(pl.Float32))
    assert evaluate(retrieval_order)["top1"] < 0.2  # the positive is in a random place: about 1 in 10
    result = run(train, infer.drop("label"), FEATURES, PARAMS, CFG, evaluate, log=lambda m: None)
    assert result["metrics"]["top1"] > 0.9
    assert result["rounds"] in (10, 20, 30) and [h["rounds"] for h in result["history"]] == [10, 20, 30]


def test_ranker_learns_nothing_from_shuffled_labels():
    train, infer = candidates(seed=0, shuffle_labels=True), candidates(seed=1)
    result = run(train, infer.drop("label"), FEATURES, PARAMS, CFG, top1(infer), log=lambda m: None)
    assert result["metrics"]["top1"] < 0.25  # no better than chance once labels carry no information


def test_plain_yes_no_objective_also_learns():
    train, infer = candidates(seed=0), candidates(seed=1)
    params = {**PARAMS, "objective": "binary:logistic"}
    result = run(train, infer.drop("label"), FEATURES, params, CFG, top1(infer), log=lambda m: None)
    assert result["metrics"]["top1"] > 0.8  # against about 0.1 for a random order


def test_shap_values_add_up_and_point_at_the_signal():
    train, infer = candidates(seed=0), candidates(seed=1)
    result = run(train, infer.drop("label"), FEATURES, PARAMS, CFG, top1(infer), log=lambda m: None)
    booster, rounds = result["booster"], result["rounds"]
    contributions = shap_values(booster, infer, FEATURES, rounds)
    assert contributions.shape == (infer.height, 3)  # one column per feature, then the base value
    margin = booster.predict(to_dmatrix(infer, FEATURES), output_margin=True, iteration_range=(0, rounds))
    assert contributions.sum(axis=1) == pytest.approx(margin, abs=1e-3)  # contributions add up to the score
    summary = shap_summary(booster, infer, FEATURES, rounds, n_rows=1000, seed=0)
    assert list(summary) == ["signal", "noise"] and summary["signal"] > 10 * summary["noise"]
    parts = explain(booster, infer.head(1), FEATURES, rounds)
    assert parts[0][0] == "signal" and parts[0][1] == pytest.approx(infer["signal"][0])


def test_tuning_table_puts_the_best_run_first():
    m = lambda v: {"ndcg@10": v, "recall@10": 0.1, "recall@20": 0.2, "recall@100": 0.3, "coverage@10": 0.4}  # noqa: E731
    rows = [{"params": {"max_depth": 4}, "rounds": 100, "metrics": m(0.03)}, {"params": {"max_depth": 6}, "rounds": 250, "metrics": m(0.04)}]
    md = render_tuning(rows, "ndcg@10", ks=[10, 20, 100, 500])
    assert "| max_depth | trees | ndcg@10 | recall@10 | recall@20 | recall@100 | coverage@10 |" in md
    assert md.index("| 6 | 250 | 0.0400 |") < md.index("| 4 | 100 | 0.0300 |")


def test_report_renders():
    ks = [10, 20, 100]
    one = {"users": 1234, **{f"{m}@{k}": 0.02 for m in ("recall", "ndcg", "coverage", "short") for k in ks}}
    better = {**one, "ndcg@10": 0.03}
    results = {"tt_id": {"params": {"dim": 256}, "val": one}, TWO_STAGE: {"params": {"max_depth": 6, "trees": 200}, "val": better}}
    gaps = {"val": {f"{m}@{k}": {"diff": 0.01, "low": 0.004, "high": 0.016} for m in ("recall", "ndcg") for k in ks}}
    summary = {"tt_z": 0.6, "track_listeners": 0.3, "sim_audio": 0.1}
    ablation = [{"without": "models", "rounds": 100, "metrics": {**one, "ndcg@10": 0.015}}]
    example = {"user_id": 7, "track_id": "x", "tt_rank": 42, "parts": [("tt_z", 1.5, 0.8), ("sim_audio", float("nan"), -0.1)]}
    md = render_report(results, gaps, {"tt_id": {"val": {**one, "ndcg@10": 0.01}}}, summary, ablation, example, ("val",), ks, "ndcg@10", 2000)
    assert "## val (1,234 users scored)" in md and "resampling users 2,000 times" in md
    assert f"| val | ndcg@10 | 0.0200 | 0.0300 | 0.0100 | +0.0040 to +0.0160 | +50.0% |" in md
    assert "| val | tt_id | ndcg@10 | 0.0100 | 0.0200 | +100.0% |" in md  # retrieval_train only, then both
    assert "| tt_z | models | 0.6000 | 60.0% |" in md and "| models | 60.0% |" in md
    assert "| models | 0.0150 | 0.0200 | 0.0200 | -50.0% |" in md
    assert "candidate number 42 from retrieval" in md and "| tt_z | 1.5000 | 0.8000 |" in md
    assert "| sim_audio | missing | -0.1000 |" in md
