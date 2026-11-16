import ast
import json
import re
from pathlib import Path

import pytest

from m4a_rec import results_page as rp
from m4a_rec.rank_features import FEATURES


def metrics(base: float) -> dict:
    """Every metric the reports hold, base + a small step per metric so each value is distinct."""
    names = [f"{m}@{k}" for k in (10, 20, 100, 500) for m in ("recall", "ndcg", "coverage", "short")]
    return {"users": 7, **{n: round(base + i / 1000, 4) for i, n in enumerate(names)}}


def gap(diff: float) -> dict:
    return {"diff": diff, "low": diff - 0.01, "high": diff + 0.01}


def fake_reports() -> dict:
    split = lambda b: {"val": metrics(b), "test": metrics(b + 0.1)}  # noqa: E731
    found = lambda shares: {g: {"relevant": 5, "share": s, "found": 0.1} for g, s in shares.items()}  # noqa: E731
    new_row = lambda b: {"retrieval": split(b), "both": split(b + 0.2)}  # noqa: E731
    bias_model = {
        "ndcg": 0.03,
        "slots": {"popularity": {"head": 0.6, "tail": 0.4}, "genre": {"rock": 1.0}},
        "avg_pct": 0.9,
        "coverage": 0.2,
        "gini": 0.8,
        "found": {
            "popularity": found({"head": 0.7, "tail": 0.29, "no group": 0.01}),
            "genre": found({"rock": 1.0}),
        },
        "users": {
            "listeners": {"niche": {"users": 3, "mean": 0.01, "low": 0.0, "high": 0.02}},
            "genre": {"rock": {"users": 3, "mean": 0.01, "low": 0.0, "high": 0.02}},
        },
    }
    timing = {"p50": 0.02, "p95": 0.03}
    return {
        "ranker": {
            "results": {"als": {"params": {"secret": 1}, **split(0.1)}, "tt_id": {"params": {}, **split(0.2)}},
            "gaps": [{"model": "tt_id", "against": "als", "val": {"ndcg@10": gap(0.01)}, "test": {"ndcg@10": gap(0.02)}}],
            "shap": {"tt_rank": 0.1, "artist_listeners": 0.3},
        },
        "listens_curve": {
            "users": {"held_out": 9, "val": 8, "test": 7},
            "popularity": split(0.01),
            "curve": {"1": {"tracks": 1.0, "known": 0.3, **new_row(0.02)}, "all": {"tracks": 50.0, **new_row(0.03)}},
        },
        "questionnaire": {
            "users": {"held_out": 9, "val": 8, "test": 7},
            "chosen": {},
            "popularity": split(0.01),
            "forms": {"1 genre": {"questions": 1, "stages": "retrieval", "retrieval": split(0.04)}},
        },
        "coldstart": {
            "cold": {"random": split(0.001)},
            "all": {"tt_content": split(0.05)},
        },
        "bias": {"split": "test", "k": 10, "sizes": {}, "results": {"als": bias_model}, "gap": {}},
        "bias_tuning": {"rows": [{"lambda": 0.0, "ndcg": 0.04, "head": 0.5, "middle": 0.3, "tail": 0.2, "coverage": 0.3, "gini": 0.8}], "chosen": 0.0},
        "serving": {
            "parity": {"users": 5, "same_top": {"10": 1.0}},
            "newcomers": {"rows": {"learned vectors": split(0.06)}},
            "latency": {"requests": 4, "k": 10, "known": {"total": timing}, "new": {"total": timing}},
        },
    }


def test_build_puts_the_page_data_in_the_slot():
    html = rp.build(fake_reports(), "<script>const DATA = /*DATA*/;</script>")
    data = json.loads(re.search(r"const DATA = (.*);</script>", html).group(1))

    als, tt = data["results"]["models"]
    assert (als["label"], tt["label"]) == ("ALS", "Two-tower")
    assert als["test"]["ndcg@10"] == 0.201  # 0.1 + 0.1 for test, + 1/1000 for the second metric
    assert data["results"]["gaps"][0] == {
        "model": "Two-tower", "against": "ALS",
        "val": {"ndcg@10": gap(0.01)}, "test": {"ndcg@10": gap(0.02)},
    }
    # SHAP largest first, with its group and words
    assert [s["feature"] for s in data["results"]["shap"]] == ["artist_listeners", "tt_rank"]
    assert data["results"]["shap"][0]["group"] == "artist"
    assert [p["tracks"] for p in data["newcomers"]["curve"]] == ["1", "all"]
    assert data["newcomers"]["forms"][0] == {"form": "1 genre", "questions": 1, "retrieval": {
        "val": {k: metrics(0.04)[k] for k in rp.NEW_METRICS},
        "test": {k: metrics(0.14)[k] for k in rp.NEW_METRICS},
    }}
    # listening shares leave out tracks with no group
    assert data["bias"]["listening"]["popularity"] == {"head": 0.7, "tail": 0.29}
    assert data["serving"]["latency"]["new"]["total"] == {"p50": 0.02, "p95": 0.03}


def test_only_the_listed_fields_reach_the_page():
    reports = fake_reports()
    reports["ranker"]["results"]["als"]["test"]["per_user"] = {"123": 0.5}
    reports["bias"]["results"]["als"]["found"]["genre"]["rock"]["tracks"] = ["t1"]
    text = rp.embed(rp.page_data(reports))
    for leaked in ("per_user", "secret", "params", "tracks\":[", "t1", "coverage@10", "short@10", '"known":0.3'):
        assert leaked not in text, leaked


def test_every_ranker_feature_has_words():
    assert set(rp.FEATURES) == set(FEATURES)


def test_build_refuses_a_template_without_one_slot():
    with pytest.raises(ValueError):
        rp.build(fake_reports(), "<html></html>")


def test_embedded_text_cannot_close_the_script():
    assert "</script>" not in rp.embed({"x": "</script><b>"})


def test_the_page_reads_only_reports():
    """The page is hosted, so it may read nothing but reports/ and write nothing but docs/."""
    source = Path(rp.__file__).read_text()
    assert set(re.findall(r'\["paths"\]\["(\w+)"\]', source)) == {"reports", "docs"}
    for name in ("data/", "tracks.parquet", "load_train", "serving/", "music4all"):
        assert name not in source.lower(), name
    imported = {
        node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imported <= {"__future__", "config", "pathlib"}, imported
