"""Phase 3 baselines: popularity, item-kNN and ALS, fitted on the train splits in configs/baselines.yaml.

Every model hands in `n_recs` tracks per user, none of which the user played in training.

  tune    every grid setting, scored on val. Writes reports/baselines_tuning.md.
  report  the chosen settings, scored on val and test. Writes reports/baselines.md and .json.

Usage: python -m m4a_rec.baselines tune | report
"""
from __future__ import annotations

import itertools
import json
import sys
import warnings
from datetime import timedelta

import numpy as np
import polars as pl
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender
from implicit.utils import ParameterWarning

from .config import load_config
from .evaluate import score
from .interactions import Interactions, build, load_train, to_frame

MODELS = ["popularity", "item_knn", "als"]
BATCH = 1000  # users ranked at a time in popularity


def popularity(
    events: pl.DataFrame, inter: Interactions, n: int, count: str, window_days: int | None
) -> pl.DataFrame:
    """The same ranking for every user, minus the tracks they have played.

    Tracks are ranked by distinct listeners ("users") or total plays ("listens") in the last
    `window_days` of `events`. A track with no plays in that period is never recommended.
    """
    if window_days is not None:
        start = events["timestamp"].max() - timedelta(days=window_days)
        events = events.filter(pl.col("timestamp") > start)
    agg = pl.len() if count == "listens" else pl.col("user_id").n_unique()
    counts = events.group_by("track_id").agg(agg.alias("score"))
    track_score = (
        inter.track_ids.to_frame()
        .join(counts, on="track_id", how="left", maintain_order="left")["score"]
        .fill_null(0)
        .to_numpy()
    )
    order = np.argsort(-track_score, kind="stable")
    # The top n + s tracks hold at least n that a user with s played tracks has not heard.
    order = order[track_score[order] > 0][: n + inter.matrix.getnnz(axis=1).max()]
    idx = []
    for start in range(0, inter.matrix.shape[0], BATCH):
        seen = inter.matrix[start : start + BATCH][:, order].toarray() != 0
        # A stable sort of the seen flags puts unheard tracks first, most popular first.
        first = np.argsort(seen, axis=1, kind="stable")[:, :n]
        idx.append(np.where(np.take_along_axis(seen, first, axis=1), -1, order[first]))
    idx = np.concatenate(idx)
    return to_frame(inter, idx, track_score[idx])


def item_knn(inter: Interactions, n: int, k: int) -> pl.DataFrame:
    """Tracks whose listeners overlap with the tracks a user played (cosine, top k per track)."""
    model = CosineRecommender(K=k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ParameterWarning)  # implicit converts its own matrix in fit
        model.fit(inter.matrix, show_progress=False)
    idx, scores = model.recommend(np.arange(inter.matrix.shape[0]), inter.matrix, N=n)
    return to_frame(inter, idx, scores)


def als(
    inter: Interactions,
    n: int,
    factors: int,
    regularization: float,
    alpha: float,
    iterations: int,
    seed: int,
) -> pl.DataFrame:
    """Implicit-feedback matrix factorisation; a cell's confidence is alpha times its value."""
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations,
        random_state=seed,
    )
    model.fit(inter.matrix, show_progress=False)
    idx, scores = model.recommend(np.arange(inter.matrix.shape[0]), inter.matrix, N=n)
    return to_frame(inter, idx, scores)


def fit_recommend(name: str, events: pl.DataFrame, cfg: dict, params: dict) -> pl.DataFrame:
    """Recommendations from one baseline. `params` is one grid point or the chosen setting."""
    n = cfg["n_recs"]
    if name == "popularity":
        return popularity(events, build(events, "binary"), n, **params)
    if name == "item_knn":
        return item_knn(build(events, params["weighting"]), n, params["k"])
    fixed = cfg["als"]
    inter = build(events, fixed["weighting"])
    return als(inter, n, iterations=fixed["iterations"], seed=cfg["seed"], **params)


def grid_points(grid: dict) -> list[dict]:
    """Every combination of the listed values, one dict per combination."""
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def _table(header: list[str], rows: list[list]) -> list[str]:
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows] + [""]


def render_tuning(rows: list[dict], select_on: str, ks: list[int]) -> str:
    """One table per model, best setting first. Each row is {model, params, metrics}."""
    shown = list(dict.fromkeys([select_on, *(f"recall@{k}" for k in ks), f"coverage@{ks[0]}", f"short@{ks[-1]}"]))
    L = [
        "# Phase 3 baseline tuning",
        "",
        "Generated by `python -m m4a_rec.baselines tune`. Do not edit by hand; rerun instead.",
        "",
        f"Scored on val. Each table is sorted by {select_on}, best first.",
        "",
    ]
    for name in dict.fromkeys(r["model"] for r in rows):
        mine = sorted((r for r in rows if r["model"] == name), key=lambda r: -r["metrics"][select_on])
        keys = list(mine[0]["params"])
        L += [f"## {name}", ""]
        L += _table(
            keys + shown,
            [[r["params"][k] for k in keys] + [r["metrics"][m] for m in shown] for r in mine],
        )
    return "\n".join(L)


def render_report(results: dict, train_splits: list[str], ks: list[int]) -> str:
    """`results` maps model name to {params, val: metrics, test: metrics}."""
    L = [
        "# Phase 3 baselines",
        "",
        "Generated by `python -m m4a_rec.baselines report`. Do not edit by hand; rerun instead.",
        "",
        f"Fitted on {' + '.join(train_splits)} with the settings chosen on val:",
        "",
    ]
    L += _table(["Model", "Setting"], [[name, json.dumps(r["params"])] for name, r in results.items()])
    titles = {
        "recall": "Recall@K",
        "ndcg": "NDCG@K",
        "coverage": "Coverage@K: share of training tracks that reach some user's top K",
        "short": "Short@K: share of users handed fewer than K tracks",
    }
    for split in ("val", "test"):
        users = next(iter(results.values()))[split]["users"]
        L += [f"## {split} ({users:,} users scored)", ""]
        for metric, title in titles.items():
            L += [title, ""]
            L += _table(
                ["Model", *(f"@{k}" for k in ks)],
                [[name, *(r[split][f"{metric}@{k}"] for k in ks)] for name, r in results.items()],
            )
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] not in {"tune", "report"}:
        print(__doc__)
        return 2
    cfg = load_config()
    b, ks, reports = cfg["baselines"], cfg["metrics"]["ks"], cfg["paths"]["reports"]
    events = load_train(cfg["paths"]["processed"] / "splits", b["train_splits"])
    if argv[0] == "tune":
        rows = []
        for name in MODELS:
            for params in grid_points(b[name]["grid"]):
                metrics = score(fit_recommend(name, events, b, params), "val")
                rows.append({"model": name, "params": params, "metrics": metrics})
                print(f"{name} {params} {b['select_on']}={metrics[b['select_on']]:.4f}", flush=True)
        (reports / "baselines_tuning.md").write_text(render_tuning(rows, b["select_on"], ks))
        print(f"wrote {reports / 'baselines_tuning.md'}")
    else:
        results = {}
        for name in MODELS:
            recs = fit_recommend(name, events, b, b[name]["chosen"])
            results[name] = {"params": b[name]["chosen"], "val": score(recs, "val"), "test": score(recs, "test")}
        # 6 places in the file: the last digits of a float mean change from run to run
        stable = json.loads(json.dumps(results), parse_float=lambda x: round(float(x), 6))
        (reports / "baselines.json").write_text(json.dumps(stable, indent=2) + "\n")
        (reports / "baselines.md").write_text(render_report(results, b["train_splits"], ks))
        print(f"wrote {reports / 'baselines.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
