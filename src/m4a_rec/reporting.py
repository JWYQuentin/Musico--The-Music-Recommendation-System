"""Small helpers shared by the modules that tune models and write reports."""
from __future__ import annotations

import itertools
import json


def grid_points(grid: dict) -> list[dict]:
    """Every combination of the listed values, one dict per combination."""
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def table(header: list[str], rows: list[list]) -> list[str]:
    """Markdown table lines, floats to 4 places, followed by a blank line."""
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows] + [""]


TITLES = {
    "recall": "Recall@K",
    "ndcg": "NDCG@K",
    "coverage": "Coverage@K: share of training tracks that reach some user's top K",
    "short": "Short@K: share of users handed fewer than K tracks",
}


def metric_tables(
    results: dict, ks: list[int], splits: tuple[str, ...] = ("val", "test"), prefix: str = ""
) -> list[str]:
    """One table per split and metric, a row per model. `results` maps model to {split: metrics}."""
    L = []
    for split in splits:
        users = next(iter(results.values()))[split]["users"]
        L += [f"## {prefix}{split} ({users:,} users scored)", ""]
        for metric, title in TITLES.items():
            L += [title, ""]
            L += table(
                ["Model", *(f"@{k}" for k in ks)],
                [[name, *(r[split][f"{metric}@{k}"] for k in ks)] for name, r in results.items()],
            )
    return L


def stable_json(results: dict) -> str:
    """JSON text with floats cut to 10 places: the last digits of a float mean change from run to run."""
    return json.dumps(json.loads(json.dumps(results), parse_float=lambda x: round(float(x), 10)), indent=2) + "\n"
