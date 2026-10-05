"""Phase 1 data audit: writes reports/audit.md and three figures.

Answers the questions later phases depend on:
  - how much data per user (can a user tower learn from it?)
  - how skewed popularity is (how strong is the popularity baseline?)
  - how much listening is repeat plays (what should "relevant" mean in evaluation?)
  - whether the last months are complete (is the test window trustworthy?)
  - how many tracks have each content feature (how real is the cold-start story?)

Usage: python -m m4a_rec.audit
"""
from __future__ import annotations

import bz2
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .config import load_config

INK, MUTED, GRID, SURFACE, BLUE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array (0 = equal, 1 = one item has everything)."""
    v = np.sort(values.astype(np.float64))
    n = v.size
    if n == 0 or v.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def feature_ids(path: Path) -> set[str]:
    """First column of a feature .tsv.bz2 (the track id), skipping a header row."""
    ids: set[str] = set()
    with bz2.open(path, "rt") as f:
        for i, line in enumerate(f):
            tid = line.split("\t", 1)[0].strip()
            if i == 0 and tid.lower() in {"id", "track_id"}:
                continue
            if tid:
                ids.add(tid)
    return ids


def compute(events: pl.DataFrame) -> dict:
    n = events.height
    per_user = events.group_by("user_id").agg(
        listens=pl.len(), unique_tracks=pl.col("track_id").n_unique()
    )
    per_track = events.group_by("track_id").len(name="listens").sort("listens", descending=True)
    pairs = events.select(["user_id", "track_id"]).unique().height
    monthly = (
        events.group_by(pl.col("timestamp").dt.truncate("1mo").alias("month"))
        .len(name="listens")
        .sort("month")
    )
    pop = per_track["listens"].to_numpy()
    top1 = max(1, int(round(0.01 * pop.size)))
    q = lambda s, p: float(s.quantile(p))  # noqa: E731
    # Complete months only: the first and last month of the window are usually partial.
    inner = monthly["listens"].to_numpy()[1:-1]
    return {
        "listens": n,
        "users": per_user.height,
        "tracks": per_track.height,
        "first": events["timestamp"].min(),
        "last": events["timestamp"].max(),
        "density": pairs / (per_user.height * per_track.height),
        "user_listens": {p: q(per_user["listens"], p) for p in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "user_unique_tracks_median": q(per_user["unique_tracks"], 0.5),
        "repeat_share": 1 - pairs / n,
        "unique_pairs": pairs,
        "top1pct_share": float(pop[:top1].sum() / n),
        "gini": gini(pop),
        "tracks_lt5": float((pop < 5).mean()),
        "month_cv": float(inner.std() / inner.mean()) if inner.size > 1 else float("nan"),
        "_per_user": per_user,
        "_per_track": per_track,
        "_monthly": monthly,
    }


def coverage(events: pl.DataFrame, per_track: pl.DataFrame, files: dict[str, Path]) -> list[dict]:
    rows = []
    n_listens, n_tracks = events.height, per_track.height
    for name, path in files.items():
        if not path.exists():
            rows.append({"feature": name, "file": path.name, "missing": True})
            continue
        ids = feature_ids(path)
        hit = per_track.filter(pl.col("track_id").is_in(list(ids)))
        rows.append(
            {
                "feature": name,
                "file": path.name,
                "missing": False,
                "ids_in_file": len(ids),
                "track_cov": hit.height / n_tracks,
                "listen_cov": int(hit["listens"].sum()) / n_listens,
            }
        )
    return rows


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def figures(stats: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        fig.savefig(out / name, dpi=160)
        plt.close(fig)

    listens = stats["_per_user"]["listens"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.logspace(np.log10(listens.min()), np.log10(listens.max() + 1), 40)
    ax.hist(listens, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.set_xscale("log")
    _style(ax, "Listens per user", "listens in window (log scale)", "users")
    save(fig, "listens_per_user.png")

    pop = stats["_per_track"]["listens"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(np.arange(1, pop.size + 1), pop, color=BLUE, linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "Track popularity long tail", "track rank (log scale)", "listens (log scale)")
    save(fig, "track_popularity.png")

    m = stats["_monthly"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = [d.strftime("%Y-%m") for d in m["month"].to_list()]
    ax.bar(labels, m["listens"].to_numpy(), color=BLUE, width=0.7)
    ax.set_ylim(bottom=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _style(ax, "Listens per month", "", "listens")
    save(fig, "listens_per_month.png")


def render(stats: dict, cov: list[dict], meta: dict | None) -> str:
    u = stats["user_listens"]
    L = [
        "# Phase 1 data audit",
        "",
        "Generated by `python -m m4a_rec.audit`. Do not edit by hand; rerun instead.",
        "",
        "## Size",
        "",
        "| | |",
        "|---|---|",
        f"| Listens | {stats['listens']:,} |",
        f"| Users | {stats['users']:,} |",
        f"| Tracks | {stats['tracks']:,} |",
        f"| First listen | {stats['first']} |",
        f"| Last listen | {stats['last']} |",
        f"| Matrix density | {stats['density']:.4%} |",
        "",
    ]
    if meta:
        L += [
            f"Sampled {meta['users_sampled']:,} of {meta['users_eligible']:,} eligible users "
            f"({meta['users_in_window']:,} active in the window; minimum "
            f"{meta['min_user_listens']} listens; seed {meta['seed']}).",
            "",
        ]
    L += [
        "## Listens per user",
        "",
        "| p5 | p25 | median | p75 | p95 |",
        "|---|---|---|---|---|",
        "| " + " | ".join(f"{u[p]:,.0f}" for p in (0.05, 0.25, 0.5, 0.75, 0.95)) + " |",
        "",
        f"Median unique tracks per user: {stats['user_unique_tracks_median']:,.0f}.",
        "",
        "![Listens per user](figures/listens_per_user.png)",
        "",
        "## Popularity",
        "",
        f"- The top 1% of tracks take {stats['top1pct_share']:.1%} of listens.",
        f"- Gini coefficient of track listens: {stats['gini']:.3f}.",
        f"- {stats['tracks_lt5']:.1%} of tracks have fewer than 5 listens.",
        "",
        "![Track popularity](figures/track_popularity.png)",
        "",
        "## Repeat plays",
        "",
        f"- {stats['repeat_share']:.1%} of listens repeat a (user, track) pair already in the window.",
        f"- Unique (user, track) pairs: {stats['unique_pairs']:,}.",
        "",
        "## Volume over time",
        "",
        f"Month-to-month variation (CV over complete months): {stats['month_cv']:.2f}. "
        "The first and last bars are partial months.",
        "",
        "![Listens per month](figures/listens_per_month.png)",
        "",
        "## Content-feature coverage",
        "",
        "| Feature | File | Tracks covered | Listens covered |",
        "|---|---|---|---|",
    ]
    for r in cov:
        if r["missing"]:
            L.append(f"| {r['feature']} | `{r['file']}` | not downloaded | not downloaded |")
        else:
            L.append(
                f"| {r['feature']} | `{r['file']}` | {r['track_cov']:.1%} | {r['listen_cov']:.1%} |"
            )
    return "\n".join(L) + "\n"


def main() -> int:
    cfg = load_config()
    path = cfg["paths"]["processed"] / "events.parquet"
    events = pl.read_parquet(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    stats = compute(events)
    files = {k: cfg["paths"]["raw"] / v for k, v in cfg["feature_files"].items()}
    cov = coverage(events, stats["_per_track"], files)
    reports = cfg["paths"]["reports"]
    figures(stats, reports / "figures")
    (reports / "audit.md").write_text(render(stats, cov, meta))
    print(f"wrote {reports / 'audit.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
