"""The second stage: an XGBoost ranker that reorders each user's candidates.

It is trained on the `train` snapshot from candidates.py, where a candidate's label says
whether the user went on to play it in ranker_train, and applied to the `infer` snapshot,
whose reordered lists go to the evaluator.

A model is a name and the feature groups it is trained on, listed under `models` in
configs/ranker.yaml: `ranker` is the Phase 5 model, `ranker_artist` adds the artist group.

XGBoost crashes when loaded into a process that has already loaded PyTorch (two OpenMP
runtimes on macOS). So this module imports nothing that imports torch, it reads other
models' output from files, and its tests run in their own process.

  tune <model>    the stages in configs/ranker.yaml, scored on val. Writes
                  reports/<model>_tuning.md. Settings already in reports/<model>_tuning.json
                  are not run again.
  ablate <model>  the chosen setting retrained without each of its feature groups, scored
                  on val. Writes reports/<model>_ablation.json.
  fit <model>     trains the chosen setting, keeps the tree count that scores best on val,
                  and saves the model under data/processed/models/<model>/ and its lists
                  in data/processed/recs/<model>.parquet.
  val             every saved list on val. Prints; writes nothing.
  report          the same on val and test. Writes reports/ranker.md, the SHAP figure and
                  the chart of every model's result.

Usage: python -m m4a_rec.ranker tune|ablate|fit <model>   or   python -m m4a_rec.ranker val|report
"""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xgboost as xgb

from .audit import GRID, INK, MUTED, SURFACE
from .config import load_config
from .evaluate import PAIR, difference, score
from .rank_features import GROUPS
from .reporting import metric_tables, stable_json, staged_search, table

OPPONENTS = ["popularity", "item_knn", "als", "tt_id"]  # their lists are in data/processed/recs/


def shown(model: str) -> str:
    """A ranker's name in tables: it is always the second stage on top of tt_id."""
    return f"tt_id + {model}"


def features_of(groups: list[str]) -> list[str]:
    """The feature columns of the named groups, in the order given."""
    return [name for group in groups for name in GROUPS[group]]


def training_rows(train: pl.DataFrame, n_candidates: int) -> pl.DataFrame:
    """Each user's first `n_candidates` candidates, for users with a positive among them.

    A user with no positive gives a ranking loss nothing to compare, so they are left out.
    """
    top = train.filter(pl.col("tt_rank") <= n_candidates)
    with_positive = top.select("user_id", "label").filter(pl.col("label") == 1).select("user_id").unique()
    return top.join(with_positive, on="user_id", how="semi", maintain_order="left")


def to_dmatrix(frame: pl.DataFrame, features: list[str], label: bool = False) -> xgb.DMatrix:
    """XGBoost's input. With `label`, for training: rows must be grouped by user, as the snapshots are."""
    data = frame.select(features).to_numpy()
    if not label:
        return xgb.DMatrix(data, feature_names=features)
    return xgb.DMatrix(data, label=frame["label"].to_numpy(), qid=frame["user_id"].to_numpy(), feature_names=features)


def train_booster(rows: pl.DataFrame, features: list[str], params: dict, cfg: dict) -> xgb.Booster:
    settings = {
        **cfg["fixed"],
        "objective": params["objective"],
        "max_depth": params["max_depth"],
        "eta": params["eta"],
        "seed": cfg["seed"],
    }
    if params["objective"].startswith("rank:"):
        settings["lambdarank_pair_method"] = params["lambdarank_pair_method"]
    return xgb.train(settings, to_dmatrix(rows, features, label=True), num_boost_round=cfg["max_rounds"])


def rerank(infer: pl.DataFrame, scores: np.ndarray, n_candidates: int) -> pl.DataFrame:
    """The final lists: each user's first `n_candidates` candidates by `scores`, then the rest in retrieval order.

    `scores` has one value per row of `infer` with tt_rank <= n_candidates, in row order.
    Candidates the ranker scores the same keep their retrieval order. The score handed on is
    minus the track's place in the final list, so nothing later has to break a tie.
    """
    keys = infer.select(*PAIR, "tt_rank")  # before filtering, so the feature columns are never copied
    top = keys.filter(pl.col("tt_rank") <= n_candidates).with_columns(model=pl.Series(scores))
    top = top.sort(["user_id", "model", "tt_rank"], descending=[False, True, False]).select(
        *PAIR, score=-pl.int_range(1, pl.len() + 1).over("user_id").cast(pl.Float32)
    )
    rest = keys.filter(pl.col("tt_rank") > n_candidates).select(*PAIR, score=-pl.col("tt_rank").cast(pl.Float32))
    return pl.concat([top, rest])


def run(
    train: pl.DataFrame,
    infer: pl.DataFrame,
    features: list[str],
    params: dict,
    cfg: dict,
    evaluate: Callable[[pl.DataFrame], dict],
    log: Callable[[str], None] = print,
) -> dict:
    """Train one ranker and keep the number of trees whose lists `evaluate` scores best.

    `params` is one setting from the config, `cfg` the `ranker` section. The model is
    trained once to `max_rounds` trees; its lists at every `check_every` trees go to
    `evaluate`, and the count with the best `select_on` is kept.
    """
    started = time.time()
    rows = training_rows(train, params["n_candidates"])
    booster = train_booster(rows, features, params, cfg)
    log(f"  trained on {rows.height:,} rows of {rows['user_id'].n_unique():,} users in {time.time() - started:.0f}s")
    keys = infer.select(*PAIR, "tt_rank")  # all that reordering needs
    top = to_dmatrix(infer.select(list(dict.fromkeys(["tt_rank", *features]))).filter(pl.col("tt_rank") <= params["n_candidates"]), features)
    best, history = None, []
    for rounds in range(cfg["check_every"], cfg["max_rounds"] + 1, cfg["check_every"]):
        scores = booster.predict(top, iteration_range=(0, rounds))
        metrics = evaluate(rerank(keys, scores, params["n_candidates"]))
        history.append({"rounds": rounds, cfg["select_on"]: metrics[cfg["select_on"]]})
        if best is None or metrics[cfg["select_on"]] > best["metrics"][cfg["select_on"]]:
            best = {"rounds": rounds, "metrics": metrics, "scores": scores}
    log(f"  {cfg['select_on']} by trees: " + ", ".join(f"{h['rounds']}: {h[cfg['select_on']]:.4f}" for h in history))
    return {**best, "booster": booster, "history": history}


def save(booster: xgb.Booster, features: list[str], params: dict, rounds: int, directory: Path) -> None:
    """The trained model and what is needed to use it: its features, its setting and how many trees to use."""
    directory.mkdir(parents=True, exist_ok=True)
    booster.save_model(directory / "model.json")
    (directory / "meta.json").write_text(json.dumps({"features": features, "params": params, "trees": rounds}, indent=2) + "\n")


def load(directory: Path) -> tuple[xgb.Booster, dict]:
    """A saved model and its meta: {features, params, trees}."""
    booster = xgb.Booster()
    booster.load_model(directory / "model.json")
    return booster, json.loads((directory / "meta.json").read_text())


def predict(booster: xgb.Booster, meta: dict, infer: pl.DataFrame) -> np.ndarray:
    """The saved model's score for each row of `infer` it reorders (tt_rank <= n_candidates), in row order."""
    top = infer.filter(pl.col("tt_rank") <= meta["params"]["n_candidates"])
    return booster.predict(to_dmatrix(top, meta["features"]), iteration_range=(0, meta["trees"]))


def shap_values(booster: xgb.Booster, frame: pl.DataFrame, features: list[str], rounds: int) -> np.ndarray:
    """Each feature's contribution to each row's score (exact TreeSHAP). The last column is the base value."""
    return booster.predict(to_dmatrix(frame, features), pred_contribs=True, iteration_range=(0, rounds))


def shap_summary(booster: xgb.Booster, frame: pl.DataFrame, features: list[str], rounds: int, n_rows: int, seed: int) -> dict:
    """Mean absolute contribution of each feature over a seeded sample of rows, largest first."""
    sample = frame.sample(n=min(n_rows, frame.height), seed=seed)
    mean = np.abs(shap_values(booster, sample, features, rounds)[:, :-1]).mean(axis=0)
    return dict(sorted(zip(features, mean.tolist()), key=lambda item: -item[1]))


def explain(booster: xgb.Booster, row: pl.DataFrame, features: list[str], rounds: int) -> list[tuple]:
    """(feature, value, contribution) for a one-row frame, largest contribution first."""
    contributions = shap_values(booster, row, features, rounds)[0, :-1]
    values = row.select(features).row(0)
    return sorted(zip(features, values, contributions.tolist()), key=lambda item: -abs(item[2]))


def shap_figure(summary: dict, path: Path) -> None:
    """Ranked bars of mean absolute SHAP value, each labelled with its feature and group."""
    group_of = {f: g for g, names in GROUPS.items() for f in names}
    names, values = list(summary)[::-1], list(summary.values())[::-1]  # barh draws from the bottom up
    fig, ax = plt.subplots(figsize=(6.4, 0.26 * len(names) + 1.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh([f"{n} ({group_of[n]})" for n in names], values, height=0.62, color="#2a78d6")
    for y, value in enumerate(values):
        ax.text(value + max(values) * 0.01, y, f"{value:.3f}", va="center", fontsize=7, color=MUTED)
    ax.set_title("What the ranker uses: mean absolute SHAP value", loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel("average change to a candidate's score", fontsize=9, color=MUTED)
    ax.set_xlim(0, max(values) * 1.12)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def results_figure(results: dict, final: str, split: str, metric: str, path: Path) -> None:
    """One bar per model for `metric` on `split`, best last, with the final model picked out."""
    names = sorted(results, key=lambda name: results[name][split][metric])  # barh draws from the bottom up
    values = [results[name][split][metric] for name in names]
    fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(names) + 1.3))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(names, values, height=0.5, color=["#2a78d6" if name == final else "#c3c2b7" for name in names])
    for y, value in enumerate(values):
        ax.text(value + max(values) * 0.01, y, f"{value:.4f}", va="center", fontsize=8, color=INK if names[y] == final else MUTED)
    ax.set_title(f"{metric.upper().replace('@', ' at ')} on {split}: what each stage adds", loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlim(0, max(values) * 1.14)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=MUTED, labelsize=9, length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_tuning(rows: list[dict], select_on: str, ks: list[int], model: str) -> str:
    """Every tuning run of `model`, best first. Each row is {params, rounds, metrics}."""
    shown = list(dict.fromkeys([select_on, f"recall@{ks[0]}", *(f"recall@{k}" for k in ks[1:3]), f"coverage@{ks[0]}"]))
    rows = sorted(rows, key=lambda r: -r["metrics"][select_on])
    keys = list(rows[0]["params"])
    L = [
        f"# Ranker tuning: `{model}`",
        "",
        f"Generated by `python -m m4a_rec.ranker tune {model}`. Do not edit by hand; rerun instead.",
        "",
        f"Scored on val, sorted by {select_on}, best first. `trees` is the number of trees kept.",
        "",
    ]
    L += table(
        [*keys, "trees", *shown],
        [[*(r["params"][k] for k in keys), r["rounds"], *(r["metrics"][m] for m in shown)] for r in rows],
    )
    return "\n".join(L)


def render_report(
    results: dict,
    gaps: list[dict],
    before: dict,
    final: str,
    summary: dict,
    ablation: list[dict],
    example: dict,
    splits: tuple[str, ...],
    ks: list[int],
    select_on: str,
    n_boot: int,
) -> str:
    """The ranker report.

    results   model -> {params, <split>: metrics}; rankers under their `shown` names
    gaps      rows of {model, against, <split>: {metric: {diff, low, high}}}: one model minus another
    before    model -> {<split>: metrics} for the same models fitted on retrieval_train only
    final     the ranker the last three sections are about, under its `shown` name
    summary   feature -> mean absolute SHAP value, largest first
    ablation  rows of {without, rounds, metrics} on val, possibly empty
    example   {user_id, track_id, tt_rank, parts: [(feature, value, contribution), ...]}
    """
    group_of = {f: g for g, names in GROUPS.items() for f in names}
    L = [
        "# Ranker",
        "",
        "Generated by `python -m m4a_rec.ranker report`. Do not edit by hand; rerun instead.",
        "",
        "Every model here is fitted on both train splits. A ranker reorders each user's first",
        "candidates from `tt_id`; the rest keep their order. `ranker` is the Phase 5 model and",
        "`ranker_artist` adds what the user has played by each candidate's artist.",
        "",
    ]
    L += table(["Model", "Setting"], [[name, json.dumps(r["params"])] for name, r in results.items()])
    L += metric_tables(results, ks, splits)

    for gap in gaps:
        a, b = gap["model"], gap["against"]
        L += [f"## `{a}` against `{b}`", ""]
        L += [f"`{a}` minus `{b}`, with a 95% interval from resampling users {n_boot:,} times.", ""]
        rows = []
        for split in splits:
            for metric in [f"{m}@{k}" for m in ("ndcg", "recall") for k in ks[:3]]:
                g, base = gap[split][metric], results[b][split][metric]
                rows.append([split, metric, base, results[a][split][metric], g["diff"], f"{g['low']:+.4f} to {g['high']:+.4f}", f"{g['diff'] / base:+.1%}"])
        L += table(["Split", "Metric", b, a, "Difference", "95% interval", "Relative"], rows)

    L += ["## What four more weeks of training data are worth", ""]
    L += ["The same model and setting, fitted on `retrieval_train` only (Phases 3 and 4) and on both train splits.", ""]
    rows = []
    for split in splits:
        for name, old in before.items():
            for metric in (f"ndcg@{ks[0]}", f"recall@{ks[2]}"):
                a, b = old[split][metric], results[name][split][metric]
                rows.append([split, name, metric, a, b, f"{b / a - 1:+.1%}"])
    L += table(["Split", "Model", "Metric", "retrieval_train only", "Both train splits", "Change"], rows)

    L += [f"## What `{final}` uses", ""]
    L += [
        "Mean absolute SHAP value: how far a feature moves a candidate's score on average.",
        "A user feature is the same for all of a user's candidates, so on its own it moves the whole",
        "list and changes no order; it matters only combined with other features. The ablation",
        "below measures that.",
        "",
    ]
    L += ["![SHAP values](figures/ranker_shap.png)", ""]
    total = sum(summary.values())
    L += table(["Feature", "Group", "Mean absolute SHAP", "Share"], [[f, group_of[f], v, f"{v / total:.1%}"] for f, v in summary.items()])
    by_group = {g: sum(summary[f] for f in names if f in summary) for g, names in GROUPS.items()}
    L += table(["Group", "Share"], [[g, f"{v / total:.1%}"] for g, v in sorted(by_group.items(), key=lambda item: -item[1]) if v])

    if ablation:
        full = results[final]["val"]
        L += [f"## What `{final}` needs", ""]
        L += ["The chosen setting retrained without one feature group at a time, on val.", ""]
        cols = [select_on, f"recall@{ks[0]}", f"recall@{ks[2]}"]
        rows = [["nothing (the full model)", *(full[m] for m in cols), ""]]
        rows += [[r["without"], *(r["metrics"][m] for m in cols), f"{r['metrics'][select_on] / full[select_on] - 1:+.1%}"] for r in ablation]
        L += table(["Without", *cols, f"Change in {select_on}"], rows)

    L += ["## One recommendation explained", ""]
    L += [
        f"User {example['user_id']}'s top track after reordering was candidate number {example['tt_rank']} from retrieval.",
        "The contributions below, with the model's base value, add up to its score.",
        "",
    ]
    L += table(["Feature", "Value", "Contribution"], [[f, "missing" if v != v else float(v), c] for f, v, c in example["parts"]])
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cfg = load_config()
    r, ks, reports, processed = cfg["ranker"], cfg["metrics"]["ks"], cfg["paths"]["reports"], cfg["paths"]["processed"]
    one_model = len(argv) == 2 and argv[0] in ("tune", "ablate", "fit") and argv[1] in r["models"]
    if not (one_model or argv in (["val"], ["report"])):
        print(__doc__)
        return 2
    infer = pl.read_parquet(processed / "ranker" / "infer.parquet")
    say = lambda message: print(message, flush=True)  # noqa: E731

    if one_model:
        command, model = argv
        train = pl.read_parquet(processed / "ranker" / "train.parquet")
        groups, chosen = r["models"][model], r["chosen"][model]

        def on_val(features: list[str], params: dict) -> dict:
            return run(train, infer, features, params, r, lambda recs: score(recs, "val"), log=say)

        if command == "tune":
            saved = reports / f"{model}_tuning.json"
            rows = json.loads(saved.read_text()) if saved.exists() else []

            def tried(params: dict) -> float:
                done = next((row for row in rows if row["params"] == params), None)
                if done is None:
                    say(f"{model} {params}")
                    result = on_val(features_of(groups), params)
                    done = {"params": params, "rounds": result["rounds"], "metrics": result["metrics"]}
                    rows.append(done)
                    saved.write_text(json.dumps(rows, indent=2) + "\n")
                return done["metrics"][r["select_on"]]

            best, _ = staged_search(chosen, r["stages"], tried)
            (reports / f"{model}_tuning.md").write_text(render_tuning(rows, r["select_on"], ks, model))
            say(f"best: {best}\nwrote {reports / f'{model}_tuning.md'}")
        elif command == "ablate":
            rows = []
            for group in groups:
                say(f"{model} without {group}")
                result = on_val(features_of([g for g in groups if g != group]), chosen)
                rows.append({"without": group, "rounds": result["rounds"], "metrics": result["metrics"]})
                (reports / f"{model}_ablation.json").write_text(stable_json(rows))
        else:
            say(f"{model} {chosen}")
            result = on_val(features_of(groups), chosen)
            save(result["booster"], features_of(groups), chosen, result["rounds"], processed / "models" / model)
            rerank(infer, result["scores"], chosen["n_candidates"]).write_parquet(processed / "recs" / f"{model}.parquet")
            say(f"kept {result['rounds']} trees; saved the model and its lists")
        return 0

    splits = ("val",) if argv == ["val"] else ("val", "test")
    models, final = list(r["models"]), r["final"]
    saved = {model: load(processed / "models" / model) for model in models}
    lists = {name: pl.read_parquet(processed / "recs" / f"{name}.parquet") for name in [*OPPONENTS, *models]}
    settings = {**{name: cfg["baselines"][name]["chosen"] for name in OPPONENTS[:3]}, "tt_id": cfg["twotower"]["chosen"]["tt_id"]}
    settings |= {model: {**meta["params"], "trees": meta["trees"]} for model, (_, meta) in saved.items()}
    label = lambda name: shown(name) if name in models else name  # noqa: E731
    results = {label(name): {"params": settings[name], **{s: score(recs, s) for s in splits}} for name, recs in lists.items()}
    # each ranker against retrieval order, then the final ranker against each other one
    pairs = [(model, "tt_id") for model in models] + [(final, model) for model in models if model != final]
    gaps = [{"model": label(a), "against": label(b), **{s: difference(lists[a], lists[b], s) for s in splits}} for a, b in pairs]
    before = {name: row for name, row in json.loads((reports / "retrieval.json").read_text()).items() if name in OPPONENTS}

    booster, meta = saved[final]
    top = infer.filter(pl.col("tt_rank") <= meta["params"]["n_candidates"])
    summary = shap_summary(booster, top, meta["features"], meta["trees"], r["shap_rows"], r["seed"])
    # the example: the first user whose top track after reordering came from below retrieval's top K
    tops = top.with_columns(score=pl.Series(predict(booster, meta, infer))).filter(pl.col("score") == pl.col("score").max().over("user_id"))
    lifted = tops.filter(pl.col("tt_rank") > ks[0])
    best_row = (lifted if lifted.height else tops).head(1)
    example = {
        "user_id": best_row["user_id"][0],
        "track_id": best_row["track_id"][0],
        "tt_rank": int(best_row["tt_rank"][0]),
        "parts": explain(booster, best_row, meta["features"], meta["trees"])[:8],
    }
    ablated = reports / f"{final}_ablation.json"
    ablation = json.loads(ablated.read_text()) if ablated.exists() else []
    text = render_report(results, gaps, before, label(final), summary, ablation, example, splits, ks, r["select_on"], cfg["bootstrap"]["n"])
    if argv == ["val"]:
        say(text)
        return 0
    shap_figure(summary, reports / "figures" / "ranker_shap.png")
    results_figure(results, label(final), "test", r["select_on"], reports / "figures" / "results.png")
    (reports / "ranker.json").write_text(stable_json({"results": results, "gaps": gaps, "shap": summary}))
    (reports / "ranker.md").write_text(text)
    say(f"wrote {reports / 'ranker.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
