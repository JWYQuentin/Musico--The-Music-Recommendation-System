"""Train a two-tower model and turn its vectors into recommendations.

  tune <model>  the stages in configs/twotower.yaml for tt_id or tt_hybrid, scored on val.
                Writes reports/twotower_tuning.md.
  report        tt_id and tt_hybrid with their chosen settings, scored on val and test beside
                the Phase 3 baselines. Writes reports/retrieval.md and saves each model's
                vectors under data/processed/models/.

Usage: python -m m4a_rec.retrieval tune tt_id | tune tt_hybrid | report
"""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch

from .config import load_config
from .evaluate import score
from .features import load as load_features
from .interactions import Interactions, build, load_train, to_frame
from .reporting import grid_points, metric_tables, stable_json, table
from .twotower import TrackTower, UserTower, in_batch_softmax_loss

BATCH = 1000  # users scored at a time when ranking tracks
EMBED_BATCH = 8192  # rows per forward pass when collecting vectors


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def training_pairs(
    matrix: sp.csr_matrix, sampling: str, exclude_tracks: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row index, column index and sampling weight of every training pair, and log q per track.

    `matrix` holds log(1 + plays). The weight is 1 for "pairs" and the cell value for
    "log_plays". q is each track's share of the total weight: how often it turns up in a batch.
    """
    coo = matrix.tocoo()
    keep = np.ones(coo.nnz, dtype=bool) if exclude_tracks is None else ~np.isin(coo.col, exclude_tracks)
    users, tracks = coo.row[keep].astype(np.int64), coo.col[keep].astype(np.int64)
    weights = np.ones(len(users)) if sampling == "pairs" else coo.data[keep].astype(np.float64)
    q = np.bincount(tracks, weights=weights, minlength=matrix.shape[1]) / weights.sum()
    # A track with no pairs never enters a batch, so its value is unused; the floor avoids log(0).
    return users, tracks, weights, np.log(np.maximum(q, 1e-12)).astype(np.float32)


def embed(user_tower: UserTower, track_tower: TrackTower, n_users: int, n_tracks: int) -> tuple[np.ndarray, np.ndarray]:
    """Every user's and every track's vector, as numpy arrays on the CPU."""
    device = user_tower.embedding.weight.device

    def run(tower, n):
        tower.eval()
        with torch.no_grad():
            chunks = [
                tower(torch.arange(start, min(start + EMBED_BATCH, n), device=device)).cpu()
                for start in range(0, n, EMBED_BATCH)
            ]
        return torch.cat(chunks).numpy()

    return run(user_tower, n_users), run(track_tower, n_tracks)


def top_unseen(
    user_vecs: np.ndarray,
    track_vecs: np.ndarray,
    matrix: sp.csr_matrix,
    n: int,
    candidates: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """For each user, the n best-scoring tracks they have not played: (users, n) indices and scores.

    A score is the dot product of the two vectors. `candidates` limits the choice to those
    track indices. A user with fewer than n tracks left gets index -1 in the spare places.
    """
    users, tracks = torch.from_numpy(user_vecs), torch.from_numpy(track_vecs)
    n = min(n, tracks.shape[0])
    blocked = None
    if candidates is not None:
        blocked = torch.full((tracks.shape[0],), float("-inf"))
        blocked[torch.from_numpy(np.asarray(candidates))] = 0
    idx, scores = [], []
    for start in range(0, users.shape[0], BATCH):
        s = users[start : start + BATCH] @ tracks.T
        if blocked is not None:
            s += blocked
        rows, cols = matrix[start : start + BATCH].nonzero()
        s[torch.from_numpy(rows.astype(np.int64)), torch.from_numpy(cols.astype(np.int64))] = float("-inf")
        top = s.topk(n, dim=1)
        idx.append(torch.where(top.values == float("-inf"), -1, top.indices))
        scores.append(top.values)
    return torch.cat(idx).numpy(), torch.cat(scores).numpy()


def recommend(
    inter: Interactions, user_vecs: np.ndarray, track_vecs: np.ndarray, n: int, candidates: np.ndarray | None = None
) -> pl.DataFrame:
    idx, scores = top_unseen(user_vecs, track_vecs, inter.matrix, n, candidates)
    return to_frame(inter, idx, scores)


def fit(
    inter: Interactions,
    features: np.ndarray | None,
    spec: dict,
    params: dict,
    cfg: dict,
    evaluate: Callable[[np.ndarray, np.ndarray], dict],
    exclude_tracks: np.ndarray | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Train one model, keeping the vectors from its best epoch.

    `inter.matrix` must hold log(1 + plays). `spec` is the model's entry under `models` in
    the config, `params` its setting, `cfg` the `twotower` section. After every epoch
    `evaluate(user_vecs, track_vecs)` returns metrics; training stops once
    `cfg["select_on"]` has not improved for `cfg["patience"]` epochs. Pairs whose track is
    in `exclude_tracks` are left out of training.
    """
    torch.manual_seed(cfg["seed"])
    draw = torch.Generator().manual_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    n_users, n_tracks = inter.matrix.shape
    user_tower = UserTower(n_users, params["dim"]).to(device)
    track_tower = TrackTower(
        n_tracks,
        params["dim"],
        use_id=spec["use_id"],
        features=torch.from_numpy(features) if spec["use_content"] else None,
        hidden=params["hidden"],
        dropout=params["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam([*user_tower.parameters(), *track_tower.parameters()], lr=params["lr"])

    users, tracks, weights, log_q = training_pairs(inter.matrix, params["sampling"], exclude_tracks)
    users, tracks = torch.from_numpy(users).to(device), torch.from_numpy(tracks).to(device)
    log_q, weights = torch.from_numpy(log_q).to(device), torch.from_numpy(weights)
    n_pairs, batch_size = len(users), cfg["batch_size"]

    best, history = None, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        started = time.time()
        user_tower.train()
        track_tower.train()
        if params["sampling"] == "pairs":
            order = torch.randperm(n_pairs, generator=draw)
        else:
            order = torch.multinomial(weights, n_pairs, replacement=True, generator=draw)
        order = order.to(device)
        total = torch.zeros((), device=device)
        for start in range(0, n_pairs, batch_size):
            batch = order[start : start + batch_size]
            u, t = users[batch], tracks[batch]
            loss = in_batch_softmax_loss(
                user_tower(u),
                track_tower(t),
                params["temperature"],
                log_q=log_q[t] if params["log_q"] else None,
                track_idx=t if params["mask_duplicates"] else None,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.detach()

        user_vecs, track_vecs = embed(user_tower, track_tower, n_users, n_tracks)
        metrics = evaluate(user_vecs, track_vecs)
        value = metrics[cfg["select_on"]]
        steps = -(-n_pairs // batch_size)
        history.append({"epoch": epoch, "loss": total.item() / steps, cfg["select_on"]: value})
        log(
            f"  epoch {epoch}: loss {history[-1]['loss']:.4f}, {cfg['select_on']} {value:.4f}, "
            f"{time.time() - started:.0f}s"
        )
        if best is None or value > best["metrics"][cfg["select_on"]]:
            best = {"epoch": epoch, "metrics": metrics, "user_vecs": user_vecs, "track_vecs": track_vecs}
        elif epoch - best["epoch"] >= cfg["patience"]:
            break
    return {**best, "history": history}


def staged_search(start: dict, stages: list[dict], run: Callable[[dict], float]) -> tuple[dict, list]:
    """Tune one stage at a time. `run(params)` returns the metric to maximise.

    Each stage tries every combination of its values on top of the best setting found in
    the stages before it. Returns the best setting and every (params, metric) tried, in order.
    """
    best, tried = dict(start), []
    for stage in stages:
        for point in grid_points(stage):
            params = {**best, **point}
            if all(params != seen for seen, _ in tried):
                tried.append((params, run(params)))
        best = max(tried, key=lambda t: t[1])[0]
    return best, tried


def save(directory: Path, inter: Interactions, result: dict, params: dict) -> None:
    """The vectors and the IDs their rows belong to. Phase 5 and the FAISS index read these."""
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "user_vecs.npy", result["user_vecs"])
    np.save(directory / "track_vecs.npy", result["track_vecs"])
    np.save(directory / "user_ids.npy", inter.user_ids.to_numpy())
    np.save(directory / "track_ids.npy", np.array(inter.track_ids.to_list()))
    meta = {"params": params, "epoch": result["epoch"], "history": result["history"]}
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def render_tuning(runs: dict, select_on: str, ks: list[int]) -> str:
    """One table per model, best run first. `runs` maps model to rows of {params, epoch, metrics}."""
    shown = list(dict.fromkeys([select_on, *(f"recall@{k}" for k in ks), f"ndcg@{ks[0]}", f"coverage@{ks[0]}"]))
    L = [
        "# Phase 4 two-tower tuning",
        "",
        "Generated by `python -m m4a_rec.retrieval tune <model>`. Do not edit by hand; rerun instead.",
        "",
        f"Scored on val. Each table is sorted by {select_on}, best first. `epoch` is the epoch kept.",
        "",
    ]
    for name, rows in runs.items():
        rows = sorted(rows, key=lambda r: -r["metrics"][select_on])
        keys = list(rows[0]["params"])
        L += [f"## {name}", ""]
        L += table(
            [*keys, "epoch", *shown],
            [[*(r["params"][k] for k in keys), r["epoch"], *(r["metrics"][m] for m in shown)] for r in rows],
        )
    return "\n".join(L)


def render_report(results: dict, train_splits: list[str], ks: list[int]) -> str:
    """`results` maps model name to {params, val: metrics, test: metrics}, baselines first."""
    L = [
        "# Phase 4 retrieval",
        "",
        "Generated by `python -m m4a_rec.retrieval report`. Do not edit by hand; rerun instead.",
        "",
        f"Every model is fitted on {' + '.join(train_splits)}. The first three rows are the Phase 3",
        "baselines, copied from `reports/baselines.json`.",
        "",
    ]
    L += table(["Model", "Setting"], [[name, json.dumps(r["params"])] for name, r in results.items()])
    return "\n".join(L + metric_tables(results, ks))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cfg = load_config()
    t, ks, reports = cfg["twotower"], cfg["metrics"]["ks"], cfg["paths"]["reports"]
    tunable = list(t["stages"])
    if argv != ["report"] and not (len(argv) == 2 and argv[0] == "tune" and argv[1] in tunable):
        print(__doc__)
        return 2
    processed = cfg["paths"]["processed"]
    inter = build(load_train(processed / "splits", t["train_splits"]), "log")
    features = load_features(processed / "features.npz", inter.track_ids.to_list())

    def train(name: str, params: dict) -> dict:
        print(f"{name} {params}", flush=True)
        on_val = lambda u, v: score(recommend(inter, u, v, t["n_recs"]), "val")  # noqa: E731
        return fit(inter, features, t["models"][name], params, t, on_val, log=lambda m: print(m, flush=True))

    if argv[0] == "tune":
        name, rows = argv[1], []

        def run(params: dict) -> float:
            result = train(name, params)
            rows.append({"params": params, "epoch": result["epoch"], "metrics": result["metrics"]})
            return result["metrics"][t["select_on"]]

        best, _ = staged_search(t["chosen"][name], t["stages"][name], run)
        saved = reports / "twotower_tuning.json"
        runs = json.loads(saved.read_text()) if saved.exists() else {}
        runs[name] = rows
        saved.write_text(json.dumps(runs, indent=2) + "\n")
        (reports / "twotower_tuning.md").write_text(render_tuning(runs, t["select_on"], ks))
        print(f"best {name}: {best}\nwrote {reports / 'twotower_tuning.md'}")
    else:
        results = json.loads((reports / "baselines.json").read_text())
        for name in tunable:
            params = t["chosen"][name]
            result = train(name, params)
            save(processed / "models" / name, inter, result, params)
            recs = recommend(inter, result["user_vecs"], result["track_vecs"], t["n_recs"])
            results[name] = {
                "params": params,
                "epoch": result["epoch"],
                "val": score(recs, "val"),
                "test": score(recs, "test"),
            }
        (reports / "retrieval.json").write_text(stable_json(results))
        (reports / "retrieval.md").write_text(render_report(results, t["train_splits"], ks))
        print(f"wrote {reports / 'retrieval.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
