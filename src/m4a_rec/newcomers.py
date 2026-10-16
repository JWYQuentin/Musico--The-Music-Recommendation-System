"""The new-listener test, first half: tt_id and ALS fitted without some listeners.

A seeded share of the users in the two train splits (`newcomers` in configs/serve.yaml) is
held out: every listen of theirs is dropped before tt_id (its chosen setting) and ALS are
fitted, as `retrieval refit` and the infer snapshot fit them on everyone. The second half
serves those listeners from their history alone, as the service would serve someone new:
`python -m m4a_rec.bundle newcomers`, then `python -m m4a_rec.serve report`.

tt_id stops on val as in `retrieval refit`. The held-out listeners get no lists while it
trains, so they score 0 at every epoch and do not change which epoch is kept.

Usage: python -m m4a_rec.newcomers     # ~30 min; writes models/tt_id_newcomers/ and models/als_newcomers/
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

from .baselines import als_vectors, save_als
from .config import load_config
from .evaluate import TRAIN, score
from .features import load as load_features
from .interactions import build, load_train
from .retrieval import fit, recommend, save


def held_out_users(user_ids: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """A seeded random `fraction` of `user_ids`, sorted."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(user_ids, size=round(fraction * len(user_ids)), replace=False))


def without_users(history: pl.DataFrame, users: np.ndarray) -> pl.DataFrame:
    """`history` minus every listen of `users`."""
    return history.filter(~pl.col("user_id").is_in(users.tolist()))


def main() -> int:
    cfg = load_config()
    t, b, n, processed = cfg["twotower"], cfg["baselines"], cfg["newcomers"], cfg["paths"]["processed"]
    history = load_train(processed / "splits", TRAIN)
    held = held_out_users(history["user_id"].unique().sort().to_numpy(), n["fraction"], n["seed"])
    rest = without_users(history, held)
    models = processed / "models"

    inter = build(rest, "log")
    params = t["chosen"]["tt_id"]
    print(f"tt_id {params}, {len(held):,} listeners held out", flush=True)
    on_val = lambda u, v: score(recommend(inter, u, v, t["n_recs"]), "val")  # noqa: E731
    features = load_features(processed / "features.npz", inter.track_ids.to_list())
    result = fit(inter, features, t["models"]["tt_id"], params, t, on_val, log=lambda m: print(m, flush=True))
    save(models / "tt_id_newcomers", inter, result, params)
    np.save(models / "tt_id_newcomers" / "held_out_users.npy", held)

    a = b["als"]
    inter = build(rest, a["weighting"])
    als_user, als_track = als_vectors(inter, iterations=a["iterations"], seed=b["seed"], **a["chosen"])
    settings = {"weighting": a["weighting"], "iterations": a["iterations"], "seed": b["seed"], **a["chosen"]}
    save_als(models / "als_newcomers", inter, als_user, als_track, settings)
    print(f"kept epoch {result['epoch']}; wrote {models / 'tt_id_newcomers'} and {models / 'als_newcomers'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
