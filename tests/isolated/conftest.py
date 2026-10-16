"""Shared by the tests in this directory, which run in their own process: see tests/test_ranker.py."""
import json

import polars as pl
import pytest
import xgboost as xgb

from m4a_rec import bundle
from m4a_rec.rank_features import FEATURES
from m4a_rec.ranker import save, to_dmatrix
from m4a_rec.serve import load


@pytest.fixture(scope="session")
def service(tmp_path_factory, write_bundle):
    """The synthetic bundle with a small ranker trained on its own candidates."""
    directory = tmp_path_factory.mktemp("serving")
    placeholder = tmp_path_factory.mktemp("ranker")
    (placeholder / "meta.json").write_text(json.dumps({"features": FEATURES, "params": {"n_candidates": 10}, "trees": 5}))
    (placeholder / "model.json").write_text("{}")
    b = write_bundle(directory, placeholder)
    users = b.user_ids
    history = bundle.listens(b, users.to_list())
    tt_user, als_user = bundle.vectors(b, history, users, fold_in=None)
    rows = bundle.candidates(b, history, users, *bundle.retrieve(b, history, users, tt_user), als_user)
    rows = rows.with_columns(label=(pl.int_range(pl.len()).hash(0) % 7 == 0).cast(pl.Int8))  # a seventh of the rows, scattered
    booster = xgb.train({"objective": "rank:ndcg", "max_depth": 2, "seed": 0}, to_dmatrix(rows, FEATURES, label=True), num_boost_round=5)
    save(booster, FEATURES, {"n_candidates": 10}, 5, directory / "ranker")
    return load(directory)
