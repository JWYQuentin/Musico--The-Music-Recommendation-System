import numpy as np
import polars as pl

from m4a_rec.newcomers import held_out_users, without_users


def test_held_out_users_are_a_seeded_share():
    users = np.arange(100, 300)
    held = held_out_users(users, 0.1, seed=1)
    assert len(held) == 20 and set(held.tolist()) <= set(users.tolist())
    assert np.all(np.diff(held) > 0)  # sorted, no repeats
    assert np.array_equal(held, held_out_users(users, 0.1, seed=1))
    assert not np.array_equal(held, held_out_users(users, 0.1, seed=2))


def test_without_users_drops_every_listen_of_theirs_and_nothing_else(parts):
    history = pl.concat([parts["retrieval_train"], parts["ranker_train"]])
    held = held_out_users(history["user_id"].unique().sort().to_numpy(), 0.2, seed=0)
    rest = without_users(history, held)
    assert not rest["user_id"].is_in(held.tolist()).any()
    assert rest.height + history.filter(pl.col("user_id").is_in(held.tolist())).height == history.height
