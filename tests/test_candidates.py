import numpy as np
import polars as pl

from m4a_rec.candidates import snapshot
from m4a_rec.evaluate import PAIR
from m4a_rec.interactions import build
from m4a_rec.rank_features import FEATURES

N = 20


def make(parts, labels):
    """A snapshot of the synthetic retrieval_train split, with random vectors standing in for the models."""
    history = parts["retrieval_train"]
    inter = build(history, "log")
    rng = np.random.default_rng(0)
    n_users, n_tracks = inter.matrix.shape
    vecs = lambda rows, dim: rng.standard_normal((rows, dim)).astype(np.float32)  # noqa: E731
    lyrics = vecs(n_tracks, 5)
    lyrics[::7] = 0  # every seventh track has no lyrics
    blocks = {"audio": vecs(n_tracks, 4), "lyrics": lyrics, "genres": np.abs(vecs(n_tracks, 6))}
    frame = snapshot(
        history, inter, vecs(n_users, 8), vecs(n_tracks, 8), vecs(n_users, 3), vecs(n_tracks, 3), blocks,
        n=N, recent_days=28, short_days=7, labels=labels,
    )
    return history, frame


def test_snapshot_rows_are_unplayed_candidates_in_retrieval_order(parts):
    history, frame = make(parts, parts["ranker_train"])
    assert frame.columns == ["user_id", "track_id", *FEATURES, "label"]
    assert frame.select(PAIR).is_duplicated().sum() == 0
    assert frame.join(history, on=PAIR, how="semi").height == 0  # nothing the user has already played
    assert set(frame["user_id"].to_list()) == set(history["user_id"].to_list())
    per_user = frame.group_by("user_id", maintain_order=True).agg(pl.col("tt_rank"))
    assert all(ranks == list(range(1, len(ranks) + 1)) for ranks in per_user["tt_rank"].to_list())
    assert max(len(ranks) for ranks in per_user["tt_rank"].to_list()) == N
    values = frame.select(FEATURES).to_numpy()
    assert np.isfinite(values[~np.isnan(values)]).all()
    assert np.isnan(frame["sim_lyrics"].to_numpy()).any() and not np.isnan(frame["tt_z"].to_numpy()).any()


def test_label_is_one_exactly_for_candidates_played_in_the_label_period(parts):
    _, frame = make(parts, parts["ranker_train"])
    played = parts["ranker_train"].select(PAIR).unique()
    assert frame["label"].sum() == frame.join(played, on=PAIR, how="semi").height > 0
    assert frame.filter(pl.col("label") == 1).join(played, on=PAIR, how="anti").height == 0
    assert set(frame["label"].unique().to_list()) == {0, 1}


def test_the_label_period_changes_labels_and_nothing_else(parts):
    """The leakage check: features come from the history alone, whatever the label period holds."""
    _, real = make(parts, parts["ranker_train"])
    _, other = make(parts, parts["test"].head(500))  # a different set of later listens
    _, none = make(parts, None)
    assert "label" not in none.columns
    assert real.drop("label").equals(other.drop("label"), null_equal=True)
    assert real.drop("label").equals(none, null_equal=True)
    assert real["label"].to_list() != other["label"].to_list()
