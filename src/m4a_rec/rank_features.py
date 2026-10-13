"""Features for the ranker: one number per (user, candidate track), all from the history handed in.

Nothing here reads a file or knows about splits. Every function takes the listens up to
some moment and describes tracks, users and pairs as of the last of those listens, so the
same code builds the snapshot the ranker is trained on and the one it is used on.

A value that cannot be computed (a share of nothing, a similarity to a track with no
lyrics) is NaN, which XGBoost treats as missing.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl
import scipy.sparse as sp

BATCH = 1000  # users handled at a time in pair_scores

GROUPS = {
    "models": ["tt_rank", "tt_z", "als_rank", "als_z"],
    "track": [
        "track_listeners",
        "track_listeners_recent",
        "track_listeners_short",
        "track_recent_share",
        "track_discovery_rate",
    ],
    "user": ["user_tracks", "user_plays_recent", "user_new_share", "user_mainstream", "user_days_since_last"],
    "pair": ["recent_affinity", "sim_audio", "sim_lyrics", "sim_genres", "pop_gap"],
    "artist": [
        "artist_tracks",
        "artist_play_share",
        "artist_days_since",
        "artist_catalog_share",
        "album_tracks",
        "artist_listeners",
    ],
}
FEATURES = [name for names in GROUPS.values() for name in names]


def unit(x: np.ndarray) -> np.ndarray:
    """Each row scaled to length 1; an all-zero row stays zero."""
    length = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.where(length > 0, length, 1)).astype(np.float32)


def standardise_rows(x: np.ndarray) -> np.ndarray:
    """Each row shifted to mean 0 and scaled to standard deviation 1; a constant row becomes 0."""
    std = x.std(axis=1, keepdims=True)
    return ((x - x.mean(axis=1, keepdims=True)) / np.where(std > 0, std, 1)).astype(np.float32)


def rank_rows(x: np.ndarray) -> np.ndarray:
    """Position of each value within its row, 1 for the largest."""
    order = np.argsort(-x, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, x.shape[1] + 1)[None, :], axis=1)
    return ranks.astype(np.float32)


def pair_scores(user_vecs: np.ndarray, track_vecs: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Dot product of each user's vector with the vector of each of their candidates.

    `idx` is (users, N) track indices, row i belonging to user i. Returns (users, N).
    """
    out = np.empty(idx.shape, dtype=np.float32)
    for start in range(0, idx.shape[0], BATCH):
        scores = user_vecs[start : start + BATCH] @ track_vecs.T
        out[start : start + BATCH] = np.take_along_axis(scores, idx[start : start + BATCH], axis=1)
    return out


def pair_matrix(pairs: pl.DataFrame, user_ids: pl.Series, track_ids: pl.Series, values: str | None = None) -> sp.csr_matrix:
    """A users x tracks matrix with 1, or the `values` column, for every (user_id, track_id) row of `pairs`."""
    rows = pairs.join(user_ids.to_frame().with_row_index("u"), on="user_id").join(
        track_ids.to_frame().with_row_index("t"), on="track_id"
    )
    data = np.ones(rows.height) if values is None else rows[values].to_numpy()
    return sp.csr_matrix(
        (data.astype(np.float32), (rows["u"].to_numpy(), rows["t"].to_numpy())), shape=(len(user_ids), len(track_ids))
    )


def group_matrix(per_track: sp.csr_matrix, codes: np.ndarray) -> sp.csr_matrix:
    """Users x groups: a users x tracks matrix summed over the tracks that share a code (an artist or an album)."""
    n = len(codes)
    member = sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), codes)), shape=(n, int(codes.max()) + 1))
    return (per_track @ member).tocsr()


def lookup(matrix: sp.csr_matrix, cols: np.ndarray) -> np.ndarray:
    """matrix[i, cols[i, j]] for every i and j, as a dense array the shape of `cols`."""
    out = np.empty(cols.shape, dtype=np.float32)
    for start in range(0, cols.shape[0], BATCH):
        dense = matrix[start : start + BATCH].toarray()
        out[start : start + BATCH] = np.take_along_axis(dense, cols[start : start + BATCH], axis=1)
    return out


def mean_direction(played: sp.csr_matrix, track_vecs: np.ndarray) -> np.ndarray:
    """For each user, the direction of the average of their played tracks' unit vectors."""
    return unit(played @ unit(track_vecs))


def first_and_last(history: pl.DataFrame) -> pl.DataFrame:
    """One row per (user, track) with the times of the user's first and last play of it."""
    return history.group_by(["user_id", "track_id"]).agg(
        first=pl.col("timestamp").min(), last=pl.col("timestamp").max()
    )


def track_stats(pairs: pl.DataFrame, track_ids: pl.Series, now, recent_days: int, short_days: int) -> dict:
    """Per-track numbers as of `now`, in `track_ids` order. `pairs` comes from first_and_last."""
    recent, short = now - timedelta(days=recent_days), now - timedelta(days=short_days)
    t = pairs.group_by("track_id").agg(
        listeners=pl.len(),
        listeners_recent=(pl.col("last") > recent).sum(),
        listeners_short=(pl.col("last") > short).sum(),
        new_recent=(pl.col("first") > recent).sum(),
    )
    t = track_ids.to_frame().join(t, on="track_id", how="left", maintain_order="left")
    listeners = t["listeners"].to_numpy().astype(np.float64)
    listeners_recent = t["listeners_recent"].to_numpy().astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        discovery = np.where(listeners_recent > 0, t["new_recent"].to_numpy() / listeners_recent, np.nan)
    return {
        "track_listeners": np.log1p(listeners),
        "track_listeners_recent": np.log1p(listeners_recent),
        "track_listeners_short": np.log1p(t["listeners_short"].to_numpy()),
        "track_recent_share": listeners_recent / listeners,
        "track_discovery_rate": discovery,
        # share of tracks with no more listeners than this one; used for user_mainstream and pop_gap
        "track_pct": t["listeners"].rank("max").to_numpy() / len(track_ids),
    }


def user_stats(
    history: pl.DataFrame, pairs: pl.DataFrame, user_ids: pl.Series, track_pct: pl.DataFrame, now, recent_days: int
) -> dict:
    """Per-user numbers as of `now`, in `user_ids` order. `track_pct` has track_id and pct."""
    recent = now - timedelta(days=recent_days)
    u = pairs.join(track_pct, on="track_id").group_by("user_id").agg(
        tracks=pl.len(),
        tracks_recent=(pl.col("last") > recent).sum(),
        new_recent=(pl.col("first") > recent).sum(),
        mainstream=pl.col("pct").mean(),
        last=pl.col("last").max(),
    )
    plays = history.filter(pl.col("timestamp") > recent).group_by("user_id").len(name="plays_recent")
    u = (
        user_ids.to_frame()
        .join(u, on="user_id", how="left", maintain_order="left")
        .join(plays, on="user_id", how="left", maintain_order="left")
        .with_columns(pl.col("plays_recent").fill_null(0))
    )
    tracks_recent = u["tracks_recent"].to_numpy().astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        new_share = np.where(tracks_recent > 0, u["new_recent"].to_numpy() / tracks_recent, np.nan)
    return {
        "user_tracks": np.log1p(u["tracks"].to_numpy()),
        "user_plays_recent": np.log1p(u["plays_recent"].to_numpy()),
        "user_new_share": new_share,
        "user_mainstream": u["mainstream"].to_numpy(),
        "user_days_since_last": (now - u["last"]).dt.total_seconds().to_numpy() / 86400,
    }


def artist_stats(
    history: pl.DataFrame,
    pairs: pl.DataFrame,
    user_ids: pl.Series,
    track_ids: pl.Series,
    idx: np.ndarray,
    artist: np.ndarray,
    album: np.ndarray,
    now,
) -> dict:
    """What each user has played by each candidate's artist, as of `now`. One (users, N) array per feature.

    `artist` and `album` give an integer code per track, in `track_ids` order.
    """
    played = pair_matrix(pairs, user_ids, track_ids)
    plays = pair_matrix(history.group_by(["user_id", "track_id"]).len(name="plays"), user_ids, track_ids, values="plays")
    by_artist = group_matrix(played, artist)  # tracks of each artist a user has played
    tracks = lookup(by_artist, artist[idx])
    artist_plays = lookup(group_matrix(plays, artist), artist[idx])
    total_plays = np.asarray(plays.sum(axis=1))  # (users, 1); every user in the history has at least one

    # a user's last play of each artist, stored as days ago plus 1 so that 0 can mean "never"
    code = pl.DataFrame({"track_id": track_ids, "a": artist})
    last = pairs.join(code, on="track_id").group_by(["user_id", "a"]).agg(pl.col("last").max())
    last = last.join(user_ids.to_frame().with_row_index("u"), on="user_id")
    days = (now - last["last"]).dt.total_seconds().to_numpy() / 86400
    ago = sp.csr_matrix((days + 1, (last["u"].to_numpy(), last["a"].to_numpy())), shape=by_artist.shape)
    since = lookup(ago, artist[idx])

    return {
        "artist_tracks": np.log1p(tracks),
        "artist_play_share": artist_plays / total_plays,
        "artist_days_since": np.where(since > 0, since - 1, np.nan),
        "artist_catalog_share": tracks / np.bincount(artist)[artist[idx]],
        "album_tracks": np.log1p(lookup(group_matrix(played, album), album[idx])),
        "artist_listeners": np.log1p(by_artist.getnnz(axis=0))[artist[idx]],
    }


def build(
    history: pl.DataFrame,
    user_ids: pl.Series,
    track_ids: pl.Series,
    idx: np.ndarray,
    tt_scores: np.ndarray,
    tt_track: np.ndarray,
    als_user: np.ndarray,
    als_track: np.ndarray,
    blocks: dict[str, np.ndarray],
    artist: np.ndarray,
    album: np.ndarray,
    recent_days: int,
    short_days: int,
) -> dict[str, np.ndarray]:
    """Every feature in FEATURES as a (users, N) array, for candidates `idx` with scores `tt_scores`.

    Row i of `idx` and `tt_scores` is user_ids[i]'s candidates, best first. Vectors,
    `blocks`, `artist` and `album` are in `user_ids` and `track_ids` order. The vectors
    must come from models that saw nothing after `history`; who made a track is a fixed fact.
    """
    now = history["timestamp"].max()
    recent = now - timedelta(days=recent_days)
    pairs = first_and_last(history)
    track = track_stats(pairs, track_ids, now, recent_days, short_days)
    pct = track.pop("track_pct")
    user = user_stats(history, pairs, user_ids, pl.DataFrame({"track_id": track_ids, "pct": pct}), now, recent_days)

    out = {name: values[idx] for name, values in track.items()}
    out |= {name: np.broadcast_to(values[:, None], idx.shape) for name, values in user.items()}

    als = pair_scores(als_user, als_track, idx)
    out |= {
        "tt_rank": np.broadcast_to(np.arange(1, idx.shape[1] + 1)[None, :], idx.shape),
        "tt_z": standardise_rows(tt_scores),
        "als_rank": rank_rows(als),
        "als_z": standardise_rows(als),
        "pop_gap": pct[idx] - user["user_mainstream"][:, None],
    }

    played = pair_matrix(pairs, user_ids, track_ids)
    played_recent = pair_matrix(pairs.filter(pl.col("last") > recent), user_ids, track_ids)
    affinity = pair_scores(mean_direction(played_recent, tt_track), unit(tt_track), idx)
    out["recent_affinity"] = np.where(played_recent.getnnz(axis=1)[:, None] > 0, affinity, np.nan)
    for name in ("audio", "lyrics", "genres"):
        block = blocks[name]
        sim = pair_scores(mean_direction(played, block), unit(block), idx)
        has = (block != 0).any(axis=1)  # a track with an all-zero vector has no direction to compare
        out[f"sim_{name}"] = np.where(has[idx], sim, np.nan)
    out |= artist_stats(history, pairs, user_ids, track_ids, idx, artist, album, now)
    return {name: np.asarray(out[name], dtype=np.float32) for name in FEATURES}
