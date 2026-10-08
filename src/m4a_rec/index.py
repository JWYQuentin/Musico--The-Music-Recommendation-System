"""A FAISS index over a model's track vectors, for looking up a user's nearest tracks.

Exact inner-product search: with 56,000 tracks there is no need for an approximate index.

FAISS and PyTorch cannot be loaded into the same process on macOS: each ships its own
OpenMP runtime and the process aborts. So this module imports neither torch nor anything
that does, it runs as its own command, and its tests run it in a subprocess.

Usage: python -m m4a_rec.index data/processed/models/<model>
       reads track_vecs.npy and user_vecs.npy there, writes tracks.faiss, and checks the
       index against plain numpy on the first users.
"""
from __future__ import annotations

import sys
from pathlib import Path

import faiss
import numpy as np

CHECK_USERS, CHECK_K = 1000, 100  # how much of the index the check compares with numpy


def build(track_vecs: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(track_vecs.shape[1])
    index.add(np.ascontiguousarray(track_vecs, dtype=np.float32))
    return index


def agreement(index: faiss.IndexFlatIP, user_vecs: np.ndarray, track_vecs: np.ndarray, k: int) -> float:
    """Share of users whose k best scores from the index equal the k best from numpy."""
    k = min(k, track_vecs.shape[0])
    found, _ = index.search(np.ascontiguousarray(user_vecs, dtype=np.float32), k)
    exact = -np.sort(-(user_vecs @ track_vecs.T), axis=1)[:, :k]
    return float(np.isclose(found, exact, atol=1e-5).all(axis=1).mean())


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    directory = Path(argv[0])
    track_vecs = np.load(directory / "track_vecs.npy")
    user_vecs = np.load(directory / "user_vecs.npy")[:CHECK_USERS]
    index = build(track_vecs)
    faiss.write_index(index, str(directory / "tracks.faiss"))
    share = agreement(index, user_vecs, track_vecs, CHECK_K)
    print(f"wrote {directory / 'tracks.faiss'}: {index.ntotal:,} tracks; top {CHECK_K} agree with numpy for {share:.1%} of {len(user_vecs):,} users")
    return 0 if share == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
