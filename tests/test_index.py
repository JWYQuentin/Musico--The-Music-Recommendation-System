"""FAISS runs in a subprocess here: loading it beside PyTorch aborts the process on macOS.

So this file must not import m4a_rec.index (or faiss) at the top level.
"""
import subprocess
import sys

import numpy as np

HAND_CHECK = """
import numpy as np
from m4a_rec.index import agreement, build

tracks = np.array([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]], dtype=np.float32)
user = np.array([[1, 0]], dtype=np.float32)  # scores 1, 0.8, 0, -1
index = build(tracks)
scores, ids = index.search(user, 3)
assert ids.tolist() == [[0, 1, 2]], ids
assert np.allclose(scores, [[1, 0.8, 0]]), scores
assert agreement(index, user, tracks, k=3) == 1.0
assert agreement(build(tracks[::-1].copy()), user, tracks * 2, k=3) == 0.0  # an index of other vectors disagrees
"""


def test_index_returns_the_nearest_tracks_in_order():
    done = subprocess.run([sys.executable, "-c", HAND_CHECK], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_index_command_writes_an_index_that_matches_numpy(tmp_path):
    rng = np.random.default_rng(0)
    np.save(tmp_path / "track_vecs.npy", rng.standard_normal((500, 16), dtype=np.float32))
    np.save(tmp_path / "user_vecs.npy", rng.standard_normal((40, 16), dtype=np.float32))
    done = subprocess.run([sys.executable, "-m", "m4a_rec.index", str(tmp_path)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "tracks.faiss").exists()
    assert "500 tracks; top 100 agree with numpy for 100.0% of 40 users" in done.stdout
