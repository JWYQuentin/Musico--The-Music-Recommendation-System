import bz2

import numpy as np
import pytest

from m4a_rec.features import BLOCKS, load, read_matrix, transform, zscore


def write(path, header, rows):
    with bz2.open(path, "wt") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(v) for v in row) + "\n")


def test_read_matrix_follows_the_requested_order(tmp_path):
    write(tmp_path / "f.tsv.bz2", ["id", "x", "hip hop"], [("a", 1, 2), ("b", 3, 4), ("c", 5, 6)])
    assert read_matrix(tmp_path / "f.tsv.bz2", ["c", "a"]).tolist() == [[5, 6], [1, 2]]
    with pytest.raises(ValueError):
        read_matrix(tmp_path / "f.tsv.bz2", ["a", "zzz"])


def test_zscore_hand_checked():
    x = np.array([[1.0, 5.0], [3.0, 5.0]])  # column 0: mean 2, sd 1; column 1 is constant
    assert zscore(x).tolist() == [[-1, 0], [1, 0]]
    # measured on rows 0 and 1 only (mean 2, sd 1), then applied to every row
    y = np.array([[1.0], [3.0], [6.0]])
    assert zscore(y, np.array([True, True, False])).ravel().tolist() == [-1, 1, 4]


def test_transform_hand_checked():
    audio = np.array([[0.0], [2.0], [4.0]])  # mean 2, sd sqrt(8/3)
    lyrics = np.array([[1.0, 0.0], [0.0, 0.0], [3.0, 0.0]])  # track 1 has no lyrics
    genres = np.array([[3.0, 4.0], [0.0, 0.0], [0.0, 2.0]])
    out = transform(audio, lyrics, genres)
    assert out["audio"].ravel() == pytest.approx(np.array([-2, 0, 2]) / np.sqrt(8 / 3))
    assert out["has_lyrics"].ravel().tolist() == [1, 0, 1]
    # scaled on tracks 0 and 2 (mean 2, sd 1); track 1 stays at zero, not at (0 - 2) / 1
    assert out["lyrics"].tolist() == [[-1, 0], [0, 0], [1, 0]]
    assert out["genres"].tolist() == [[0.6, 0.8], [0, 0], [0, 1]]  # each row has length 1, or stays zero


def test_load_lines_rows_up_with_track_ids(tmp_path):
    blocks = {
        "audio": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        "lyrics": np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=np.float32),
        "has_lyrics": np.array([[1.0], [0.0], [1.0]], dtype=np.float32),
        "genres": np.array([[0.1], [0.2], [0.3]], dtype=np.float32),
    }
    np.savez_compressed(tmp_path / "features.npz", track_ids=np.array(["a", "b", "c"]), **blocks)
    got = load(tmp_path / "features.npz", ["c", "a"])
    assert BLOCKS == ["audio", "lyrics", "has_lyrics", "genres"]
    assert got == pytest.approx(np.array([[3, 30, 31, 1, 0.3], [1, 10, 11, 1, 0.1]]))
    assert got.dtype == np.float32
