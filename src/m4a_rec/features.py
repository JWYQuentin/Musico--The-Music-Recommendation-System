"""Content features per track: audio, lyrics and genres, standardised and saved in one file.

  audio   100 numbers, each column scaled to mean 0 and standard deviation 1
  lyrics  300 numbers, scaled the same way using only the tracks that have lyrics; a track
          whose lyrics vector is all zeros keeps zeros and gets has_lyrics = 0
  genres  685 TF-IDF weights, each row scaled to length 1

Covers every track in the two train splits. The numbers come from the track itself, not
from listens, so nothing here depends on a split boundary.

Usage: python -m m4a_rec.features     # writes data/processed/features.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv

from .config import load_config
from .evaluate import TRAIN
from .interactions import load_train

BLOCKS = ["audio", "lyrics", "has_lyrics", "genres"]  # column order of the matrix `load` returns


def read_matrix(path: Path, track_ids: list[str]) -> np.ndarray:
    """Rows of a feature .tsv.bz2 for `track_ids`, in that order. Fails if a track has no row."""
    table = pacsv.read_csv(
        pa.CompressedInputStream(pa.OSFile(str(path)), "bz2"),
        parse_options=pacsv.ParseOptions(delimiter="\t"),
    )
    table = table.rename_columns(["id", *(f"f{i}" for i in range(table.num_columns - 1))])
    rows = pl.DataFrame({"id": track_ids}).join(
        pl.from_arrow(table), on="id", how="left", maintain_order="left"
    )
    out = rows.drop("id").to_numpy().astype(np.float32)
    if np.isnan(out).any():
        raise ValueError(f"{path.name} has no row for {int(np.isnan(out).any(axis=1).sum())} tracks")
    return out


def zscore(x: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """Scale each column to mean 0 and standard deviation 1, measured on `rows` (default: all)."""
    ref = x if rows is None else x[rows]
    std = ref.std(axis=0)
    return (x - ref.mean(axis=0)) / np.where(std > 0, std, 1)


def transform(audio: np.ndarray, lyrics: np.ndarray, genres: np.ndarray) -> dict[str, np.ndarray]:
    has_lyrics = (lyrics != 0).any(axis=1)
    lyrics = zscore(lyrics, has_lyrics)
    lyrics[~has_lyrics] = 0
    length = np.linalg.norm(genres, axis=1, keepdims=True)
    return {
        "audio": zscore(audio),
        "lyrics": lyrics,
        "has_lyrics": has_lyrics[:, None].astype(np.float32),
        "genres": genres / np.where(length > 0, length, 1),
    }


def load_blocks(path: Path, track_ids: list[str]) -> dict[str, np.ndarray]:
    """The saved features, one float32 matrix per name in BLOCKS, one row per track in `track_ids`."""
    saved = np.load(path)
    row = {t: i for i, t in enumerate(saved["track_ids"].tolist())}
    rows = np.array([row[t] for t in track_ids])
    return {name: saved[name][rows] for name in BLOCKS}


def load(path: Path, track_ids: list[str]) -> np.ndarray:
    """The saved features as one matrix, columns in BLOCKS order."""
    return np.concatenate(list(load_blocks(path, track_ids).values()), axis=1)


def main() -> int:
    cfg = load_config()
    raw, files = cfg["paths"]["raw"], cfg["feature_files"]
    track_ids = load_train(cfg["paths"]["processed"] / "splits", TRAIN)["track_id"].unique().sort().to_list()
    blocks = transform(
        read_matrix(raw / files["audio_ivec256"], track_ids),
        read_matrix(raw / files["lyrics_word2vec"], track_ids),
        read_matrix(raw / files["genres_tfidf"], track_ids),
    )
    dest = cfg["paths"]["processed"] / "features.npz"
    np.savez_compressed(dest, track_ids=np.array(track_ids), **blocks)
    widths = {name: block.shape[1] for name, block in blocks.items()}
    no_lyrics = 1 - blocks["has_lyrics"].mean()
    print(f"wrote {dest}: {len(track_ids):,} tracks, columns {widths}, {no_lyrics:.1%} without lyrics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
