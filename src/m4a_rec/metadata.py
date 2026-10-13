"""Titles, artists and genres for every track, from the base Music4All dataset, as one table.

Only fixed facts about a track are read: title, artist, album, release year, language and
genre labels. Music4All also ships a popularity score from Spotify, Last.fm tags and its
own file of listens. All three were collected after the training period ends, so nothing
here reads them.

Each track also gets one broad genre group. Its first-listed label decides, and the first
group in configs/metadata.yaml with a word found in that label wins.

Usage: python -m m4a_rec.metadata     # writes data/processed/tracks.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from .config import load_config
from .evaluate import TRAIN
from .interactions import load_train
from .reporting import table


def read(directory: Path, files: dict[str, str]) -> pl.DataFrame:
    """One row per Music4All track: track_id, title, artist, album, year, lang, genres."""
    # everything is read as text: an ID of digits must keep its leading zeros
    load = lambda name: pl.read_csv(directory / files[name], separator="\t", quote_char=None, infer_schema=False)  # noqa: E731
    return (
        load("information")
        .join(load("genres"), on="id")
        .join(load("lang"), on="id")
        .join(load("metadata").select("id", "release"), on="id")
        .select(
            track_id="id",
            title="song",
            artist="artist",
            album="album_name",
            year=pl.col("release").cast(pl.Int32),
            lang="lang",
            genres="genres",
        )
        .sort("track_id")
    )


def genre_group(label: str, groups: list[dict]) -> str:
    """The first group with a word found in `label`, else "other"."""
    return next((g["name"] for g in groups if any(word in label for word in g["words"])), "other")


def with_groups(tracks: pl.DataFrame, groups: list[dict]) -> pl.DataFrame:
    """Adds `genre`, the track's first-listed label in lower case, and its `genre_group`."""
    tracks = tracks.with_columns(genre=pl.col("genres").str.split(",").list.first().str.strip_chars().str.to_lowercase())
    mapping = {label: genre_group(label, groups) for label in tracks["genre"].drop_nulls().unique().to_list()}
    return tracks.with_columns(genre_group=pl.col("genre").replace_strict(mapping, default="other"))


def load(path: Path, track_ids: list[str]) -> pl.DataFrame:
    """Rows of the saved table for `track_ids`, in that order. Fails if a track has no row."""
    rows = pl.DataFrame({"track_id": track_ids}).join(pl.read_parquet(path), on="track_id", how="left", maintain_order="left")
    missing = rows.filter(pl.col("artist").is_null())["track_id"].to_list()
    if missing:
        raise ValueError(f"{len(missing)} tracks have no metadata, e.g. {missing[:3]}")
    return rows


def codes(rows: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """An integer per row for its artist, and one for its album: two albums are the same if artist and name both match."""
    album = pl.concat_str(["artist", "album"], separator="\x1f")
    out = rows.select(artist=pl.col("artist").rank("dense") - 1, album=album.rank("dense") - 1)
    return out["artist"].to_numpy().astype(np.int64), out["album"].to_numpy().astype(np.int64)


def main() -> int:
    cfg = load_config()
    m, processed = cfg["metadata"], cfg["paths"]["processed"]
    tracks = with_groups(read(cfg["paths"]["raw"] / m["dir"], m["files"]), m["genre_groups"])
    dest = processed / "tracks.parquet"
    tracks.write_parquet(dest)
    plays = load_train(processed / "splits", TRAIN).group_by("track_id").len(name="plays")
    known = load(dest, plays["track_id"].to_list()).join(plays, on="track_id")  # fails if a training track is missing
    print(f"wrote {dest}: {tracks.height:,} tracks; all {known.height:,} training tracks covered")
    print(f"{known['artist'].n_unique():,} artists and {known['genre'].n_unique():,} first-listed genre labels in training")
    by_group = known.group_by("genre_group").agg(tracks=pl.len(), plays=pl.col("plays").sum()).sort("plays", descending=True)
    rows = [[g, t, f"{p / known['plays'].sum():.1%}"] for g, t, p in by_group.rows()]
    print("\n".join(table(["Genre group", "Training tracks", "Share of plays"], rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
