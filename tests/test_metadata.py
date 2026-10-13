import polars as pl
import pytest

from m4a_rec.metadata import codes, genre_group, load, read, with_groups

FILES = {"information": "info.csv", "genres": "genres.csv", "lang": "lang.csv", "metadata": "meta.csv"}
GROUPS = [
    {"name": "metal", "words": ["metal"]},
    {"name": "punk and hardcore", "words": ["punk", "core"]},
    {"name": "rock", "words": ["rock", "post-"]},
    {"name": "pop", "words": ["pop"]},
]


def write(path, header, rows):
    path.write_text("\n".join("\t".join(str(v) for v in row) for row in [header, *rows]) + "\n")


def write_dataset(directory):
    write(directory / "info.csv", ["id", "artist", "song", "album_name"], [
        ("0012", "Band/One", 'Say "Hi"', "Album, The"),  # an ID of digits, a title with quotes, a comma in the album
        ("b", "Zoë Two", "It's a Test", "Second"),
    ])
    write(directory / "genres.csv", ["id", "genres"], [("0012", "Hard Rock,pop"), ("b", "art pop,electronic")])
    write(directory / "lang.csv", ["id", "lang"], [("0012", "en"), ("b", "en")])
    write(directory / "meta.csv", ["id", "spotify_id", "popularity", "release"], [("0012", "x", "55.0", "1980"), ("b", "y", "60.0", "1995")])


def test_read_keeps_text_as_it_is(tmp_path):
    write_dataset(tmp_path)
    tracks = read(tmp_path, FILES)
    assert tracks.columns == ["track_id", "title", "artist", "album", "year", "lang", "genres"]  # no popularity
    assert tracks.rows() == [
        ("0012", 'Say "Hi"', "Band/One", "Album, The", 1980, "en", "Hard Rock,pop"),
        ("b", "It's a Test", "Zoë Two", "Second", 1995, "en", "art pop,electronic"),
    ]


def test_genre_group_takes_the_first_group_that_matches():
    assert genre_group("alternative metal", GROUPS) == "metal"
    assert genre_group("metalcore", GROUPS) == "metal"  # metal comes before the group with "core"
    assert genre_group("pop punk", GROUPS) == "punk and hardcore"  # punk comes before pop
    assert genre_group("post-hardcore", GROUPS) == "punk and hardcore"
    assert genre_group("post-rock", GROUPS) == "rock"
    assert genre_group("indie pop", GROUPS) == "pop"
    assert genre_group("zydeco", GROUPS) == "other"


def test_the_first_listed_label_decides_the_group(tmp_path):
    write_dataset(tmp_path)
    tracks = with_groups(read(tmp_path, FILES), GROUPS)
    assert tracks["genre"].to_list() == ["hard rock", "art pop"]
    assert tracks["genre_group"].to_list() == ["rock", "pop"]  # not pop for the first, although it is also listed


def test_load_follows_the_requested_order_and_fails_on_a_missing_track(tmp_path):
    write_dataset(tmp_path)
    with_groups(read(tmp_path, FILES), GROUPS).write_parquet(tmp_path / "tracks.parquet")
    assert load(tmp_path / "tracks.parquet", ["b", "0012"])["artist"].to_list() == ["Zoë Two", "Band/One"]
    with pytest.raises(ValueError):
        load(tmp_path / "tracks.parquet", ["b", "zzz"])


def test_codes_tell_apart_albums_with_the_same_name():
    rows = pl.DataFrame({"artist": ["B", "A", "B", "A"], "album": ["Greatest Hits", "Greatest Hits", "Greatest Hits", "Debut"]})
    artist, album = codes(rows)
    assert artist.tolist() == [1, 0, 1, 0]  # artists in alphabetical order
    assert album[0] == album[2] and len({album[0], album[1], album[3]}) == 3
