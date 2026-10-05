import numpy as np
import polars as pl
import pytest

from m4a_rec.interactions import build, load_train, to_frame


def tiny():
    # user 1 plays a twice and b once; user 2 plays a and c; user 3 plays a three times
    return pl.DataFrame(
        {
            "user_id": [2, 1, 1, 3, 1, 2, 3, 3],
            "track_id": ["c", "a", "b", "a", "a", "a", "a", "a"],
        }
    )


def test_build_hand_checked():
    inter = build(tiny(), "binary")
    assert inter.user_ids.to_list() == [1, 2, 3]
    assert inter.track_ids.to_list() == ["a", "b", "c"]
    assert inter.matrix.toarray().tolist() == [[1, 1, 0], [1, 0, 1], [1, 0, 0]]

    log = build(tiny(), "log").matrix.toarray()
    assert log == pytest.approx(np.log([[3, 2, 1], [2, 1, 2], [4, 1, 1]]))  # log(1 + plays)
    assert log.dtype == np.float32


def test_to_frame_drops_padding_and_played_tracks():
    inter = build(tiny(), "binary")
    item_idx = np.array([[2, 0], [1, 2], [1, -1]])  # user 1: c, a; user 2: b, c; user 3: b, padding
    scores = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, -3.4e38]])
    recs = to_frame(inter, item_idx, scores)
    # user 1 has played a and user 2 has played c
    assert recs.rows() == [(1, "c", pytest.approx(0.9)), (2, "b", pytest.approx(0.7)), (3, "b", 0.5)]
    assert recs.schema == {"user_id": pl.Int64, "track_id": pl.String, "score": pl.Float32}


def test_load_train_reads_only_train_splits(tmp_path):
    for name in ("retrieval_train", "ranker_train"):
        tiny().write_parquet(tmp_path / f"{name}.parquet")
    assert load_train(tmp_path, ["retrieval_train"]).height == 8
    assert load_train(tmp_path, ["retrieval_train", "ranker_train"]).height == 16
    tiny().write_parquet(tmp_path / "val.parquet")
    with pytest.raises(ValueError):
        load_train(tmp_path, ["retrieval_train", "val"])
