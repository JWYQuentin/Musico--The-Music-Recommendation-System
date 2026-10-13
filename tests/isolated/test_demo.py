"""Tests for demo.py. It imports ranker.py, so they run in their own process: see tests/test_ranker.py."""
import json
from datetime import datetime

import numpy as np
import polars as pl

from m4a_rec.demo import SLOT, TEMPLATE, USER_FEATURE, embed, page, reason, reasons, recent_tracks


def say(feature, value):
    return reason(feature, value, recent_days=28, short_days=7)


def test_reason_states_the_value_in_words():
    assert say("artist_tracks", np.log1p(12)) == "has played 12 other tracks by this artist"
    assert say("artist_tracks", np.log1p(1)) == "has played 1 other track by this artist"
    assert say("artist_days_since", 3.2) == "last played this artist 3 days ago"
    assert say("artist_days_since", 0.2) == "played this artist within the last day"
    assert say("artist_days_since", float("nan")) == "has never played this artist"
    assert say("track_recent_share", 0.26) == "26% of its listeners played it in the last 28 days"
    assert say("track_listeners_short", np.log1p(1500)) == "1,500 listeners in the last 7 days"
    assert say("tt_rank", 35.0) == "the retrieval model's number 35 candidate for them"
    assert say("pop_gap", -0.2) == "less popular than what they usually play"
    assert say("user_tracks", 5.0) == say("user_mainstream", 0.8) == USER_FEATURE


def test_every_ranker_feature_has_a_statement():
    from m4a_rec.rank_features import FEATURES

    assert all(say(f, 1.0) and say(f, float("nan")) for f in FEATURES)


def test_reasons_are_the_largest_positive_contributions():
    features = ["tt_rank", "artist_tracks", "user_tracks", "user_mainstream", "pop_gap"]
    values = np.array([4.0, np.log1p(2), 5.0, 0.7, 0.3])
    contributions = np.array([0.1, 0.9, 0.5, 0.4, -0.8])
    got = reasons(features, values, contributions, n=3, recent_days=28, short_days=7)
    # largest first; the two user features share one statement; pop_gap pushes the score down, so it is no reason
    assert got == ["has played 2 other tracks by this artist", USER_FEATURE, "the retrieval model's number 4 candidate for them"]
    assert reasons(features, values, -contributions, 3, 28, 7) == ["more popular than what they usually play"]
    assert reasons(features, values, np.full(5, -1.0), 3, 28, 7) == []


def test_recent_tracks_are_distinct_and_latest_first():
    history = pl.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2],
            "track_id": ["a", "b", "a", "c", "a"],
            "timestamp": [datetime(2020, 1, 1), datetime(2020, 1, 5), datetime(2020, 2, 9), datetime(2020, 1, 3), datetime(2020, 1, 2)],
        }
    )
    got = recent_tracks(history, [1, 2], n=2)
    assert got == {1: [("a", "9 Feb 2020"), ("b", "5 Jan 2020")], 2: [("a", "2 Jan 2020")]}  # a counts once, by its last play


def test_embedded_data_cannot_end_the_script_element():
    payload = {"tracks": [['</script><script>alert("x")</script>', "A & B <i>"]]}
    text = embed(payload)
    assert "<" not in text and json.loads(text) == payload  # nothing lost, nothing that reads as a tag
    html = page(payload, f'<script type="application/json">{SLOT}</script>')
    assert html.count("<script") == 1 and html.count("</script>") == 1


def test_the_template_puts_data_on_the_page_as_text_only():
    template = TEMPLATE.read_text()
    assert template.count(SLOT) == 1
    for unsafe in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert unsafe not in template
    assert "http://" not in template and "https://" not in template  # self-contained: nothing is fetched
    assert "IWSSIP 2020" in template  # the dataset's citation
