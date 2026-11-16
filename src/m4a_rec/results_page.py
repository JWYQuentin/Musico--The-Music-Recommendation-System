"""Build the public results page: one self-contained HTML file made from totals only.

The page draws the project's results (model comparison with intervals, what the ranker
relies on, new listeners, tracks nobody has played, who the system serves, serving). It
reads only the report files committed under reports/, which hold aggregates, and keeps
only the fields it draws. Nothing per listener or per track reaches it, so it can be
hosted: it is written to docs/index.html, which GitHub Pages serves.

It imports nothing that reads the data, so it runs without the data or the models.

Usage: python -m m4a_rec.results_page     # writes docs/index.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import load_config

TEMPLATE = Path(__file__).with_name("results_page.html")
SLOT = "/*DATA*/"  # where the page's data goes in the template
SOURCES = ["ranker", "listens_curve", "questionnaire", "coldstart", "bias", "bias_tuning", "serving"]

MODELS = {
    "popularity": "Popularity",
    "item_knn": "Item-kNN",
    "als": "ALS",
    "tt_id": "Two-tower",
    "tt_id + ranker": "Two-tower + ranker",
    "tt_id + ranker_artist": "Two-tower + ranker with artist features",
    "tt_id + ranker_artist with penalty": "The same with the popularity penalty",
}
KS = [10, 20, 100, 500]
NEW_METRICS = ["ndcg@10", "recall@10", "recall@100", "recall@500"]  # what the new-listener reports hold
COLD_METRICS = ["ndcg@10", "recall@10", "recall@100"]
COLD_MODELS = {
    "tt_content": "Two-tower from audio, lyrics and genres",
    "content_sim": "Closest in audio, lyrics and genres",
    "random": "Random",
}
# Ranker features: (group, plain words). The groups are rank_features.GROUPS.
FEATURES = {
    "tt_rank": ("models", "Two-tower rank among the listener's candidates"),
    "tt_z": ("models", "Two-tower score against the listener's other candidates"),
    "als_rank": ("models", "ALS rank among the listener's candidates"),
    "als_z": ("models", "ALS score against the listener's other candidates"),
    "track_listeners": ("track", "Track's listeners in all"),
    "track_listeners_recent": ("track", "Track's listeners in recent weeks"),
    "track_listeners_short": ("track", "Track's listeners in the last days"),
    "track_recent_share": ("track", "Share of the track's listeners who are recent"),
    "track_discovery_rate": ("track", "Share of recent listeners hearing it for the first time"),
    "user_tracks": ("listener", "Listener's tracks played"),
    "user_plays_recent": ("listener", "Listener's recent plays"),
    "user_new_share": ("listener", "Share of the listener's plays that are new tracks"),
    "user_mainstream": ("listener", "How mainstream the listener is"),
    "user_days_since_last": ("listener", "Days since the listener last played"),
    "recent_affinity": ("listener and track", "Closeness to what the listener played recently"),
    "sim_audio": ("listener and track", "Sounds like what the listener plays"),
    "sim_lyrics": ("listener and track", "Lyrics close to what the listener plays"),
    "sim_genres": ("listener and track", "Genres match what the listener plays"),
    "pop_gap": ("listener and track", "More or less popular than the listener's usual"),
    "artist_tracks": ("artist", "Listener's tracks by this artist"),
    "artist_play_share": ("artist", "Share of the listener's plays by this artist"),
    "artist_days_since": ("artist", "Days since the listener last played this artist"),
    "artist_catalog_share": ("artist", "Share of the artist's tracks the listener has played"),
    "album_tracks": ("artist", "Listener's tracks from this album"),
    "artist_listeners": ("artist", "Artist's listeners"),
}


def pick(d: dict, keys: list[str]) -> dict:
    """Only `keys` of `d`, in that order; a key `d` lacks is left out."""
    return {k: d[k] for k in keys if k in d}


def splits(d: dict, keys: list[str]) -> dict:
    """{split: only `keys` of its metrics} for val and test."""
    return {s: pick(d[s], keys) for s in ("val", "test")}


def model_data(ranker: dict) -> dict:
    metrics = [f"{m}@{k}" for k in KS for m in ("ndcg", "recall")]
    one = next(iter(ranker["results"].values()))
    return {
        "users": {s: one[s]["users"] for s in ("val", "test")},
        "ks": KS,
        "models": [
            {"key": key, "label": MODELS[key], **splits(r, metrics)} for key, r in ranker["results"].items()
        ],
        "gaps": [
            {"model": MODELS[g["model"]], "against": MODELS[g["against"]], **splits(g, metrics)}
            for g in ranker["gaps"]
        ],
        "shap": [
            {"feature": f, "group": FEATURES[f][0], "label": FEATURES[f][1], "value": v}
            for f, v in sorted(ranker["shap"].items(), key=lambda kv: -kv[1])
        ],
    }


def newcomer_data(curve: dict, questionnaire: dict, serving: dict) -> dict:
    def stages(row: dict) -> dict:
        return {stage: splits(row[stage], NEW_METRICS) for stage in ("retrieval", "both") if stage in row}

    return {
        "users": pick(curve["users"], ["held_out", "val", "test"]),
        "popularity": splits(curve["popularity"], NEW_METRICS),
        "learned": splits(serving["newcomers"]["rows"]["learned vectors"], NEW_METRICS),
        "curve": [{"tracks": n, **stages(row)} for n, row in curve["curve"].items()],
        "forms": [
            {"form": name, "questions": row["questions"], **stages(row)} for name, row in questionnaire["forms"].items()
        ],
    }


def coldstart_data(cold: dict) -> dict:
    return {
        "cold": [{"label": COLD_MODELS[m], **splits(r, ["users", *COLD_METRICS])} for m, r in cold["cold"].items()],
        "all": splits(cold["all"]["tt_content"], ["users", *COLD_METRICS]),
    }


def bias_data(bias: dict, tuning: dict) -> dict:
    def shares(found: dict) -> dict:
        return {group: row["share"] for group, row in found.items() if group != "no group"}

    first = next(iter(bias["results"].values()))
    return {
        "split": bias["split"],
        "k": bias["k"],
        "listening": {
            "popularity": shares(first["found"]["popularity"]),
            "genre": shares(first["found"]["genre"]),
        },
        "models": [
            {
                "label": MODELS[m],
                **pick(r, ["ndcg", "coverage", "gini"]),
                "slots": pick(r["slots"], ["popularity", "genre"]),
                "listeners": r["users"]["listeners"],
                "listener_genres": r["users"]["genre"],
            }
            for m, r in bias["results"].items()
        ],
        "penalty": {
            "chosen": tuning["chosen"],
            "rows": [pick(r, ["lambda", "ndcg", "head", "middle", "tail", "coverage"]) for r in tuning["rows"]],
        },
    }


def serving_data(serving: dict) -> dict:
    p = serving["parity"]
    lat = serving["latency"]
    return {
        "users": p["users"],
        "same_top": p["same_top"],
        "requests": lat["requests"],
        "latency": {who: {stage: pick(t, ["p50", "p95"]) for stage, t in lat[who].items()} for who in ("known", "new")},
    }


def page_data(reports: dict[str, dict]) -> dict:
    """The fields the page draws, from the reports named in SOURCES, and nothing else."""
    return {
        "results": model_data(reports["ranker"]),
        "newcomers": newcomer_data(reports["listens_curve"], reports["questionnaire"], reports["serving"]),
        "coldstart": coldstart_data(reports["coldstart"]),
        "bias": bias_data(reports["bias"], reports["bias_tuning"]),
        "serving": serving_data(reports["serving"]),
    }


def embed(payload: dict) -> str:
    """The data as JSON that is safe inside a <script> element: no character that could end it early."""
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def build(reports: dict[str, dict], template: str) -> str:
    if template.count(SLOT) != 1:
        raise ValueError(f"the template must hold {SLOT} exactly once")
    return template.replace(SLOT, embed(page_data(reports)))


def main() -> int:
    cfg = load_config()
    reports = {name: json.loads((cfg["paths"]["reports"] / f"{name}.json").read_text()) for name in SOURCES}
    dest = cfg["paths"]["docs"] / "index.html"
    dest.write_text(build(reports, TEMPLATE.read_text()))
    print(f"wrote {dest}: {dest.stat().st_size / 1024:.0f} KB from {len(SOURCES)} reports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
