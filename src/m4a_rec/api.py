"""The service over HTTP: recommendations for known listeners and for anyone from their listens, and similar tracks.

Run it, then open http://127.0.0.1:8000/docs to try every call from the browser. It serves
the bundle in data/processed/serving/ (`python -m m4a_rec.bundle`) and listens on
127.0.0.1 only: the responses carry titles and artists from the Music4All database, which
is shared for research only. Do not expose it beyond this machine.

  GET  /health                           what the bundle holds
  GET  /users/{user_id}/recommendations  a listener the models know
  POST /recommendations                  anyone, from their listens: {"plays": [{"track_id", "timestamp"}], "k", "explain"}
  GET  /tracks/{track_id}/similar        the nearest tracks in the two-tower space

Like serve.py it must not import torch, faiss or implicit.

Usage: python -m m4a_rec.api [--host HOST] [--port PORT]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import polars as pl
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import serve
from .config import load_config


class Play(BaseModel):
    track_id: str
    timestamp: datetime | None = Field(None, description="When it was played; left out, the moment the data ends")


class Listens(BaseModel):
    plays: list[Play] = Field(min_length=1)
    k: int = Field(10, ge=1)
    explain: bool = True


class Track(BaseModel):
    track_id: str
    rank: int
    retrieval_rank: int = Field(description="The track's place in the two-tower order, before the ranker")
    title: str | None
    artist: str | None
    reasons: list[str] | None = Field(description="What pushed the track up the most, when asked to explain")


class Recommendations(BaseModel):
    recommendations: list[Track]
    unknown_tracks: list[str] = Field([], description="Track IDs handed in that are not in the catalogue; they were left out")


class SimilarTrack(BaseModel):
    track_id: str
    rank: int
    similarity: float
    title: str | None
    artist: str | None


def naive_utc(t: datetime | None) -> datetime | None:
    """The data's times are UTC without a zone, so a time with a zone is converted and the zone dropped."""
    return t.astimezone(timezone.utc).replace(tzinfo=None) if t is not None and t.tzinfo else t


def create_app(s: serve.Service, max_k: int) -> FastAPI:
    app = FastAPI(title="m4a-rec", summary="Two-stage music recommender on Music4All-Onion. Local only.")

    @app.get("/health")
    def health() -> dict:
        m = s.bundle.meta
        return {"now": m["now"], "listeners": m["listeners"], "tracks": m["tracks"], "models": m["sources"], "fold_in": m["fold_in"]}

    @app.get("/users/{user_id}/recommendations", response_model=Recommendations)
    def for_user(user_id: int, k: int = Query(10, ge=1, le=max_k), explain: bool = True) -> dict:
        try:
            return {"recommendations": serve.for_user(s, user_id, k, explain)}
        except KeyError:
            raise HTTPException(404, f"no listener {user_id} in the bundle; POST their listens to /recommendations") from None

    @app.post("/recommendations", response_model=Recommendations)
    def for_listens(listens: Listens) -> dict:
        if listens.k > max_k:
            raise HTTPException(422, f"k is at most {max_k}")
        plays = pl.DataFrame(
            {"track_id": [p.track_id for p in listens.plays], "timestamp": [naive_utc(p.timestamp) for p in listens.plays]},
            schema={"track_id": pl.String, "timestamp": pl.Datetime("ms")},
        )
        try:
            recs, unknown = serve.for_history(s, plays, listens.k, listens.explain)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return {"recommendations": recs, "unknown_tracks": unknown}

    @app.get("/tracks/{track_id}/similar", response_model=list[SimilarTrack])
    def similar(track_id: str, k: int = Query(10, ge=1, le=max_k)) -> list[dict]:
        try:
            return serve.similar(s, track_id, k)
        except KeyError:
            raise HTTPException(404, f"no track {track_id} in the catalogue") from None

    return app


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    c = cfg["serve"]
    parser = argparse.ArgumentParser(description="Serve recommendations over HTTP.")
    parser.add_argument("--host", default=c["host"], help="0.0.0.0 inside a container only")
    parser.add_argument("--port", type=int, default=c["port"])
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    s = serve.load(cfg["paths"]["processed"] / "serving")
    uvicorn.run(create_app(s, c["max_k"]), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
