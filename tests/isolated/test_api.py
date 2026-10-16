"""Tests for api.py. It imports ranker.py through serve.py, so they run in their own process: see tests/test_ranker.py."""
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from m4a_rec import bundle
from m4a_rec.api import create_app

MAX_K = 20


@pytest.fixture(scope="module")
def client(service):
    return TestClient(create_app(service, MAX_K))


def listens_of(service, user):
    rows = bundle.listens(service.bundle, [user]).select("track_id", "timestamp").rows()
    return [{"track_id": track, "timestamp": when.isoformat()} for track, when in rows]


def test_health_says_what_the_bundle_holds(client, service):
    got = client.get("/health").json()
    assert got["listeners"] == len(service.bundle.user_ids) and got["tracks"] == len(service.bundle.track_ids)


def test_a_known_listener(client, service):
    user = int(service.bundle.user_ids[4])
    got = client.get(f"/users/{user}/recommendations", params={"k": 5})
    assert got.status_code == 200
    recs = got.json()["recommendations"]
    assert [r["rank"] for r in recs] == [1, 2, 3, 4, 5] and all(r["title"] and r["reasons"] is not None for r in recs)
    plain = client.get(f"/users/{user}/recommendations", params={"k": 5, "explain": False}).json()["recommendations"]
    assert [r["track_id"] for r in plain] == [r["track_id"] for r in recs] and all(r["reasons"] is None for r in plain)


def test_a_listener_the_models_do_not_know_is_404(client):
    assert client.get("/users/-5/recommendations").status_code == 404


def test_k_is_checked(client, service):
    user = int(service.bundle.user_ids[4])
    assert client.get(f"/users/{user}/recommendations", params={"k": MAX_K + 1}).status_code == 422
    assert client.get(f"/users/{user}/recommendations", params={"k": 0}).status_code == 422
    body = {"plays": listens_of(service, user), "k": MAX_K + 1}
    assert client.post("/recommendations", json=body).status_code == 422


def test_someone_new_from_their_listens(client, service):
    plays = listens_of(service, int(service.bundle.user_ids[4]))
    got = client.post("/recommendations", json={"plays": plays + [{"track_id": "nope"}], "k": 5})
    assert got.status_code == 200
    assert len(got.json()["recommendations"]) == 5 and got.json()["unknown_tracks"] == ["nope"]
    # a time with a zone is converted to UTC: the same moments written two hours ahead, +02:00, give the same list
    ahead = lambda when: (datetime.fromisoformat(when) + timedelta(hours=2)).isoformat() + "+02:00"  # noqa: E731
    shifted = [{"track_id": p["track_id"], "timestamp": ahead(p["timestamp"])} for p in plays]
    assert client.post("/recommendations", json={"plays": shifted, "k": 5}).json() == client.post(
        "/recommendations", json={"plays": plays, "k": 5}
    ).json()


def test_bad_listens_are_422(client, service):
    later = (service.bundle.meta["now"] + timedelta(days=1)).isoformat()
    assert client.post("/recommendations", json={"plays": [{"track_id": service.bundle.track_ids[0], "timestamp": later}]}).status_code == 422
    assert client.post("/recommendations", json={"plays": [{"track_id": "nope"}]}).status_code == 422
    assert client.post("/recommendations", json={"plays": []}).status_code == 422


def test_similar_tracks(client, service):
    track = service.bundle.track_ids[0]
    got = client.get(f"/tracks/{track}/similar", params={"k": 3})
    assert got.status_code == 200 and [r["rank"] for r in got.json()] == [1, 2, 3]
    assert track not in [r["track_id"] for r in got.json()]
    assert client.get("/tracks/nope/similar").status_code == 404
