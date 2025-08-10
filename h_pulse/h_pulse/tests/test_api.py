from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from h_pulse.output_generation.api import app

client = TestClient(app)


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_and_anchor():
    payload = {"name": "Alice", "birth_time": "1990-01-01T08:00:00+00:00", "lat": 35.0, "lon": 139.0}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "trajectory" in data and "anchor" in data
    anchor = data["anchor"]
    vr = client.post("/anchor/verify", json={"digest": anchor["digest"], "signature": anchor["signature"], "public_key": anchor["public_key"]})
    assert vr.status_code == 200
    assert vr.json()["ok"] is True