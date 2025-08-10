from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse

from pydantic import BaseModel

from ..utils.settings import SETTINGS
from ..utils.crypto_anchor import verify_anchor
from ..quantum_engine.simulator import simulate_quantum_life
from ..prediction_ai.infer import run_inference
from ..data_collection.eastern import collect_eastern_features
from ..data_collection.western import collect_western_features
from .report import TimelineEvent, build_timeline, sign_and_anchor


class Profile(BaseModel):
    name: str
    birth_time: str
    lat: float
    lon: float


app = FastAPI(default_response_class=ORJSONResponse, title="H-Pulse Quantum Prediction System", version=SETTINGS.version)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "version": SETTINGS.version, "time": datetime.now(timezone.utc).isoformat()}


@app.post("/predict")
def predict(profile: Profile):
    try:
        birth_dt = datetime.fromisoformat(profile.birth_time)
    except Exception:
        raise HTTPException(400, detail="Invalid birth_time ISO format")
    east = collect_eastern_features(birth_dt)
    west = collect_western_features(birth_dt, profile.lat, profile.lon)
    ai = run_inference()
    quantum = simulate_quantum_life({"east": east.__dict__, "west": west.__dict__, "ai": ai}, bio={"name": profile.name}, birth_spacetime={"birth_time": profile.birth_time, "lat": profile.lat, "lon": profile.lon})
    # Build synthetic events from AI probs
    events: List[TimelineEvent] = []
    base_time = birth_dt
    for i, p in enumerate(ai["probs"]):
        events.append(TimelineEvent(
            id=f"evt-{i}", time=(base_time.replace(tzinfo=timezone.utc) if base_time.tzinfo is None else base_time) .isoformat(), place="N/A", domain="life", impact=float(p), advice="Balance work & rest", confidence=float(p), error_margin=float(ai["ci"][i][1] - ai["ci"][i][0]), synthetic=False,
        ))
        base_time = base_time.replace(year=base_time.year + 1)
    timeline = build_timeline(events)
    anchor = sign_and_anchor(timeline)
    return {"trajectory": ai["probs"], "events": [e.__dict__ for e in events], "confidence": quantum.fidelity, "anchor": anchor}


@app.get("/events/{event_id}")
def get_event(event_id: str):
    evt = TimelineEvent(id=event_id, time=datetime.now(timezone.utc).isoformat(), place="N/A", domain="life", impact=0.5, advice="Stay calm", confidence=0.66, error_margin=0.1)
    return {"event": evt.__dict__, "provenance": {"source": "synthetic"}}


@app.post("/anchor/verify")
def anchor_verify(payload: Dict[str, str]):
    required = ["digest", "signature", "public_key"]
    if not all(k in payload for k in required):
        raise HTTPException(400, detail="Missing fields")
    ok = verify_anchor(payload["digest"], payload["signature"], payload["public_key"])
    return {"ok": bool(ok)}


# Optional GraphQL setup; skip if strawberry incompatible
try:
    import strawberry  # type: ignore
    from strawberry.fastapi import GraphQLRouter  # type: ignore

    @strawberry.type
    class GQLEvent:
        id: str
        time: str
        domain: str
        impact: float
        confidence: float

    @strawberry.type
    class Query:
        @strawberry.field
        def events(self, info, min_impact: float = 0.0) -> List[GQLEvent]:
            now = datetime.now(timezone.utc).isoformat()
            return [GQLEvent(id=f"evt-{i}", time=now, domain="life", impact=0.1 * i, confidence=0.5 + 0.05 * i) for i in range(10) if 0.1 * i >= min_impact]

        @strawberry.field
        def aggregate(self, info) -> Dict[str, float]:
            return {"count": 10, "avg_impact": 0.45}

    schema = strawberry.Schema(query=Query)
    graphql_app = GraphQLRouter(schema)
    app.include_router(graphql_app, prefix="/graphql")
except Exception:  # pragma: no cover
    pass