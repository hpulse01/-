from __future__ import annotations

import hashlib
from typing import Dict

from sqlalchemy.orm import Session

from .celery_app import celery_app
from .database import SessionLocal
from .models import User, Prediction


@celery_app.task(name="generate_prediction_task")
def generate_prediction_task(payload: Dict[str, str | None]) -> dict:
    subject_name = str(payload.get("subject_name", ""))
    seed = str(payload.get("seed", "") or "")
    user_email = str(payload.get("user_email", ""))

    # Deterministic pseudo prediction using SHA256
    h = hashlib.sha256(f"{subject_name}|{seed}".encode()).hexdigest()
    score = int(h[:8], 16) % 100
    life_path = int(h[8:12], 16) % 9 + 1
    summary = f"Trajectory score={score}, life_path={life_path}. Hash={h[:10]}..."

    with SessionLocal() as db:  # type: Session
        user = db.query(User).filter(User.email == user_email).first()
        if not user:
            return {"status": "error", "message": "user not found"}
        pred = Prediction(user_id=user.id, subject_name=subject_name, seed=seed, result_summary=summary)
        db.add(pred)
        db.commit()
        db.refresh(pred)
        return {"status": "ok", "prediction_id": pred.id}