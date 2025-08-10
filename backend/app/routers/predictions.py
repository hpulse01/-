from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from slowapi.util import get_remote_address

from ..deps import get_db_dep, get_current_user
from ..models import Prediction, User
from ..schemas import PredictionCreate, PredictionOut, TaskEnqueueResponse
from ..tasks import generate_prediction_task
from ..rate_limit import limiter

router = APIRouter()


@router.post("/predictions", response_model=TaskEnqueueResponse)
@limiter.limit("5/minute")
def create_prediction(payload: PredictionCreate, current: User = Depends(get_current_user)):
    task = generate_prediction_task.delay({"subject_name": payload.subject_name, "seed": payload.seed, "user_email": current.email})
    return TaskEnqueueResponse(task_id=task.id)


@router.get("/predictions", response_model=List[PredictionOut])
@limiter.limit("30/minute")
def list_predictions(current: User = Depends(get_current_user), db: Session = Depends(get_db_dep)):
    items = (
        db.query(Prediction)
        .filter(Prediction.user_id == current.id)
        .order_by(Prediction.id.desc())
        .all()
    )
    return [
        PredictionOut(
            id=p.id,
            subject_name=p.subject_name,
            seed=p.seed,
            result_summary=p.result_summary,
            created_at=str(p.created_at),
        )
        for p in items
    ]