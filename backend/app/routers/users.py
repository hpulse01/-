from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..deps import get_db_dep, get_current_user, require_role
from ..models import User
from ..schemas import UserOut

router = APIRouter()


@router.get("/users/me", response_model=UserOut)
def me(current: User = Depends(get_current_user)):
    return UserOut(id=current.id, email=current.email, role=current.role)


@router.get("/users", response_model=List[UserOut])
def list_users(_: User = Depends(require_role("admin")), db: Session = Depends(get_db_dep)):
    users = db.query(User).order_by(User.id.asc()).all()
    return [UserOut(id=u.id, email=u.email, role=u.role) for u in users]