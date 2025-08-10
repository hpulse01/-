from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..deps import get_db_dep
from ..models import User
from ..schemas import UserCreate, TokenResponse, UserOut, LoginRequest
from ..security import hash_password, verify_password, create_access_token
from ..audit import write_audit_log

router = APIRouter()


@router.post("/auth/register", response_model=UserOut)
def register(payload: UserCreate, db: Session = Depends(get_db_dep)):
    exists = db.query(User).filter(User.email == payload.email).first()
    if exists:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email exists")
    user = User(email=payload.email, password_hash=hash_password(payload.password), role="user")
    db.add(user)
    db.commit()
    db.refresh(user)
    write_audit_log(db, user_id=user.id, action="register", details=f"email={user.email}")
    return UserOut(id=user.id, email=user.email, role=user.role)


@router.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db_dep)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(subject=user.email, extra_claims={"role": user.role})
    write_audit_log(db, user_id=user.id, action="login", details=f"email={user.email}")
    return TokenResponse(access_token=token, token_type="bearer")