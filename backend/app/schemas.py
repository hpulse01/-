from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PredictionCreate(BaseModel):
    subject_name: str
    seed: str | None = None


class PredictionOut(BaseModel):
    id: int
    subject_name: str
    seed: str | None
    result_summary: str | None
    created_at: str


class TaskEnqueueResponse(BaseModel):
    task_id: str