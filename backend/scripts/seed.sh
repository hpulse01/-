#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import os
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import User
from app.security import hash_password

admin_email = os.environ.get("ADMIN_EMAIL")
admin_password = os.environ.get("ADMIN_PASSWORD")
role = os.environ.get("ADMIN_ROLE", "admin")

if not admin_email or not admin_password:
    raise SystemExit("ADMIN_EMAIL and ADMIN_PASSWORD are required")

with SessionLocal() as db:  # type: Session
    user = db.query(User).filter(User.email == admin_email).first()
    if user:
        print("Admin already exists:", admin_email)
    else:
        user = User(email=admin_email, password_hash=hash_password(admin_password), role=role)
        db.add(user)
        db.commit()
        print("Admin created:", admin_email)
PY