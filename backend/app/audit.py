from __future__ import annotations

from sqlalchemy.orm import Session

from .models import AuditLog


def write_audit_log(db: Session, user_id: int | None, action: str, details: str | None = None) -> None:
    log = AuditLog(user_id=user_id, action=action, details=details)
    db.add(log)
    db.commit()