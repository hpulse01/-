from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import get_db
from app.models import Base

engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


def test_register_and_login():
    r = client.post("/api/v1/auth/register", json={"email": "u@example.com", "password": "Password1!"})
    assert r.status_code == 200

    r2 = client.post("/api/v1/auth/login", json={"email": "u@example.com", "password": "Password1!"})
    assert r2.status_code == 200
    assert "access_token" in r2.json()