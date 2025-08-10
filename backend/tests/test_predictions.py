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


def test_enqueue_prediction(monkeypatch):
    # register and login
    client.post("/api/v1/auth/register", json={"email": "u2@example.com", "password": "Password1!"})
    login = client.post("/api/v1/auth/login", json={"email": "u2@example.com", "password": "Password1!"}).json()
    token = login["access_token"]

    class Task:
        id = "fake-id"
    def fake_delay(payload):
        return Task()
    from app import tasks as tasks_mod
    monkeypatch.setattr(tasks_mod.generate_prediction_task, "delay", fake_delay)

    r = client.post("/api/v1/predictions", json={"subject_name": "Alice"}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["task_id"] == "fake-id"