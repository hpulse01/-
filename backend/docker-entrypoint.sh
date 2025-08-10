#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import os, time
import socket
from urllib.parse import urlparse

db_url = os.environ.get("DATABASE_URL", "")
if not db_url:
    raise SystemExit("DATABASE_URL not set")

u = urlparse(db_url.replace("+psycopg2", ""))
host, port = u.hostname, u.port or 5432

for i in range(60):
    try:
        with socket.create_connection((host, port), timeout=2):
            print("Postgres is up")
            break
    except OSError:
        print("Waiting for Postgres...", i)
        time.sleep(1)
else:
    raise SystemExit("Postgres not available")
PY

alembic upgrade head || true

exec opentelemetry-instrument uvicorn app.main:app --host ${BACKEND_HOST:-0.0.0.0} --port ${BACKEND_PORT:-8000} --workers 2