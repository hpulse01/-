# H-Pulse Quantum Prediction System

Monorepo containing backend (FastAPI), frontend (Next.js), ops (Docker/Compose), docs, and scripts.

## Requirements
- Docker 24+ and Docker Compose v2
- Make (optional)

## Quick Start
1. Copy env: `cp .env.example .env` and review values
2. Start: `docker compose up -d --build`
3. Run migrations: `make migrate`
4. Seed admin: `make seed`
5. Open UI: `http://localhost:3000`

## Health and Observability
- Backend health: `GET http://localhost:8000/healthz`
- OpenAPI/Swagger: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`
- Prometheus metrics: `GET http://localhost:8000/metrics`

## Default Admin
- Email: `admin@hpulse.local`
- Password: `Admin123!#`
- Change these in `.env` before seeding in production.

## Security Initialization
- Set strong `JWT_SECRET` and restrict `CORS_ALLOW_ORIGINS`
- Rotate admin password and create additional admin users
- Consider configuring `OTEL_EXPORTER_OTLP_ENDPOINT`

## Makefile targets
- `make up` / `make down`
- `make test` / `make lint` / `make fmt`
- `make migrate` / `make seed`

## Directory Structure
```
backend/        FastAPI app, Celery worker, tests, Dockerfile
frontend/       Next.js 14 app with TailwindCSS
ops/            Infra assets (db init scripts)
docs/           Architecture, sequences, data dictionary, threat model, compliance
scripts/        Utility scripts (license generation)
.github/        GitHub Actions workflows
```

## Design Decisions (ADR)
See ADR entries under this section.

### ADR-0001: Vector store choice
- Decision: Use PostgreSQL + pgvector
- Rationale: Lower operational complexity vs running separate Qdrant service; ACID transactions; single backup/restore path
- Consequences: Vector search performance is sufficient at small/medium scale; may revisit Qdrant for larger scale

## FAQ
- DB extensions: `ops/db/init/01-enable-extensions.sql` enables pgvector automatically
- Ports: 3000 (frontend), 8000 (backend), 5432 (Postgres), 6379 (Redis)
- Credentials: configured via `.env`

## License
AGPL-3.0. See `LICENSE` and `NOTICE`. Generate third-party licenses:
```
bash scripts/gen_third_party_licenses.sh
```

## ADR Template
```
# ADR-XXXX: Title
- Status: Proposed/Accepted/Rejected/Deprecated
- Context: What is the problem?
- Decision: What is decided?
- Consequences: Positive/negative outcome
- Alternatives: Options considered
- References: Links
```