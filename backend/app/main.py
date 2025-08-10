from __future__ import annotations

from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from .config import get_settings, ErrorResponse
from .logging import configure_logging
from .rate_limit import limiter
from .observability import setup_tracing
from .routers import auth as auth_router
from .routers import users as users_router
from .routers import predictions as predictions_router

settings = get_settings()
configure_logging(settings.log_level)
setup_tracing(service_name=settings.otel_service_name)

app = FastAPI(
    title="H-Pulse Backend",
    version="0.1.0",
    description="API for H-Pulse Quantum Prediction System",
    contact={"name": "H-Pulse Team", "email": "dev@hpulse.local"},
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: JSONResponse({"detail": "rate limit exceeded"}, status_code=429))
app.add_middleware(SlowAPIMiddleware)

# CORS
allow_origins: List[str] = (
    [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    if "," in settings.cors_allow_origins or settings.cors_allow_origins != "*"
    else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz", tags=["health"], responses={200: {"description": "OK"}})
async def healthz():
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
async def metrics():
    data = generate_latest()
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"service": "hpulse-backend", "version": "0.1.0"})


# Routers
app.include_router(auth_router.router, prefix="/api/v1", tags=["auth"])
app.include_router(users_router.router, prefix="/api/v1", tags=["users"])
app.include_router(predictions_router.router, prefix="/api/v1", tags=["predictions"])