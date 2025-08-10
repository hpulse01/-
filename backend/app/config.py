from __future__ import annotations

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, case_sensitive=False)

    env: str = "development"

    database_url: str
    redis_url: str

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    log_level: str = "info"

    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expires_minutes: int = 60

    rate_limit_per_minute: int = 120
    cors_allow_origins: str = "*"

    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "hpulse-backend"
    prometheus_metrics_port: int = 9000


class ErrorResponse(BaseModel):
    code: str
    message: str


def get_settings() -> Settings:
    return Settings()