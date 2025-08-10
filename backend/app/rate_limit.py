from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from redis import Redis

from .config import get_settings

settings = get_settings()

redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
limiter = Limiter(key_func=get_remote_address, storage_uri=settings.redis_url)