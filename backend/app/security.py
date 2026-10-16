"""Auth: API-key header check on all /api routes (except health/ready)."""
from __future__ import annotations

from fastapi import Header

from .config import settings
from .errors import APIError, UNAUTHORIZED


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
    if not x_api_key or x_api_key != settings.API_KEY:
        raise APIError(UNAUTHORIZED[0], "Missing or invalid X-API-Key header.", UNAUTHORIZED[1])
    return x_api_key
