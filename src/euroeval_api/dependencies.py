"""Shared request and client dependencies for the native API."""

import os
import typing as t

import httpx
from fastapi import HTTPException, Request
from pydantic_core import from_json
from starlette.requests import ClientDisconnect
from upstash_redis.asyncio import Redis

PROTOCOL_VERSION = "volunteer-worker/v1"


def get_http(request: Request) -> httpx.AsyncClient:
    """Return the application's outbound connection pool.

    Args:
        request: Incoming request.

    Returns:
        Shared HTTP client.

    Raises:
        HTTPException: If application startup has not completed.
    """
    client = request.app.state.http
    if client is None:
        raise HTTPException(status_code=503, detail="HTTP client unavailable")
    return client


async def get_redis(request: Request) -> Redis:
    """Return the shared Redis client without requiring Redis for public assets.

    Args:
        request: Incoming request.

    Returns:
        Official asynchronous Upstash client.
    """
    if request.app.state.redis is None:
        request.app.state.redis = Redis(
            url=require_env(name="UPSTASH_REDIS_REST_URL"),
            token=require_env(name="UPSTASH_REDIS_REST_TOKEN"),
        )
    return request.app.state.redis


def require_env(name: str) -> str:
    """Read required configuration without loading credential files.

    Args:
        name: Environment variable name.

    Returns:
        Configured non-empty value.

    Raises:
        HTTPException: If configuration is absent.
    """
    value = os.getenv(name)
    if not value:
        raise HTTPException(status_code=503, detail=f"Deployment is missing {name}.")
    return value


async def read_json(request: Request, limit: int = 8192) -> dict[str, object]:
    """Read a bounded JSON object using the existing protocol's strict syntax.

    Args:
        request: Incoming request.
        limit (optional): Maximum UTF-8 body size in bytes.

    Returns:
        Parsed request object.

    Raises:
        HTTPException: If the request is disconnected, oversized, or invalid.
    """
    try:
        body = await request.body()
        if len(body) > limit:
            raise HTTPException(status_code=413, detail="Request body is too large.")
        parsed = from_json(body, allow_inf_nan=False)
    except (ValueError, ClientDisconnect) as error:
        raise HTTPException(status_code=400, detail="Invalid JSON") from error
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=400, detail="Request body must be a JSON object."
        )
    return t.cast(dict[str, object], parsed)


def require_protocol(body: dict[str, object]) -> None:
    """Require the deployed worker protocol rather than accepting numeric versions.

    Args:
        body: Parsed worker request.

    Raises:
        HTTPException: If the protocol is unsupported.
    """
    if body.get("protocol_version") != PROTOCOL_VERSION:
        raise HTTPException(
            status_code=400, detail=f"protocol_version must be {PROTOCOL_VERSION}."
        )
