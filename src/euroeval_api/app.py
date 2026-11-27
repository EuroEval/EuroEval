"""Native EuroEval HTTP application and shared route dependencies."""

from __future__ import annotations

import json
import os
import typing as t
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic_core import from_json
from upstash_redis.asyncio import Redis


@asynccontextmanager
async def _lifespan(app: FastAPI) -> t.AsyncIterator[None]:
    """Manage the shared outbound HTTP connection pool."""
    client = app.state.http
    owned = client is None
    if owned:
        client = httpx.AsyncClient()
        app.state.http = client
    try:
        yield
    finally:
        if owned:
            await client.aclose()


def create_app(
    *,
    http_client: httpx.AsyncClient | None = None,
    evaluated_models: dict[str, list[str]] | None = None,
) -> FastAPI:
    """Create the native HTTP application.

    Args:
        http_client (optional): Shared outbound client. Defaults to an owned client.
        evaluated_models (optional): Evaluated model index used instead of the asset.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        lifespan=_lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        redirect_slashes=False,
    )
    app.state.http = http_client
    app.state.evaluated_models = evaluated_models
    app.state.redis = None

    @app.get("/evaluated-models.json")
    async def evaluated_models_index(request: Request) -> JSONResponse:
        models = request.app.state.evaluated_models
        if models is None:
            try:
                models = json.loads(Path("dist/evaluated-models.json").read_text())
            except (OSError, json.JSONDecodeError):
                return JSONResponse(
                    {"error": "Evaluated model index unavailable"}, status_code=503
                )
        return JSONResponse(models, headers={"cache-control": "public, max-age=300"})

    return app


def get_http(request: Request) -> httpx.AsyncClient:
    """Return the application's shared HTTP client.

    Args:
        request: Incoming request.

    Returns:
        Shared asynchronous HTTP client.
    """
    return request.app.state.http


async def get_redis(request: Request) -> object:
    """Return a lazily created Upstash Redis client.

    Args:
        request: Incoming request.

    Returns:
        Upstash asynchronous Redis client.
    """
    if request.app.state.redis is None:
        request.app.state.redis = _create_redis()
    return request.app.state.redis


def _create_redis() -> object:
    """Create an Upstash client from required credentials.

    Returns:
        Official asynchronous Redis client.
    """
    return Redis(
        url=require_env("UPSTASH_REDIS_REST_URL"),
        token=require_env("UPSTASH_REDIS_REST_TOKEN"),
    )


def require_env(name: str) -> str:
    """Read a required environment variable.

    Args:
        name: Environment variable name.

    Returns:
        Non-empty configured value.

    Raises:
        HTTPException: If the variable is unset or empty.
    """
    value = os.getenv(name)
    if not value:
        raise HTTPException(status_code=503, detail=f"{name} is not configured")
    return value


async def read_json(request: Request, limit: int = 8192) -> object:
    """Read one bounded, strict JSON request body.

    Args:
        request: Incoming request.
        limit (optional): Maximum body size in bytes. Defaults to 8192.

    Returns:
        Parsed JSON value.

    Raises:
        HTTPException: If the body is too large or malformed.
    """
    body = await request.body()
    if len(body) > limit:
        raise HTTPException(status_code=413, detail="Request body is too large")
    try:
        return from_json(body, allow_inf_nan=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc


def require_protocol(body: object) -> int:
    """Validate the worker protocol version in a JSON object.

    Args:
        body: Parsed JSON body.

    Returns:
        Supported protocol version.

    Raises:
        HTTPException: If the body has no supported version.
    """
    if not isinstance(body, dict) or body.get("protocol_version") != 1:
        raise HTTPException(status_code=400, detail="Unsupported protocol version")
    return 1
