"""Standalone native HTTP application for EuroEval."""

from __future__ import annotations

import json
import os
import typing as t
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .dependencies import PROTOCOL_VERSION
from .dependencies import get_http as get_http
from .dependencies import get_redis as get_redis
from .dependencies import read_json as read_json
from .dependencies import require_env as require_env
from .dependencies import require_protocol as require_protocol


class _Transport:
    """Enforce bounded bodies and route-specific CORS without trusting proxy headers."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the native application.

        Args:
            app: Native ASGI application.
        """
        self.app = app
        self.origins = {
            origin.strip()
            for origin in os.getenv(
                "API_ALLOWED_ORIGINS", "https://euroeval.com"
            ).split(",")
            if origin.strip()
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Apply per-request transport limits and origin handling.

        Args:
            scope: ASGI request metadata.
            receive: Incoming request channel.
            send: Outgoing response channel.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        origin = headers.get("origin")
        allowed = origin in self.origins

        async def send_headers(message: Message) -> None:
            """Apply CORS and cache metadata to actual HTTP responses.

            Args:
                message: Outgoing ASGI message.
            """
            if message["type"] == "http.response.start":
                outgoing = MutableHeaders(scope=message)
                outgoing.setdefault("cache-control", "no-store")
                vary = outgoing.get("vary", "")
                if "origin" not in vary.lower().split(","):
                    outgoing["vary"] = f"{vary}, Origin".strip(", ")
                if "access-control-allow-origin" in outgoing:
                    del outgoing["access-control-allow-origin"]
                if allowed:
                    outgoing["access-control-allow-origin"] = t.cast(str, origin)
            await send(message)

        if scope["method"] == "OPTIONS":
            routes = t.cast(FastAPI, scope["app"]).routes
            methods = {
                method
                for route in routes
                if isinstance(route, APIRoute) and route.path == scope["path"]
                for method in route.methods
            }
            requested = headers.get("access-control-request-method", "")
            supplied = {
                header.strip().lower()
                for header in headers.get("access-control-request-headers", "").split(
                    ","
                )
                if header.strip()
            }
            permitted = {
                "content-type",
                "authorization",
                "x-coordinator-secret",
                "x-promotion-secret",
            }
            status = (
                404
                if not methods
                else 403
                if (origin and not allowed) or supplied - permitted
                else 405
                if requested and requested not in methods
                else 204
            )
            response = Response(
                status_code=status,
                headers={
                    "access-control-allow-methods": ", ".join(
                        sorted(methods | {"OPTIONS"})
                    ),
                    "access-control-allow-headers": ", ".join(sorted(permitted)),
                },
            )
            await response(scope, receive, send_headers)
            return
        declared = headers.get("content-length", "0")
        if declared.isdecimal() and int(declared) > 2 * 1024 * 1024:
            await JSONResponse(
                {"error": "Request body is too large."}, status_code=413
            )(scope, receive, send_headers)
            return
        consumed = 0

        async def bounded_receive() -> Message:
            """Bound streamed bodies even when Content-Length is absent or false.

            Returns:
                Next bounded ASGI request message.

            Raises:
                HTTPException: If the stream exceeds two MiB.
            """
            nonlocal consumed
            message = await receive()
            consumed += len(message.get("body", b""))
            if consumed > 2 * 1024 * 1024:
                raise HTTPException(
                    status_code=413, detail="Request body is too large."
                )
            return message

        await self.app(scope, bounded_receive, send_headers)


async def _evaluated_index(request: Request) -> JSONResponse:
    """Serve the validated index used to prevent duplicate evaluation requests.

    Args:
        request: Incoming request.

    Returns:
        Cacheable generated index, or an unavailable response.
    """
    try:
        index = get_evaluated_models(request=request)
    except HTTPException as error:
        return JSONResponse({"error": error.detail}, status_code=error.status_code)
    return JSONResponse(index, headers={"cache-control": "public, max-age=300"})


def get_evaluated_models(request: Request) -> dict[str, list[str]]:
    """Load and validate the packaged index before any issue-creation decision.

    Args:
        request: Incoming request.

    Returns:
        Model IDs mapped to their evaluated language groups.

    Raises:
        HTTPException: If the index is missing or malformed.
    """
    index = request.app.state.evaluated_models
    try:
        if index is None:
            index = json.loads(Path("dist/evaluated-models.json").read_text())
    except (OSError, ValueError) as error:
        raise HTTPException(
            status_code=503, detail="Evaluated model index unavailable"
        ) from error
    if not isinstance(index, dict) or any(
        not isinstance(key, str)
        or not isinstance(groups, list)
        or any(not isinstance(group, str) for group in groups)
        for key, groups in index.items()
    ):
        raise HTTPException(status_code=503, detail="Evaluated model index unavailable")
    request.app.state.evaluated_models = index
    return t.cast(dict[str, list[str]], index)


async def _invalid_request(request: Request, error: Exception) -> JSONResponse:
    """Reject invalid JSON/parameters with the existing API's 400 status.

    Args:
        request: Incoming request.
        error: Native validation failure.

    Returns:
        Public protocol failure.
    """
    return await _http_error(
        request=request, error=HTTPException(status_code=400, detail="Invalid request")
    )


async def _http_error(request: Request, error: Exception) -> JSONResponse:
    """Keep public error objects compatible with existing browser and worker clients.

    Args:
        request: Incoming request.
        error: Native HTTP failure.

    Returns:
        Top-level protocol error rather than FastAPI's detail wrapper.
    """
    failure = t.cast(StarletteHTTPException, error)
    body = (
        dict(failure.detail)
        if isinstance(failure.detail, dict)
        else {"error": failure.detail}
    )
    if request.url.path.startswith("/api/worker/"):
        body.setdefault("protocol_version", PROTOCOL_VERSION)
    headers = dict(failure.headers or {})
    if failure.status_code == 429:
        headers.setdefault("retry-after", "60")
    return JSONResponse(body, status_code=failure.status_code, headers=headers)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> t.AsyncIterator[None]:
    """Own the outbound HTTP pool for the application's lifetime.

    Args:
        app: Application entering or leaving service.

    Yields:
        Control while the application serves requests.
    """
    owned = app.state.http is None
    if owned:
        app.state.http = httpx.AsyncClient()
    try:
        yield
    finally:
        if app.state.redis is not None:
            await app.state.redis.close()
        if owned:
            await app.state.http.aclose()


def create_app(
    *,
    http_client: httpx.AsyncClient | None = None,
    evaluated_models: dict[str, list[str]] | None = None,
) -> FastAPI:
    """Create the native API without importing the evaluation/ML framework.

    Args:
        http_client (optional): Supplied outbound client, otherwise lifespan-owned.
        evaluated_models (optional): Supplied generated model index.

    Returns:
        Configured FastAPI application.
    """
    application = FastAPI(
        lifespan=_lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        redirect_slashes=False,
    )
    application.state.http = http_client
    application.state.redis = None
    application.state.evaluated_models = evaluated_models
    application.add_middleware(_Transport)
    application.add_exception_handler(StarletteHTTPException, _http_error)
    application.add_exception_handler(RequestValidationError, _invalid_request)
    application.add_api_route(
        "/evaluated-models.json", _evaluated_index, methods=["GET"]
    )
    return application


app = create_app()
