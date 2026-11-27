"""Shared protocol, HTTP, and Redis foundations for the native API."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import typing as t
from collections.abc import Mapping

import httpx
import rfc8785
from fastapi import Request
from fastapi.responses import JSONResponse

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)

PROTOCOL_VERSION = "volunteer-worker/v1"
REPO = "EuroEval/EuroEval"
MARKER_VERSION = 1
MARKER_DOMAIN = "euroeval-volunteer-marker"
REQUEST_LABEL = "model evaluation request"
TITLE_PREFIX = "[MODEL EVALUATION REQUEST]"
GROUPS: dict[str, list[str]] = {
    "Baltic languages (Latvian, Lithuanian)": ["lv", "lt"],
    "Finnic languages (Estonian, Finnish)": ["et", "fi"],
    "Romance languages (Catalan, French, Italian, Portuguese, Romanian, Spanish)": [
        "ca",
        "fr",
        "it",
        "pt",
        "ro",
        "es",
    ],
    "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)": [
        "da",
        "fo",
        "is",
        "no",
        "sv",
    ],
    (
        "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, "
        "Czech, Polish, Serbian, Slovak, Slovenian, Ukrainian)"
    ): ["be", "bg", "bs", "hr", "cs", "pl", "sr", "sk", "sl", "uk"],
    "West Germanic languages (Dutch, English, German, Luxembourgish)": [
        "nl",
        "en",
        "de",
        "lb",
    ],
    "Albanian": ["sq"],
    "Greek": ["el"],
    "Hungarian": ["hu"],
}


class Runtime:
    """Shared outgoing client, configuration, and request-local cache."""

    def __init__(
        self, client: httpx.AsyncClient, environ: Mapping[str, str] | None = None
    ) -> None:
        """Configure the shared runtime or a protocol failure.

        Args:
            client:
                Shared outgoing HTTP client.
            environ:
                Process configuration or explicitly supplied test configuration.
        """
        self.client = client
        self.environ = os.environ if environ is None else environ
        self.cache: dict[str, object] = {}

    async def delete(self, key: str) -> None:
        """Delete the named Redis key.

        Args:
            key:
                Persisted Redis key.
        """
        await self.redis("DEL", key)

    async def redis(self, command: str, *args: str) -> object:
        """Execute a Redis REST command with the existing bounded retries.

        Args:
            command:
                Redis command name.
            args:
                Redis command arguments.

        Returns:
            The command result returned by Redis.

        Raises:
            BrokerError:
                If Redis rejects the command or remains unavailable.
            httpx.HTTPError:
                If transport attempts fail.
            ValueError:
                If Redis returns malformed JSON.
        """
        for attempt in range(3):
            try:
                token = self.value("UPSTASH_REDIS_REST_TOKEN")
                response = await self.fetch(
                    "POST",
                    self.value("UPSTASH_REDIS_REST_URL"),
                    headers={
                        "authorization": f"Bearer {token}",
                        "content-type": "application/json",
                    },
                    json=[command, *args],
                )
                if response.status_code >= 400:
                    raise BrokerError(
                        503, f"Upstash returned HTTP {response.status_code}."
                    )
                payload = response.json()
                if payload.get("error"):
                    raise BrokerError(503, f"Upstash error: {payload['error']}")
                return payload.get("result")
            except (httpx.HTTPError, ValueError, BrokerError):
                if attempt == 2:
                    raise
                await asyncio.sleep(0.1 * (attempt + 1))
        raise BrokerError(503, "Upstash request failed.")

    async def fetch(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json: object | None = None,
        data: Mapping[str, str] | None = None,
        content: bytes | str | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """Send HTTP requests, retrying only safe methods after transient failures.

        Args:
            method:
                HTTP method.
            url:
                Destination URL.
            headers:
                Additional HTTP headers.
            json:
                JSON request body.
            data:
                Form request fields.
            content:
                Exact outgoing body bytes or text.
            timeout:
                Optional I/O deadline in seconds.

        Returns:
            The requested protocol value or operation result.

        Raises:
            AssertionError:
                If the request fails or its protocol requirements are not met.
            httpx.TransportError:
                If the request fails or its protocol requirements are not met.
        """
        safe = method.upper() in {"GET", "HEAD", "OPTIONS"}
        for attempt in range(3 if safe else 1):
            try:
                response = await self.client.request(
                    method,
                    url,
                    headers=headers,
                    json=json,
                    data=dict(data) if data is not None else None,
                    content=content,
                    timeout=timeout,
                )
            except httpx.TransportError:
                if attempt == (2 if safe else 0):
                    raise
                await asyncio.sleep(0.1 * (attempt + 1))
                continue
            if (
                not safe
                or response.status_code not in {408, 425, 429, 500, 502, 503, 504}
                or attempt == 2
            ):
                return response
            retry_after = response.headers.get("retry-after", "")
            delay = min(30, int(retry_after)) if retry_after.isdigit() else 0
            await asyncio.sleep(max(0.1 * (attempt + 1), delay))
        raise AssertionError("unreachable")

    async def get(self, key: str) -> object | None:
        """Read and decode a persisted JSON value.

        Args:
            key:
                Persisted Redis key.

        Returns:
            The requested protocol value or operation result.
        """
        value = await self.redis("GET", key)
        if not isinstance(value, str):
            return None
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None

    def value(self, name: str, default: str | None = None) -> str:
        """Read configuration, failing when a required value is absent.

        Args:
            name:
                Environment variable name.
            default:
                Fallback value, or None for required configuration.

        Returns:
            The requested protocol value or operation result.

        Raises:
            ConfigurationError:
                If the request fails or its protocol requirements are not met.
        """
        value = self.environ.get(name, default)
        if value is None:
            raise ConfigurationError(f"Missing required environment variable: {name}")
        return value

    async def set(self, key: str, value: str, ttl: int, nx: bool = False) -> bool:
        """Store exact bytes in Redis with the required expiry and optional NX fence.

        Args:
            key:
                Persisted Redis key.
            value:
                Protocol value or exact text to persist.
            ttl:
                Expiry in seconds.
            nx:
                Whether the key must not already exist.

        Returns:
            Whether the operation succeeded without losing ownership.
        """
        return (
            await self.redis("SET", key, value, "EX", str(ttl), *(["NX"] if nx else []))
            == "OK"
        )


class BrokerError(RuntimeError):
    """An expected broker failure with an HTTP status and optional code."""

    def __init__(self, status: int, message: str, code: str | None = None) -> None:
        """Configure the shared runtime or a protocol failure.

        Args:
            status:
                HTTP status code.
            message:
                Public failure description.
            code:
                Optional stable protocol error code.
        """
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


class ConfigurationError(RuntimeError):
    """Required runtime configuration is unavailable."""


def _reject_json_constant(value: str) -> None:
    """Reject numeric tokens excluded by the JSON protocol.

    Args:
        value:
            Non-finite numeric token encountered by the decoder.

    Raises:
        ValueError:
            Always, because JavaScript JSON.parse rejects these tokens.
    """
    raise ValueError(f"Invalid JSON number: {value}")


def canonical_json(value: object) -> str:
    """Serialise protocol evidence with JavaScript-compatible canonical JSON.

    Args:
        value:
            JSON evidence with string keys and finite, safe numeric values.

    Returns:
        The canonical UTF-8 text used by existing signatures and digests.
    """
    return rfc8785.dumps(t.cast(JsonValue, value)).decode("utf-8")


def get_runtime(request: Request) -> Runtime:
    """Return the shared runtime attached to the FastAPI application.

    Args:
        request:
            Incoming HTTP request.

    Returns:
        The requested protocol value or operation result.
    """
    return request.app.state.runtime


def hmac_digest(secret: str, value: str) -> str:
    """Sign the exact UTF-8 payload using HMAC-SHA256.

    Args:
        secret:
            Signing secret.
        value:
            Protocol value or exact text to persist.

    Returns:
        The requested protocol value or operation result.
    """
    signature = hmac.new(secret.encode(), value.encode(), hashlib.sha256).digest()
    return base64.b64encode(signature).decode("ascii")


def json_response(
    status: int, body: object, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    """Build a no-store JSON response with the protocol headers.

    Args:
        status:
            HTTP status code.
        body:
            Protocol JSON object.
        headers:
            Additional HTTP headers.

    Returns:
        The requested protocol value or operation result.
    """
    result = {
        "cache-control": "no-store",
        "access-control-allow-origin": "*",
        "access-control-allow-headers": "authorization, content-type",
        "access-control-allow-methods": "GET, POST, OPTIONS",
    }
    result.update(headers or {})
    return JSONResponse(body, status_code=status, headers=result)


def now() -> int:
    """Return the current Unix timestamp in milliseconds.

    Returns:
        The requested protocol value or operation result.
    """
    return time.time_ns() // 1_000_000


def random_token(length: int = 32) -> str:
    """Generate an opaque URL-safe token from secure random bytes.

    Args:
        length:
            Number of secure random bytes.

    Returns:
        The requested protocol value or operation result.
    """
    return secrets.token_hex(length)


async def read_json(request: Request, limit: int) -> dict[str, object]:
    """Read a bounded JSON object, rejecting malformed or oversized requests.

    Args:
        request:
            Incoming HTTP request.
        limit:
            Maximum accepted UTF-8 body size.

    Returns:
        The requested protocol value or operation result.

    Raises:
        BrokerError:
            If the request fails or its protocol requirements are not met.
    """
    raw = await request.body()
    if len(raw) > limit:
        raise BrokerError(413, "Request body is too large.")
    try:
        body = json.loads(raw, parse_constant=_reject_json_constant)
    except (ValueError, UnicodeDecodeError) as error:
        raise BrokerError(400, "Invalid JSON.") from error
    if not isinstance(body, dict):
        raise BrokerError(400, "JSON body must be an object.")
    return body


def require_protocol(body: Mapping[str, object]) -> None:
    """Reject an unsupported volunteer-worker protocol version.

    Args:
        body:
            Protocol JSON object.

    Raises:
        BrokerError:
            If the request fails or its protocol requirements are not met.
    """
    if body.get("protocol") != PROTOCOL_VERSION:
        raise BrokerError(400, "Unsupported protocol version.", "protocol")


def sha256(value: str) -> str:
    """Hash exact UTF-8 bytes using SHA-256.

    Args:
        value:
            Protocol value or exact text to persist.

    Returns:
        The requested protocol value or operation result.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
