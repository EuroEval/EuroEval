"""Shared protocol, HTTP, and Redis foundations for the native API."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
from collections.abc import Mapping
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse

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
    "Slavic languages (Belarusian, Bulgarian, Bosnian, Croatian, Czech, Polish, Serbian, Slovak, Slovenian, Ukrainian)": [
        "be",
        "bg",
        "bs",
        "hr",
        "cs",
        "pl",
        "sr",
        "sk",
        "sl",
        "uk",
    ],
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
        self.client = client
        self.environ = os.environ if environ is None else environ
        self.cache: dict[str, object] = {}

    async def delete(self, key: str) -> None:
        await self.redis("DEL", key)

    async def redis(self, command: str, *args: str) -> object:
        response = await self.fetch(
            "POST",
            self.value("UPSTASH_REDIS_REST_URL"),
            headers={
                "authorization": f"Bearer {self.value('UPSTASH_REDIS_REST_TOKEN')}",
                "content-type": "application/json",
            },
            json=[command, *args],
        )
        if response.status_code >= 400:
            raise BrokerError(503, f"Upstash returned HTTP {response.status_code}.")
        payload = response.json()
        if payload.get("error"):
            raise BrokerError(503, f"Upstash error: {payload['error']}")
        return payload.get("result")

    async def fetch(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        data: Any = None,
        content: Any = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        safe = method.upper() in {"GET", "HEAD", "OPTIONS"}
        for attempt in range(3 if safe else 1):
            try:
                response = await self.client.request(
                    method,
                    url,
                    headers=headers,
                    json=json,
                    data=data,
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
        value = await self.redis("GET", key)
        if not isinstance(value, str):
            return None
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None

    def value(self, name: str, default: str | None = None) -> str:
        value = self.environ.get(name, default)
        if value is None:
            raise ConfigurationError(f"Missing required environment variable: {name}")
        return value

    async def set(self, key: str, value: str, ttl: int, nx: bool = False) -> bool:
        return (
            await self.redis("SET", key, value, "EX", str(ttl), *(["NX"] if nx else []))
            == "OK"
        )


class BrokerError(RuntimeError):
    """An expected broker failure with an HTTP status and optional code."""

    def __init__(self, status: int, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


class ConfigurationError(RuntimeError):
    """Required runtime configuration is unavailable."""


def canonical_json(value: object) -> str:
    def valid(v: object) -> None:
        if isinstance(v, bool) or v is None or isinstance(v, str):
            return
        if isinstance(v, int):
            if abs(v) > 9007199254740991:
                raise ValueError("Integer is outside JavaScript safe range")
            return
        if isinstance(v, float):
            if not __import__("math").isfinite(v):
                raise ValueError("Non-finite number")
            return
        if isinstance(v, list):
            for item in v:
                valid(item)
            return
        if isinstance(v, dict):
            for key, item in v.items():
                if item is not ...:
                    valid(item)
            return
        raise TypeError(f"Unsupported JSON value: {type(v).__name__}")

    valid(value)

    def order(v: object) -> object:
        if isinstance(v, dict):
            return {k: order(v[k]) for k in sorted(v) if v[k] is not ...}
        if isinstance(v, list):
            return [order(x) for x in v]
        return v

    return json.dumps(
        order(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def hmac_digest(secret: str, value: str) -> str:
    return hmac.new(secret.encode(), value.encode(), hashlib.sha256).hexdigest()


def json_response(
    status: int, body: object, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    result = {
        "cache-control": "no-store",
        "access-control-allow-origin": "*",
        "access-control-allow-headers": "authorization, content-type",
        "access-control-allow-methods": "GET, POST, OPTIONS",
    }
    result.update(headers or {})
    return JSONResponse(body, status_code=status, headers=result)


def now() -> int:
    return __import__("time").time_ns() // 1_000_000


def random_token(length: int = 32) -> str:
    return secrets.token_hex(length)


async def read_json(request: Request, limit: int) -> dict[str, object]:
    raw = await request.body()
    if len(raw) > limit:
        raise BrokerError(413, "Request body is too large.")
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise BrokerError(400, "Invalid JSON.") from error
    if not isinstance(body, dict):
        raise BrokerError(400, "JSON body must be an object.")
    return body


def require_protocol(body: Mapping[str, object]) -> None:
    if body.get("protocol") != PROTOCOL_VERSION:
        raise BrokerError(400, "Unsupported protocol version.", "protocol")


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
