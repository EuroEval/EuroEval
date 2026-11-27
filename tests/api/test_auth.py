"""Public HTTP workflows for legacy GitHub device authorisation."""

from __future__ import annotations

import collections.abc as c
import hashlib
import json

import httpx
import pytest
from upstash_redis.asyncio import Redis

from euroeval_api.app import create_app
from euroeval_api.auth import router
from euroeval_api.dependencies import PROTOCOL_VERSION

_APP_SETTINGS = (
    "VOLUNTEER_GITHUB_APP_CLIENT_ID",
    "VOLUNTEER_GITHUB_APP_CLIENT_SECRET",
    "VOLUNTEER_GITHUB_APP_ID",
    "VOLUNTEER_GITHUB_APP_PRIVATE_KEY",
    "VOLUNTEER_GITHUB_APP_INSTALLATION_ID",
    "VOLUNTEER_GITHUB_APP_REPOSITORY_ID",
    "VOLUNTEER_GITHUB_APP_TOKEN_ENCRYPTION_KEY",
)


@pytest.fixture
async def auth_client(
    monkeypatch: pytest.MonkeyPatch, upstash_client: Redis
) -> c.AsyncIterator[httpx.AsyncClient]:
    """Provide the public auth routes, real Redis, and a mocked GitHub transport.

    Args:
        monkeypatch: Isolated process environment updates.
        upstash_client: Official async SDK connected to real local Redis.

    Yields:
        HTTP client dispatching requests through the ASGI application.
    """
    for name in _APP_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_ID", "test-oauth-client")
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_SECRET", "test-oauth-secret")
    provider = httpx.MockTransport(_github_response)
    async with httpx.AsyncClient(transport=provider) as outbound:
        application = create_app(http_client=outbound, evaluated_models={})
        application.include_router(router)
        application.state.redis = upstash_client
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://testserver"
        ) as client:
            yield client


@pytest.mark.anyio
async def test_authorisation_is_single_use_and_revocable(
    auth_client: httpx.AsyncClient, upstash_client: Redis
) -> None:
    """Issue only a hashed opaque credential and revoke its local authority.

    Args:
        auth_client: Public HTTP workflow client.
        upstash_client: Real Redis SDK for observable credential storage.
    """
    started = await auth_client.post(
        "/api/worker/auth/start", json={"protocol_version": PROTOCOL_VERSION}
    )
    assert started.status_code == 200
    session = started.json()["session_id"]
    authorised = await auth_client.post(
        "/api/worker/auth/poll",
        json={"protocol_version": PROTOCOL_VERSION, "session_id": session},
    )
    assert authorised.status_code == 200
    result = authorised.json()
    credential = result["credential"]
    assert result["status"] == "authorised"
    assert len(credential) == 43
    digest = hashlib.sha256(credential.encode()).hexdigest()
    stored = await upstash_client.get(f"euroeval:worker:credential:{digest}")
    assert stored is not None
    assert credential not in str(stored)

    replay = await auth_client.post(
        "/api/worker/auth/poll",
        json={"protocol_version": PROTOCOL_VERSION, "session_id": session},
    )
    assert replay.status_code == 400
    revoked = await auth_client.post(
        "/api/worker/auth/revoke", headers={"authorization": f"Bearer {credential}"}
    )
    assert revoked.status_code == 200
    assert await upstash_client.get(f"euroeval:worker:credential:{digest}") is None


@pytest.mark.anyio
async def test_failed_provider_revocation_never_issues_credential(
    monkeypatch: pytest.MonkeyPatch, upstash_client: Redis
) -> None:
    """Fail closed if GitHub does not revoke its temporary OAuth grant.

    Args:
        monkeypatch: Isolated process environment updates.
        upstash_client: Real Redis SDK for the public workflow.
    """
    for name in _APP_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_ID", "test-oauth-client")
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_SECRET", "test-oauth-secret")

    def failed_revoke(request: httpx.Request) -> httpx.Response:
        """Return provider success except for the required grant deletion.

        Args:
            request: Outbound provider request.

        Returns:
            Endpoint-specific GitHub response.
        """
        response = _github_response(request)
        if request.method == "DELETE":
            return httpx.Response(500, json={"message": "revocation unavailable"})
        return response

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(failed_revoke)
    ) as outbound:
        application = create_app(http_client=outbound, evaluated_models={})
        application.include_router(router)
        application.state.redis = upstash_client
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://testserver"
        ) as client:
            started = await client.post(
                "/api/worker/auth/start", json={"protocol_version": PROTOCOL_VERSION}
            )
            result = await client.post(
                "/api/worker/auth/poll",
                json={
                    "protocol_version": PROTOCOL_VERSION,
                    "session_id": started.json()["session_id"],
                },
            )
    assert result.status_code == 503
    assert "credential" not in result.json()
    assert await upstash_client.keys("euroeval:worker:credential:*") == []


def _github_response(request: httpx.Request) -> httpx.Response:
    """Model the exact legacy device, identity, and grant-revocation endpoints.

    Args:
        request: Outbound request issued by the authentication workflow.

    Returns:
        Provider response matching the requested operation.
    """
    if request.method == "POST" and request.url.path == "/login/device/code":
        assert json.loads(request.content) == {
            "client_id": "test-oauth-client",
            "scope": "read:user",
        }
        return httpx.Response(
            200,
            json={
                "device_code": "device-secret",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://github.com/login/device",
                "expires_in": 600,
                "interval": 5,
            },
        )
    if request.method == "POST" and request.url.path == "/login/oauth/access_token":
        assert json.loads(request.content) == {
            "client_id": "test-oauth-client",
            "device_code": "device-secret",
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }
        return httpx.Response(200, json={"access_token": "temporary-github-token"})
    if request.method == "GET" and request.url.path == "/user":
        assert request.headers["authorization"] == "Bearer temporary-github-token"
        return httpx.Response(200, json={"login": "volunteer"})
    if (
        request.method == "DELETE"
        and request.url.path == "/applications/test-oauth-client/grant"
    ):
        assert request.headers.get("authorization", "").startswith("Basic ")
        assert json.loads(request.content) == {"access_token": "temporary-github-token"}
        return httpx.Response(200)
    return httpx.Response(404, json={"message": "Unexpected provider request"})
