"""HTTP tests for coordinator mutex endpoints."""

from __future__ import annotations

import typing as t

import httpx
import pytest
from fastapi import FastAPI
from upstash_redis.asyncio import Redis

from euroeval_api.app import create_app
from euroeval_api.coordinator import router
from euroeval_worker import PROTOCOL_VERSION


@pytest.mark.anyio
async def test_coordinator_lock_ownership_lifecycle(
    upstash_client: Redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose busy and ownership-lost outcomes through coordinator HTTP routes."""
    monkeypatch.setenv("WORKER_COORDINATOR_SECRET", "coordinator-secret")
    application: FastAPI = create_app(
        http_client=httpx.AsyncClient(), evaluated_models={}
    )
    application.state.redis = upstash_client
    application.include_router(router)
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"x-coordinator-secret": "coordinator-secret"}

        async def post(path: str, payload: dict[str, object]) -> httpx.Response:
            """Send a protocol request to a coordinator endpoint.

            Args:
                path: Endpoint path.
                payload: Request fields.

            Returns:
                HTTP response.
            """
            return await client.post(
                path,
                headers=headers,
                json={"protocol_version": PROTOCOL_VERSION, **payload},
            )

        acquired = await post("/api/worker/coordinator-lock", {"issue_number": 987654})
        assert acquired.status_code == 200
        owner_token = acquired.json()["token"]
        busy = await post("/api/worker/coordinator-lock", {"issue_number": 987654})
        assert busy.status_code == 409

        wrong_renew = await post(
            "/api/worker/coordinator-renew",
            {"issue_number": 987654, "token": "not-owner"},
        )
        wrong_release = await post(
            "/api/worker/coordinator-release",
            {"issue_number": 987654, "token": "not-owner"},
        )
        assert wrong_renew.status_code == 409
        assert wrong_release.status_code == 200

        renewed = await post(
            "/api/worker/coordinator-renew",
            {"issue_number": 987654, "token": owner_token},
        )
        assert renewed.status_code == 200
        released = await post(
            "/api/worker/coordinator-release",
            {"issue_number": 987654, "token": owner_token},
        )
        assert released.status_code == 200
        assert released.json()["status"] == "released"
        reacquired = await post(
            "/api/worker/coordinator-lock", {"issue_number": 987654}
        )
        assert reacquired.status_code == 200
        await post(
            "/api/worker/coordinator-release",
            {"issue_number": 987654, "token": t.cast(str, reacquired.json()["token"])},
        )
