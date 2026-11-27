"""Public HTTP outcomes for the native application."""

import pytest
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient

from euroeval_api.app import create_app, get_http, read_json


def test_cors_and_bounded_invalid_json() -> None:
    """Allowed preflight and invalid bounded JSON use public HTTP responses."""
    app = create_app(evaluated_models={})
    router = APIRouter()

    @router.post("/probe")
    async def probe(
        body: object = Depends(read_json), client: object = Depends(get_http)
    ) -> object:
        return body

    app.include_router(router)
    client = TestClient(app)
    preflight = client.options(
        "/evaluated-models.json",
        headers={
            "Origin": "https://euroeval.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert preflight.status_code == 204
    invalid = client.post("/probe", content=b"not json")
    assert invalid.status_code == 400
    assert invalid.json()["error"] == "Invalid JSON"


def test_evaluated_index_is_cached_and_served() -> None:
    """The configured model index is available as a cacheable public asset."""
    client = TestClient(create_app(evaluated_models={"models": ["one"]}))
    response = client.get("/evaluated-models.json")
    assert response.status_code == 200
    assert response.json() == {"models": ["one"]}
    assert response.headers["cache-control"] == "public, max-age=300"


@pytest.mark.parametrize("action", ["claim", "result", "release", "heartbeat"])
def test_worker_actions_require_authentication(action: str) -> None:
    """Expose worker actions through the application without anonymous authority.

    Args:
        action:
            Protected worker operation.
    """
    client = TestClient(create_app(evaluated_models={}))
    response = client.post(f"/api/worker/{action}", json={})
    assert response.status_code == 401
    assert "error" in response.json()
