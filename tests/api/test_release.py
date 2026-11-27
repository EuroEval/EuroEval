"""Tests for the native volunteer-worker release route."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from packaging.version import Version

from euroeval_api.app import create_app
from euroeval_api.release import canonical_worker_version


def test_release_route_rejects_missing_and_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report unavailable release metadata as a service configuration error."""
    monkeypatch.delenv("VOLUNTEER_WORKER_IMAGE_DIGEST", raising=False)
    monkeypatch.delenv("VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64", raising=False)
    monkeypatch.setenv("VOLUNTEER_WORKER_VERSION", "not a version")
    monkeypatch.setenv("EUROEVAL_VERSION", "1.0")
    client = TestClient(create_app())

    missing = client.get("/api/worker/release-info?architecture=amd64")
    assert missing.status_code == 503

    monkeypatch.setenv("VOLUNTEER_WORKER_IMAGE_DIGEST", "sha256:" + "A" * 64)
    invalid = client.get("/api/worker/release-info?architecture=amd64")
    assert invalid.status_code == 503


def test_release_route_supports_architecture_aliases_and_arm_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the preferred digest and canonical architecture aliases."""
    monkeypatch.setenv("VOLUNTEER_WORKER_IMAGE_DIGEST", f"sha256:{'a' * 64}")
    monkeypatch.setenv("VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64", f"sha256:{'b' * 64}")
    monkeypatch.setenv("VOLUNTEER_WORKER_VERSION", "v1.0.0")
    monkeypatch.setenv("EUROEVAL_VERSION", "2.0.dev")
    client = TestClient(create_app())

    amd64 = client.get("/api/worker/release-info?architecture=x86_64")
    arm64 = client.get("/api/worker/release-info?architecture=aarch64")

    assert amd64.status_code == 200
    assert amd64.json()["architecture"] == "amd64"
    assert amd64.json()["worker_version"] == "v1.0.0"
    assert amd64.json()["euroeval_version"] == "2.0.dev0"
    assert amd64.json()["image_digest"] == f"sha256:{'a' * 64}"
    assert arm64.status_code == 200
    assert arm64.json()["architecture"] == "arm64"
    assert arm64.json()["image_digest"] == f"sha256:{'b' * 64}"


def test_worker_version_uses_pep440_equivalence() -> None:
    """Treat equivalent PEP 440 version spellings as the same version."""
    first = canonical_worker_version("v1.0.0")
    second = canonical_worker_version("1")

    assert first == Version("1")
    assert first == second
    assert canonical_worker_version(object()) is None
