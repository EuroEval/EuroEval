"""Approved volunteer-worker release information."""

from __future__ import annotations

import os
import re
import typing as t

from fastapi import APIRouter, HTTPException, Query
from packaging.version import InvalidVersion, Version

from .dependencies import PROTOCOL_VERSION, require_env

Architecture = t.Literal["amd64", "arm64"]
IMAGE_REPOSITORY = "ghcr.io/euroeval/euroeval-worker"

router = APIRouter()


@router.get("/api/worker/release-info")
def release_info(architecture: str | None = Query(default=None)) -> dict[str, str]:
    """Return approved release metadata for a worker architecture.

    Args:
        architecture: Requested architecture query value.

    Returns:
        Protocol and approved release metadata.

    Raises:
        HTTPException: If architecture is missing, unsupported, or configuration
            is invalid.
    """
    if not architecture:
        raise HTTPException(status_code=400, detail="architecture is required.")
    normalized = normalize_architecture(architecture)
    if normalized is None:
        raise HTTPException(status_code=400, detail="Unsupported worker architecture.")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "launcher_protocol_version": "volunteer-launcher/v1",
        **approved_worker_release(architecture=normalized),
        "image_repository": IMAGE_REPOSITORY,
    }


def approved_worker_release(architecture: Architecture) -> dict[str, str]:
    """Resolve the configured approved worker release.

    Args:
        architecture: Canonical target architecture.

    Returns:
        Release metadata preserving configured version spellings.

    Raises:
        HTTPException: If required configuration or its values are invalid.
    """
    digest = ""
    if architecture == "arm64":
        digest = os.getenv("VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64", "")
    digest = digest or require_env("VOLUNTEER_WORKER_IMAGE_DIGEST")
    if re.fullmatch(r"sha256:[a-f0-9]{64}", digest) is None:
        raise HTTPException(
            status_code=503, detail="Invalid approved worker image digest."
        )

    worker_version = require_env("VOLUNTEER_WORKER_VERSION")
    euroeval_version = require_env("EUROEVAL_VERSION")
    if euroeval_version.endswith(".dev"):
        euroeval_version += "0"
    if (
        canonical_worker_version(worker_version) is None
        or canonical_worker_version(euroeval_version) is None
    ):
        raise HTTPException(status_code=503, detail="Invalid worker release version.")

    return {
        "architecture": architecture,
        "image_digest": digest,
        "image_reference": f"{IMAGE_REPOSITORY}@{digest}",
        "worker_version": worker_version,
        "euroeval_version": euroeval_version,
    }


def canonical_worker_version(value: object) -> Version | None:
    """Parse a worker version using PEP 440 equivalence.

    Args:
        value: Candidate version value.

    Returns:
        Parsed version, or None when invalid or not a string.
    """
    if not isinstance(value, str):
        return None
    try:
        return Version(value.strip())
    except InvalidVersion:
        return None


def normalize_architecture(value: str) -> Architecture | None:
    """Normalise a supported worker architecture alias.

    Args:
        value: Architecture query value.

    Returns:
        Canonical architecture, or None when unsupported.
    """
    if value in {"x86_64", "amd64"}:
        return "amd64"
    if value in {"aarch64", "arm64"}:
        return "arm64"
    return None
