"""Tests for public volunteer-worker release interfaces."""

import pytest

from scripts.release_volunteer_worker import (
    validate_deployment_evidence,
    validate_finalization_evidence,
    validate_stage_evidence,
)


def test_deployment_requires_explicit_snapshot_and_versions() -> None:
    """A deployment without a verified public-config snapshot fails closed."""
    plan = validate_stage_evidence(candidate())
    evidence = {
        "plan": plan,
        "worker_version": plan["worker_version"],
        "euroeval_version": plan["euroeval_version"],
        "policy_hash": plan["policy_hash"],
        "expected_previous_release": "sha256:" + "e" * 64,
        "production_only": True,
        "preserve_unrelated_variables": True,
        "config_snapshot_verified": False,
    }
    with pytest.raises(ValueError):
        validate_deployment_evidence(evidence)


def candidate() -> dict[str, object]:
    """Return complete fake trusted workflow evidence."""
    digest = "sha256:" + "a" * 64
    return {
        "physical_acceptance": {
            architecture: {
                "accepted": True,
                "source_sha": "1" * 40,
                "record_id": f"acceptance-{architecture}",
            }
            for architecture in ("amd64", "arm64")
        },
        "repository": "EuroEval/EuroEval",
        "workflow_run_conclusion": "success",
        "ci_run_conclusion": "success",
        "run_sha": "1" * 40,
        "artifact_run_id": 123,
        "run_id": 123,
        "event": "push",
        "branch": "main",
        "sha": "1" * 40,
        "run_attempt": 2,
        "index_digest": digest,
        "combined_index_digest": digest,
        "platform_digests": {
            "amd64": "sha256:" + "b" * 64,
            "arm64": "sha256:" + "c" * 64,
        },
        "worker_version": "1.2.3",
        "euroeval_version": "2.3.4",
        "policy_hash": "sha256:" + "d" * 64,
    }


def test_finalization_requires_both_architectures_and_latest() -> None:
    """Finalization cannot rely on an amd64-only deployment check."""
    plan = validate_stage_evidence(candidate())
    evidence = {
        "plan": plan,
        "deployment_ready": True,
        "production_alias_verified": True,
        "production_architectures": {"amd64"},
        "latest_digest": plan["image_digest"],
        "release_info_verified": True,
    }
    with pytest.raises(ValueError):
        validate_finalization_evidence(evidence)


def test_stage_emits_immutable_plan_for_both_platforms() -> None:
    """The plan points only at the combined immutable index."""
    plan = validate_stage_evidence(candidate())
    assert (
        plan["image_reference"] == f"{plan['image_repository']}@{plan['image_digest']}"
    )
    platforms = plan["platform_digests"]
    assert isinstance(platforms, dict)
    assert set(platforms) == {"amd64", "arm64"}


def test_stage_rejects_untrusted_event() -> None:
    """Pull requests cannot yield release plans."""
    evidence = candidate()
    evidence["event"] = "pull_request"
    with pytest.raises(ValueError):
        validate_stage_evidence(evidence)
