"""Validate and exchange immutable volunteer-worker release plans.

The command deliberately does not publish images or mutate Vercel. Trusted
workflow and local deployment adapters consume the validated JSON interfaces.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROTOCOL_VERSION = "volunteer-worker/v1"
LAUNCHER_PROTOCOL_VERSION = "volunteer-launcher/v1"
IMAGE_REPOSITORY = "ghcr.io/euroeval/euroeval-worker"
ARCHITECTURES = {"amd64", "arm64"}


def main(argv: list[str] | None = None) -> int:
    """Validate release metadata using a non-mutating CLI.

    Args:
        argv: Optional command-line arguments. Defaults to process arguments.

    Returns:
        Zero on success, or one when the supplied evidence is unsafe.
    """
    arguments = parse_arguments(argv)
    try:
        data = json.loads(arguments.input.read_text())
        if arguments.command == "stage":
            plan = validate_stage_evidence(data)
            arguments.output.write_text(json.dumps(plan, indent=2) + "\n")
        elif arguments.command == "prepare-deployment":
            validate_deployment_evidence(data)
        else:
            validate_finalization_evidence(data)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"release check failed: {error}", file=sys.stderr)
        return 1
    return 0


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse the release-plan command line.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Parsed command-line options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("stage", "prepare-deployment", "finalize-verify"):
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        if name == "stage":
            command.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def validate_deployment_evidence(evidence: dict[str, object]) -> None:
    """Require explicit plan/source/policy and previous-release agreement.

    Args:
        evidence: Deployment inputs and verified public configuration snapshot.

    Raises:
        ValueError: If the plan, versions, or deployment safety checks disagree.
    """
    plan = evidence.get("plan")
    if not isinstance(plan, dict) or plan.get("status") != "ready":
        raise ValueError("a ready release plan is required")
    for field in ("worker_version", "euroeval_version", "policy_hash"):
        if evidence.get(field) != plan.get(field):
            raise ValueError(f"deployment {field} does not match the staged plan")
    if not isinstance(evidence.get("expected_previous_release"), str):
        raise ValueError("expected previous release is required")
    if (
        evidence.get("production_only") is not True
        or evidence.get("preserve_unrelated_variables") is not True
    ):
        raise ValueError(
            "deployment must target production variables only and preserve others"
        )
    if evidence.get("config_snapshot_verified") is not True:
        raise ValueError(
            "public configuration snapshot/intervening-change check missing"
        )


def validate_finalization_evidence(evidence: dict[str, object]) -> None:
    """Permit finalization only after verified production deployment and discovery.

    Args:
        evidence: Production deployment and public discovery evidence.

    Raises:
        ValueError: If deployment or either architecture is not verified.
    """
    plan = evidence.get("plan")
    if not isinstance(plan, dict) or plan.get("status") != "ready":
        raise ValueError("a ready release plan is required")
    if (
        evidence.get("deployment_ready") is not True
        or evidence.get("production_alias_verified") is not True
    ):
        raise ValueError(
            "successful READY deployment and production alias are required"
        )
    if evidence.get("production_architectures") != ARCHITECTURES:
        raise ValueError("both production architectures must match")
    if evidence.get("latest_digest") != plan.get("image_digest"):
        raise ValueError("anonymous latest index does not match the plan")
    if evidence.get("release_info_verified") is not True:
        raise ValueError("anonymous release-info verification is required")


def validate_stage_evidence(evidence: dict[str, object]) -> dict[str, object]:
    """Produce a ready plan only from trusted, complete dual-architecture evidence.

    Args:
        evidence: Workflow-provided candidate and baseline evidence.

    Returns:
        A small public release plan containing the immutable index digest.

    Raises:
        ValueError: If provenance, acceptance, baseline, or platform evidence fails.
    """
    # Acceptance records describe physical tests, not caller assertions. They are
    # deliberately explicit per architecture; absent records always hold release.
    acceptance = evidence.get("physical_acceptance")
    if not isinstance(acceptance, dict) or set(acceptance) != ARCHITECTURES:
        raise ValueError("physical acceptance records for both architectures required")
    if any(
        not isinstance(record, dict)
        or record.get("accepted") is not True
        or record.get("source_sha") != evidence.get("sha")
        or not isinstance(record.get("record_id"), str)
        for record in acceptance.values()
    ):
        raise ValueError("candidate-specific physical acceptance is missing")
    if evidence.get("event") != "push" or evidence.get("branch") != "main":
        raise ValueError("only trusted main push evidence is eligible")
    if evidence.get("repository") != "EuroEval/EuroEval":
        raise ValueError("candidate repository is not trusted")
    if evidence.get("workflow_run_conclusion") != "success":
        raise ValueError("candidate workflow run must have completed successfully")
    if evidence.get("ci_run_conclusion") != "success":
        raise ValueError("matching CI run must have completed successfully")
    if evidence.get("run_sha") != evidence.get("sha"):
        raise ValueError("candidate run SHA does not match source SHA")
    if evidence.get("artifact_run_id") != evidence.get("run_id"):
        raise ValueError("candidate artifact is not from the selected workflow run")
    sha = evidence.get("sha")
    attempt = evidence.get("run_attempt")
    digest = evidence.get("index_digest")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("invalid source SHA")
    if not isinstance(attempt, int) or attempt < 1:
        raise ValueError("invalid workflow attempt")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("invalid immutable index digest")
    platforms = evidence.get("platform_digests")
    if not isinstance(platforms, dict) or set(platforms) != ARCHITECTURES:
        raise ValueError("both architecture manifests are required")
    if any(
        not isinstance(value, str) or not value.startswith("sha256:")
        for value in platforms.values()
    ):
        raise ValueError("invalid platform manifest digest")
    if evidence.get("combined_index_digest") != digest:
        raise ValueError("platforms do not resolve through the approved combined index")
    return {
        "schema": "euroeval-volunteer-worker-release/v1",
        "status": "ready",
        "source_sha": sha,
        "run_attempt": attempt,
        "worker_version": evidence["worker_version"],
        "euroeval_version": evidence["euroeval_version"],
        "policy_hash": evidence["policy_hash"],
        "image_repository": IMAGE_REPOSITORY,
        "image_digest": digest,
        "image_reference": f"{IMAGE_REPOSITORY}@{digest}",
        "platform_digests": platforms,
        "protocol_version": PROTOCOL_VERSION,
        "launcher_protocol_version": LAUNCHER_PROTOCOL_VERSION,
    }


if __name__ == "__main__":
    raise SystemExit(main())
