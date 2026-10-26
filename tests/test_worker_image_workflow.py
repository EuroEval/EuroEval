"""Static safety checks for the worker image publication workflow."""

from pathlib import Path

WORKFLOW = Path(__file__).parents[1] / ".github/workflows/worker-image.yaml"
DOCKERFILE = Path(__file__).parents[1] / "Dockerfile.worker"


def test_amd64_publication_remains_separate_from_spark_candidate() -> None:
    """The existing amd64 candidate and its strict manifest check stay unchanged."""
    contents = WORKFLOW.read_text(encoding="utf-8")
    amd64_job = contents[: contents.index("  spark-arm64-candidate:")]

    assert "platforms: linux/amd64" in amd64_job
    assert (
        '(.platform.os == "linux" and .platform.architecture == "amd64")' in amd64_job
    )
    assert "prefix=spark-arm64-sha-" not in amd64_job


def test_anonymous_verification_proves_tag_digest_and_amd64() -> None:
    """Both references and the required platform descriptor are checked."""
    step = _verification_step()

    assert 'digest_manifest="$(docker manifest inspect' in step
    assert 'tag_manifest="$(docker manifest inspect' in step
    assert step.count("jq -S -c .") == 2
    assert "Candidate SHA tag does not match its digest reference." in step
    assert '(.platform.os == "linux" and .platform.architecture == "amd64")' in step
    assert "(.digest != null)" in step


def _verification_step() -> str:
    """Return the anonymous manifest verification step.

    Returns:
        The workflow YAML for the verification step.
    """
    contents = WORKFLOW.read_text(encoding="utf-8")
    start = contents.index("      - name: Verify candidate manifest anonymously")
    end = contents.index("      - name: Publish candidate metadata", start)
    return contents[start:end]


def test_anonymous_verification_uses_standard_docker_client() -> None:
    """Verification must not depend on a Buildx builder or credentials."""
    step = _verification_step()

    assert 'docker manifest inspect "$REFERENCE"' in step
    assert 'docker manifest inspect "$SHA_TAG"' in step
    assert "docker buildx imagetools inspect" not in step
    assert 'export DOCKER_CONFIG="$ANONYMOUS_DOCKER_CONFIG"' in step
    assert 'rm -rf "$ANONYMOUS_DOCKER_CONFIG"' in step
    assert 'mkdir -m 700 -p "$ANONYMOUS_DOCKER_CONFIG"' in step
    assert "unset BUILDX_BUILDER" in step
    assert "trap cleanup EXIT" in step
    assert 'rm -rf "$PREVIOUS_DOCKER_CONFIG"' in step


def test_metadata_and_upload_are_gated_after_verification() -> None:
    """Metadata cannot be written or uploaded after a failed verification."""
    contents = WORKFLOW.read_text(encoding="utf-8")
    verification = contents.index("      - name: Verify candidate manifest anonymously")
    metadata = contents.index("      - name: Publish candidate metadata")
    upload = contents.index("      - name: Upload candidate metadata")

    assert verification < metadata < upload
    assert "if: success() && github.event_name != 'pull_request'" in contents[metadata:]
    assert "if: success() && github.event_name != 'pull_request'" in contents[upload:]


def test_spark_arm64_is_a_distinct_candidate_and_not_promotable() -> None:
    """ARM64 must publish to an isolated candidate tag, never the amd64 channel."""
    contents = WORKFLOW.read_text(encoding="utf-8")
    spark_job = contents[contents.index("  spark-arm64-candidate:") :]

    assert "runs-on: ubuntu-24.04-arm" in spark_job
    assert "platforms: linux/arm64" in spark_job
    assert "prefix=spark-arm64-sha-" in spark_job
    assert (
        '(.platform.os == "linux" and .platform.architecture == "arm64")' in spark_job
    )
    assert (
        "Spark candidate manifest does not contain a linux/arm64 child digest."
        in spark_job
    )
    assert "blocked-pending-physical-spark-canary" in spark_job
    assert "pending a physical DGX Spark canary" in spark_job
    assert "latest" not in spark_job
    assert "type=sha,format=long,prefix=sha-" not in spark_job


def test_worker_runtime_includes_pinned_nvcc() -> None:
    """Provide nvcc for generative startup without changing amd64 publication."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    runtime = dockerfile.split("\nFROM ", maxsplit=2)[-1]

    assert (
        "nvidia/cuda:13.1.1-devel-ubuntu24.04@sha256:"
        "9cf8694a27722418a1f175d90f85d5afb5a728fd4a9907d7f0565efecfa14d32" in runtime
    )
    assert "PATH=/opt/euroeval/.venv/bin:/usr/local/cuda/bin:$PATH" in runtime
    assert "USER worker" in runtime

    workflow = WORKFLOW.read_text(encoding="utf-8")
    amd64_job = workflow[: workflow.index("  spark-arm64-candidate:")]
    assert "platforms: linux/amd64" in amd64_job
    assert workflow.count('docker run --rm --entrypoint nvcc "$IMAGE" --version') == 2
