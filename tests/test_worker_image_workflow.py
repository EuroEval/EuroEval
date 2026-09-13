"""Static safety checks for the worker image publication workflow."""

from pathlib import Path

WORKFLOW = Path(__file__).parents[1] / ".github/workflows/worker-image.yaml"


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
    """Return the anonymous manifest verification step."""
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
