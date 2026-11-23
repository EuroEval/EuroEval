"""Tests for the external GGUF execution contract in `euroeval_worker.types`."""

import dataclasses

import pytest

from euroeval_worker.types import (
    ExternalExecution,
    GgufArtifact,
    HardwareReport,
    Lease,
    ModelEvidence,
    execution_from_dict,
    external_execution_from_dict,
    gguf_artifact_from_dict,
    lease_from_dict,
)

PINNED_ARTIFACT = {
    "repo_id": "LiquidAI/LFM3-1.7B-GGUF",
    "revision": "cd0c74c575b5c416f3011b5b4c298077f26ca8c2",
    "filename": "LFM3-1.7B-Q8_0.gguf",
    "sha256": "8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6",
    "source_repo_id": "LiquidAI/LFM3-1.7B",
    "source_revision": "1da06bec6a819da10b54c67e5c90972b46d81871",
}


def test_execution_from_dict_absent_means_historical_cuda() -> None:
    """A missing or null execution discriminator decodes to CUDA mode."""
    assert execution_from_dict(value=None) is None
    assert (
        execution_from_dict(
            value={"kind": "external_server", "artifact": dict(PINNED_ARTIFACT)}
        ).kind
        == "external_server"
    )  # type: ignore[union-attr]


def test_execution_from_dict_rejects_unknown_kinds_and_shapes() -> None:
    """Unknown execution kinds and malformed objects are rejected."""
    with pytest.raises(ValueError, match="kind"):
        execution_from_dict(
            value={"kind": "cuda_server", "artifact": dict(PINNED_ARTIFACT)}
        )
    with pytest.raises(ValueError, match="kind"):
        execution_from_dict(value={"artifact": dict(PINNED_ARTIFACT)})
    with pytest.raises(ValueError):
        execution_from_dict(value="external_server")
    with pytest.raises(ValueError):
        execution_from_dict(value={"kind": "external_server"})
    with pytest.raises(ValueError):
        execution_from_dict(
            value={
                "kind": "external_server",
                "artifact": dict(PINNED_ARTIFACT),
                "extra": True,
            }
        )


def test_external_execution_from_dict_decodes_artifact_strictly() -> None:
    """The external execution object wraps one strictly decoded artifact."""
    execution = external_execution_from_dict(
        data={"kind": "external_server", "artifact": dict(PINNED_ARTIFACT)}
    )
    assert execution == ExternalExecution(
        kind="external_server",
        artifact=gguf_artifact_from_dict(data=dict(PINNED_ARTIFACT)),
    )
    broken = dict(PINNED_ARTIFACT)
    broken["filename"] = "../evil.gguf"
    with pytest.raises(ValueError, match="filename"):
        external_execution_from_dict(
            data={"kind": "external_server", "artifact": broken}
        )


def test_gguf_artifact_decodes_the_pinned_contract_artifact() -> None:
    """The contract's fixed approved artifact decodes strictly and is frozen."""
    artifact = gguf_artifact_from_dict(data=dict(PINNED_ARTIFACT))
    assert artifact == GgufArtifact(
        repo_id="LiquidAI/LFM3-1.7B-GGUF",
        revision="cd0c74c575b5c416f3011b5b4c298077f26ca8c2",
        filename="LFM3-1.7B-Q8_0.gguf",
        sha256="8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6",
        source_repo_id="LiquidAI/LFM3-1.7B",
        source_revision="1da06bec6a819da10b54c67e5c90972b46d81871",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        artifact.filename = "other.gguf"  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    argnames=["field", "value"],
    argvalues=[
        ("revision", "c" * 39),
        ("revision", "c" * 41),
        ("revision", "g" * 40),
        ("revision", "C" * 40),
        ("revision", ""),
        ("sha256", "8" * 63),
        ("sha256", "8" * 65),
        ("sha256", "z" * 64),
        ("sha256", "8F6B24182258205E9E01E21F1F24A1F208C1E8115578825D202CE39976AF20B6"),
        ("source_revision", "not-hex"),
        ("repo_id", "NoSlash"),
        ("repo_id", "too/many/slashes"),
        ("repo_id", ""),
        ("repo_id", "org/ model"),
        ("source_repo_id", "/leading"),
        ("filename", 123),
        ("revision", None),
    ],
)
def test_gguf_artifact_rejects_malformed_fields(field: str, value: object) -> None:
    """Non-hex revisions/hashes and malformed repo ids fail closed.

    Args:
        field:
            Artifact field to corrupt.
        value:
            Invalid replacement value for the field.
    """
    data: dict[str, object] = dict(PINNED_ARTIFACT)
    data[field] = value
    with pytest.raises(ValueError):
        gguf_artifact_from_dict(data=data)


def test_gguf_artifact_rejects_missing_and_extra_keys() -> None:
    """The artifact object must carry exactly the six contract fields."""
    missing = dict(PINNED_ARTIFACT)
    del missing["sha256"]
    with pytest.raises(ValueError, match="artifact"):
        gguf_artifact_from_dict(data=missing)
    extra = dict(PINNED_ARTIFACT)
    extra["quantization"] = "Q8_0"
    with pytest.raises(ValueError, match="artifact"):
        gguf_artifact_from_dict(data=extra)


@pytest.mark.parametrize(
    argnames="filename",
    argvalues=[
        "../LFM3-1.7B-Q8_0.gguf",
        "nested/LFM3-1.7B-Q8_0.gguf",
        "back\\slash.gguf",
        "LFM3-1.7B-Q8_0.GGUF",
        "LFM3-1.7B-Q8_0.gguf.bak",
        "no-extension",
        "",
        ".",
        "..",
        ".gguf",
        "spaced name.gguf",
        "tab\tname.gguf",
    ],
)
def test_gguf_artifact_rejects_unsafe_filenames(filename: str) -> None:
    """Only a single safe ``.gguf`` basename is a valid artifact filename.

    Args:
        filename:
            Candidate filename that must be rejected.
    """
    data: dict[str, object] = dict(PINNED_ARTIFACT)
    data["filename"] = filename
    with pytest.raises(ValueError, match="filename"):
        gguf_artifact_from_dict(data=data)


def test_hardware_report_can_represent_no_gpus() -> None:
    """Native external mode reports hardware without any GPU."""
    report = HardwareReport(
        architecture="arm64",
        ram_bytes=64,
        free_disk_bytes=100,
        driver_version=None,
        cuda_version=None,
        pytorch_version=None,
    )
    assert report.gpus == ()
    assert report.selected_gpu_index is None
    assert report.selected_gpu_uuid is None


def test_lease_execution_field_defaults_to_none() -> None:
    """Existing Lease constructors keep working with execution defaulted."""
    lease = Lease(
        lease_id="lease-1",
        issue_number=42,
        model_id="org/model",
        model_revision="a" * 40,
        language="da",
        euroeval_version="18.0.0",
        image_digest="sha256:image",
        expires_at="2099-01-01T00:00:00Z",
        model_type="generative",
        model_metadata=ModelEvidence(
            pipeline_tag="text-generation",
            architectures=("LlamaForCausalLM",),
            model_type="generative",
        ),
        expected_scope=None,
    )
    assert lease.execution is None
    assert lease.image_digest == "sha256:image"


def test_lease_execution_round_trips_through_asdict() -> None:
    """Both execution modes survive a persisted dict round trip."""
    cuda = lease_from_dict(data=_cuda_lease_dict())
    cuda_state = dataclasses.asdict(cuda)
    cuda_state["protocol_version"] = "volunteer-worker/v1"
    assert lease_from_dict(data=cuda_state) == cuda
    external = lease_from_dict(data=_external_lease_dict())
    external_state = dataclasses.asdict(external)
    external_state["protocol_version"] = "volunteer-worker/v1"
    assert lease_from_dict(data=external_state) == external


def _cuda_lease_dict() -> dict[str, object]:
    """Build one historical CUDA lease response.

    Returns:
        A minimal valid volunteer-worker/v1 lease payload.
    """
    return {
        "protocol_version": "volunteer-worker/v1",
        "lease_id": "lease-1",
        "issue_number": 42,
        "model_id": "org/model",
        "model_revision": "a" * 40,
        "language": "da",
        "euroeval_version": "18.0.0",
        "image_digest": "sha256:image",
        "worker_version": "worker-1",
        "expires_at": "2099-01-01T00:00:00Z",
        "model_type": "generative",
        "selected_gpu_uuid": "GPU-1",
        "selected_gpu_index": 0,
        "model_metadata": {
            "pipeline_tag": "text-generation",
            "architectures": ["LlamaForCausalLM"],
            "model_type": "generative",
            "is_encoder_decoder": None,
        },
        "expected_scope": {
            "policy_version": "test-policy",
            "language_group": "da",
            "allowed_identity_suffix_sets": [['["test",false,true]']],
            "warnings": [],
            "task_groups": ["text_generation"],
        },
    }


def _external_lease_dict() -> dict[str, object]:
    """Build one external-server lease response with a null image digest.

    Returns:
        A valid external execution lease payload.
    """
    data = _cuda_lease_dict()
    data["image_digest"] = None
    del data["selected_gpu_uuid"]
    del data["selected_gpu_index"]
    data["execution"] = {"kind": "external_server", "artifact": dict(PINNED_ARTIFACT)}
    return data


def test_lease_from_dict_keeps_historical_cuda_meaning() -> None:
    """A lease without an execution object keeps its CUDA image and GPU."""
    decoded = lease_from_dict(data=_cuda_lease_dict())
    assert decoded.execution is None
    assert decoded.image_digest == "sha256:image"
    assert decoded.selected_gpu_uuid == "GPU-1"
    assert decoded.selected_gpu_index == 0
    explicit_null = _cuda_lease_dict()
    explicit_null["execution"] = None
    assert lease_from_dict(data=explicit_null).execution is None


def test_lease_from_dict_parses_external_lease_with_null_image() -> None:
    """An external lease carries the artifact and no image digest or GPU."""
    decoded = lease_from_dict(data=_external_lease_dict())
    assert decoded.image_digest is None
    assert decoded.selected_gpu_uuid is None
    assert decoded.selected_gpu_index is None
    assert decoded.execution == ExternalExecution(
        kind="external_server",
        artifact=gguf_artifact_from_dict(data=dict(PINNED_ARTIFACT)),
    )


def test_lease_from_dict_rejects_cuda_lease_with_null_image() -> None:
    """Historical CUDA leases still require a nonempty image digest."""
    null_image = _cuda_lease_dict()
    null_image["image_digest"] = None
    with pytest.raises(ValueError, match="image_digest"):
        lease_from_dict(data=null_image)
    empty_image = _cuda_lease_dict()
    empty_image["image_digest"] = ""
    with pytest.raises(ValueError, match="image_digest"):
        lease_from_dict(data=empty_image)


def test_lease_from_dict_rejects_external_image_or_gpu_contradictions() -> None:
    """External leases must not claim a CUDA image digest or a GPU."""
    with_image = _external_lease_dict()
    with_image["image_digest"] = "sha256:image"
    with pytest.raises(ValueError, match="image_digest"):
        lease_from_dict(data=with_image)
    with_gpu = _external_lease_dict()
    with_gpu["selected_gpu_uuid"] = "GPU-1"
    with pytest.raises(ValueError, match="GPU"):
        lease_from_dict(data=with_gpu)
    with_index = _external_lease_dict()
    with_index["selected_gpu_index"] = 0
    with pytest.raises(ValueError, match="GPU"):
        lease_from_dict(data=with_index)
