"""Regression tests for the third volunteer-worker review."""

import dataclasses
import json
import sys
import types
import typing as t
from pathlib import Path

import pytest
from transformers import BertConfig, ViTConfig

from euroeval_worker import runtime, safety
from euroeval_worker.broker import BrokerError, BrokerProtocol
from euroeval_worker.hardware import NoGpuError
from euroeval_worker.safety import ModelMetadata, SafetyError, check_model_safety
from euroeval_worker.state import StateStore
from euroeval_worker.types import EEERecord, Gpu, ModelEvidence, lease_from_dict
from tests.test_euroeval_worker import GPU, HARDWARE, LEASE


def test_active_lease_persists_identity_and_gpu_selection(tmp_path: Path) -> None:
    """Restart state keeps the contributor and UUID-pinned GPU."""
    state = StateStore(tmp_path)
    state.save_active(LEASE, github_login="contributor")

    active = state.load_active()
    assert active is not None
    assert active.github_login == "contributor"
    assert active.lease.selected_gpu_uuid == GPU.uuid
    assert active.lease.selected_gpu_index == GPU.index
    assert '"github_login":"contributor"' in (tmp_path / "active-lease.json").read_text(
        encoding="utf-8"
    )


def test_heartbeat_persists_renewed_expiry(tmp_path: Path) -> None:
    """A renewal remains usable after the original lease expiry."""
    renewed = dataclasses.replace(
        LEASE, model_type="encoder", expires_at="2099-01-02T00:00:00Z"
    )
    state = StateStore(tmp_path)
    state.save_active(LEASE, github_login="contributor")

    class Broker:
        def heartbeat(self, credential: str, lease_id: str) -> str:
            return renewed.expires_at

    heartbeat = runtime.Heartbeat(
        t.cast(BrokerProtocol, Broker()),
        "credential",
        LEASE,
        persist=state.renew_active,
    )
    heartbeat._renew()
    assert state.load_active() is not None
    assert state.load_active().lease.expires_at == renewed.expires_at


def test_heartbeat_reauthenticates_once_and_retries() -> None:
    """A 401 heartbeat retries with the replacement credential."""
    calls: list[str] = []

    class Broker:
        def heartbeat(self, credential: str, lease_id: str) -> str:
            calls.append(credential)
            if credential == "old":
                raise BrokerError("expired", status=401)
            return "2099-01-02T00:00:00Z"

    heartbeat = runtime.Heartbeat(
        t.cast(BrokerProtocol, Broker()),
        "old",
        LEASE,
        reauthenticate=lambda credential: "new",
    )
    heartbeat._renew()
    assert calls == ["old", "new"]
    assert heartbeat.failed is None


def test_model_type_and_gpu_safety_are_fail_closed() -> None:
    """Capability evidence must match and fit at 80 percent."""
    metadata = ModelMetadata(
        private=False,
        gated=False,
        auto_map=False,
        files=("config.json", "model.safetensors"),
        safetensors=True,
        estimated_bytes=1,
        architectures=("RobertaModel",),
        repository_bytes=1,
        pipeline_tag="fill-mask",
        model_type="encoder",
        backend_compatible=True,
    )
    lease = dataclasses.replace(LEASE, model_type="encoder")
    larger_gpu = dataclasses.replace(
        GPU,
        uuid="GPU-2",
        free_memory_bytes=30 * 1024**3,
        total_memory_bytes=40 * 1024**3,
    )
    assert check_model_safety(
        lease, (larger_gpu, GPU), metadata, free_disk_bytes=1, selected_gpu=GPU
    ).available_bytes == int(GPU.free_memory_bytes * 0.8)
    with pytest.raises(SafetyError, match="match"):
        check_model_safety(
            dataclasses.replace(lease, model_type="generative"), (GPU,), metadata
        )
    with pytest.raises(SafetyError, match="unsupported"):
        check_model_safety(
            dataclasses.replace(lease, model_type="unknown"), (GPU,), metadata
        )


def test_huggingface_metadata_records_immutable_capability_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fetch pipeline metadata and run the installed-stack preflight."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"model_type": "novel", "architectures": ["NovelBaseModel"]}),
        encoding="utf-8",
    )

    class Info:
        id = "org/model"
        private = False
        gated = False
        pipeline_tag = "fill-mask"
        siblings = [
            type("Sibling", (), {"rfilename": "config.json", "size": 10})(),
            type("Sibling", (), {"rfilename": "model.safetensors", "size": 20})(),
        ]

    class Api:
        def __init__(self, token: bool) -> None:
            assert token is False

        def model_info(self, **kwargs: object) -> Info:
            assert kwargs["revision"] == "a" * 40
            return Info()

    monkeypatch.setattr(safety, "HfApi", Api)
    monkeypatch.setattr(safety, "hf_hub_download", lambda **kwargs: str(config_path))

    class Config:
        model_type = "novel"

    monkeypatch.setattr(
        safety.AutoConfig, "from_pretrained", lambda *args, **kwargs: Config()
    )
    monkeypatch.setattr(safety, "_installed_backend_supports", lambda **kwargs: True)

    metadata = safety.HuggingFaceMetadata().fetch("org/model", "a" * 40)
    assert metadata.pipeline_tag == "fill-mask"
    assert metadata.architectures == ("NovelBaseModel",)
    assert metadata.backend_compatible


def test_encoder_backend_requires_each_leased_task_mapping() -> None:
    """Require task-specific Transformers mappings, not base-model membership."""
    config = BertConfig()
    for task_group in (
        "sequence_classification",
        "token_classification",
        "question_answering",
        "multiple_choice_classification",
    ):
        assert safety._installed_backend_supports(
            model_type="encoder",
            config=config,
            architectures=("BertModel",),
            pipeline_tag="text-classification",
            task_groups=(task_group,),
        )
    assert not safety._installed_backend_supports(
        model_type="encoder",
        config=ViTConfig(),
        architectures=("ViTModel",),
        pipeline_tag="image-classification",
        task_groups=("sequence_classification",),
    )
    assert not safety._installed_backend_supports(
        model_type="encoder",
        config=config,
        architectures=("BertModel",),
        pipeline_tag="image-classification",
        task_groups=("sequence_classification",),
    )


def test_vllm_inspection_rejects_pooling_only_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use vLLM's model inspection result rather than registry membership."""

    class Info:
        is_text_generation_model = False
        is_pooling_model = True

    class Registry:
        models = {"PoolingModel": object()}

        @staticmethod
        def get_supported_archs() -> object:
            return Registry.models.keys()

        @staticmethod
        def inspect_model_cls(
            architectures: object, config: object
        ) -> tuple[Info, str]:
            return Info(), "PoolingModel"

    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    registry_module = types.ModuleType("vllm.model_executor.models.registry")
    registry_module.ModelRegistry = Registry
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor", types.ModuleType("vllm.model_executor")
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models",
        types.ModuleType("vllm.model_executor.models"),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.registry", registry_module
    )
    assert not safety._installed_backend_supports(
        model_type="generative",
        config=object(),
        architectures=("PoolingModel",),
        pipeline_tag="text-generation",
    )
    Info.is_text_generation_model = True
    Info.is_pooling_model = False
    assert safety._installed_backend_supports(
        model_type="generative",
        config=object(),
        architectures=("PoolingModel",),
        pipeline_tag="text-generation",
    )


def test_pinned_vllm_registry_is_used_when_installed() -> None:
    """Exercise the pinned registry in environments that provide vLLM."""
    vllm = pytest.importorskip("vllm")
    version = getattr(vllm, "__version__", "")
    assert version.startswith("0.27.1")
    assert safety._installed_backend_supports(
        model_type="generative",
        config=object(),
        architectures=("LlamaForCausalLM",),
        pipeline_tag="text-generation",
    )


def test_model_metadata_is_required_and_contradictions_fail_closed() -> None:
    """Reject missing, contradictory, and unverified capability evidence."""
    metadata = ModelMetadata(
        private=False,
        gated=False,
        auto_map=False,
        files=("config.json", "model.safetensors"),
        safetensors=True,
        estimated_bytes=1,
        architectures=("InventedModel",),
        repository_bytes=1,
        pipeline_tag="text-generation",
        model_type="encoder",
        backend_compatible=False,
    )
    with pytest.raises(SafetyError, match="capability"):
        check_model_safety(
            dataclasses.replace(LEASE, model_type="generative"), (GPU,), metadata
        )
    with pytest.raises(SafetyError, match="backend"):
        check_model_safety(
            dataclasses.replace(
                LEASE,
                model_type="generative",
                model_metadata=ModelEvidence(
                    pipeline_tag="text-generation",
                    architectures=("InventedModel",),
                    model_type="generative",
                ),
            ),
            (GPU,),
            ModelMetadata(
                private=False,
                gated=False,
                auto_map=False,
                files=("config.json", "model.safetensors"),
                safetensors=True,
                estimated_bytes=1,
                architectures=("InventedModel",),
                repository_bytes=1,
                pipeline_tag="text-generation",
                model_type="generative",
                backend_compatible=False,
            ),
        )
    with pytest.raises(SafetyError, match="immutable"):
        check_model_safety(
            dataclasses.replace(LEASE, model_metadata=None),
            (GPU,),
            ModelMetadata(
                private=False,
                gated=False,
                auto_map=False,
                files=("config.json", "model.safetensors"),
                safetensors=True,
                estimated_bytes=1,
                architectures=("InventedModel",),
                repository_bytes=1,
                pipeline_tag="fill-mask",
                model_type="encoder",
                backend_compatible=True,
            ),
        )


def test_model_type_is_required_under_the_wire_name() -> None:
    """The lease decoder accepts only the capability-based model_type field."""
    wire = dataclasses.asdict(LEASE)
    wire["protocol_version"] = "volunteer-worker/v1"
    del wire["model_type"]
    with pytest.raises(ValueError, match="model_type"):
        lease_from_dict(wire)
    wire["model_type"] = "generative"
    wire["model_metadata"]["model_type"] = "generative"
    assert lease_from_dict(wire).model_type == "generative"


def test_reauthentication_rejects_different_contributor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A replacement login preserves the active evidence and submits nothing."""
    state = StateStore(tmp_path)
    state.save_active(LEASE, github_login="old-login")
    state.save_auth("old-credential", "old-login")
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("new-credential", "new-login")
    )
    worker = runtime.Worker(t.cast(BrokerProtocol, object()), state)
    with pytest.raises(runtime.AuthenticationIdentityError):
        worker._reauthenticate("old-credential", "old-login")
    assert state.load_active() is not None


def test_restart_requires_the_leased_gpu_uuid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing UUID-pinned GPU preserves evidence and performs no upload."""
    state = StateStore(tmp_path)
    state.save_active(LEASE, github_login="login")
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    broker = t.cast(BrokerProtocol, object())
    worker = runtime.Worker(
        client=broker,
        state=state,
        hardware_factory=lambda: dataclasses.replace(
            HARDWARE, gpus=(Gpu("A100", "GPU-other", 10, 20, "8.0", 4),)
        ),
    )

    with pytest.raises(NoGpuError):
        worker.run(once=True)
    assert state.load_active() is not None


def test_result_and_finalise_reauthenticate_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Upload and finalisation resume with a replacement same-login credential."""
    state = StateStore(tmp_path)
    state.save_auth("old", "contributor")
    credentials: list[str] = []
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("new", "contributor")
    )
    worker = runtime.Worker(t.cast(BrokerProtocol, object()), state)
    worker._credential = "old"
    worker._login = "contributor"
    record = EEERecord({"id": "one"})

    class Broker:
        def finalise(self, credential: str, lease_id: str) -> str:
            credentials.append(credential)
            if credential == "old":
                raise BrokerError("expired", status=401)
            return "stable-submission"

        def submit_result(
            self, credential: str, lease: object, result: EEERecord
        ) -> None:
            credentials.append(credential)
            if credential == "old":
                raise BrokerError("expired", status=401)

    worker.client = t.cast(BrokerProtocol, Broker())
    worker._submit_record("old", LEASE, record)
    assert worker._finalise("old", LEASE.lease_id) == "stable-submission"
    assert credentials == ["old", "new", "new"]
    assert state.load_auth() == ("new", "contributor")


def test_result_retry_honours_backoff_and_terminal_statuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Uploads retry transient responses but stop on lease and validation errors."""
    delays: list[float] = []
    monkeypatch.setattr(runtime.time, "sleep", delays.append)
    state = StateStore(tmp_path)
    worker = runtime.Worker(t.cast(BrokerProtocol, object()), state)
    worker._credential = "credential"
    worker._login = "contributor"
    record = EEERecord({"id": "one"})
    attempts = 0

    class Transient:
        def submit_result(self, **kwargs: object) -> None:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise BrokerError("busy", status=503, retry_after=0)

    worker.client = t.cast(BrokerProtocol, Transient())
    worker._submit_record("credential", LEASE, record)
    assert attempts == 3
    assert delays == [0, 0]

    for status in (409, 422):

        class Terminal:
            def submit_result(self, **kwargs: object) -> None:
                raise BrokerError("terminal", status=status)

        worker.client = t.cast(BrokerProtocol, Terminal())
        with pytest.raises(BrokerError):
            worker._submit_record("credential", LEASE, record)
