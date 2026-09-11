"""Regression tests for the third volunteer-worker review."""

import dataclasses
import typing as t
from pathlib import Path

import pytest

from euroeval_worker import runtime
from euroeval_worker.broker import BrokerError, BrokerProtocol
from euroeval_worker.hardware import NoGpuError
from euroeval_worker.safety import ModelMetadata, SafetyError, check_model_safety
from euroeval_worker.state import StateStore
from euroeval_worker.types import EEERecord, Gpu, lease_from_dict
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
    """Capabilities must match the architecture and fit at 80 percent."""
    metadata = ModelMetadata(
        private=False,
        gated=False,
        auto_map=False,
        files=("config.json", "model.safetensors"),
        safetensors=True,
        estimated_bytes=1,
        architectures=("RobertaForSequenceClassification",),
        repository_bytes=1,
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


def test_model_type_is_required_under_the_wire_name() -> None:
    """The lease decoder accepts only the capability-based model_type field."""
    wire = dataclasses.asdict(LEASE)
    wire["protocol_version"] = "volunteer-worker/v1"
    del wire["model_type"]
    with pytest.raises(ValueError, match="model_type"):
        lease_from_dict(wire)
    wire["model_type"] = "generative"
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
