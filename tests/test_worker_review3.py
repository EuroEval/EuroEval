"""Regression tests for the third volunteer-worker review."""

import dataclasses
from pathlib import Path

import pytest

from euroeval_worker import runtime
from euroeval_worker.broker import BrokerError
from euroeval_worker.safety import ModelMetadata, SafetyError, check_model_safety
from euroeval_worker.state import StateStore
from euroeval_worker.types import EEERecord, lease_from_dict
from tests.test_euroeval_worker import GPU, LEASE


def test_heartbeat_persists_renewed_expiry(tmp_path: Path) -> None:
    """A renewal remains usable after the original lease expiry."""
    renewed = dataclasses.replace(
        LEASE, model_profile="bert", expires_at="2099-01-02T00:00:00Z"
    )
    state = StateStore(tmp_path)
    state.save_active(LEASE, github_login="contributor")

    class Broker:
        def heartbeat(self, credential: str, lease_id: str) -> str:
            return renewed.expires_at

    heartbeat = runtime.Heartbeat(
        Broker(), "credential", LEASE, persist=state.renew_active
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
        Broker(), "old", LEASE, reauthenticate=lambda credential: "new"
    )
    heartbeat._renew()
    assert calls == ["old", "new"]
    assert heartbeat.failed is None


def test_result_retry_honours_backoff_and_terminal_statuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Uploads retry transient responses but stop on lease and validation errors."""
    delays: list[float] = []
    monkeypatch.setattr(runtime.time, "sleep", delays.append)
    state = StateStore(tmp_path)
    worker = runtime.Worker(object(), state)  # type: ignore[arg-type]
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

    worker.client = Transient()  # type: ignore[assignment]
    worker._submit_record("credential", LEASE, record)
    assert attempts == 3
    assert delays == [0, 0]

    for status in (409, 422):

        class Terminal:
            def submit_result(self, **kwargs: object) -> None:
                raise BrokerError("terminal", status=status)

        worker.client = Terminal()  # type: ignore[assignment]
        with pytest.raises(BrokerError):
            worker._submit_record("credential", LEASE, record)


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
    worker = runtime.Worker(object(), state)  # type: ignore[arg-type]
    worker._credential = "old"
    worker._login = "contributor"
    record = EEERecord({"id": "one"})

    class Broker:
        def submit_result(
            self, credential: str, lease: object, result: EEERecord
        ) -> None:
            credentials.append(credential)
            if credential == "old":
                raise BrokerError("expired", status=401)

        def finalise(self, credential: str, lease_id: str) -> str:
            credentials.append(credential)
            if credential == "old":
                raise BrokerError("expired", status=401)
            return "stable-submission"

    worker.client = Broker()  # type: ignore[assignment]
    worker._submit_record("old", LEASE, record)
    assert worker._finalise("old", LEASE.lease_id) == "stable-submission"
    assert credentials == ["old", "new", "new"]
    assert state.load_auth() == ("new", "contributor")


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
    worker = runtime.Worker(object(), state)  # type: ignore[arg-type]
    with pytest.raises(runtime.AuthenticationIdentityError):
        worker._reauthenticate("old-credential", "old-login")
    assert state.load_active() is not None


def test_profile_decodes_under_the_wire_name() -> None:
    """The lease decoder accepts model_profile, not the obsolete profile key."""
    wire = dataclasses.asdict(LEASE)
    wire["protocol_version"] = "volunteer-worker/v1"
    assert lease_from_dict(wire).model_profile is None
    wire["model_profile"] = "bert"
    assert lease_from_dict(wire).model_profile == "bert"


def test_profile_and_gpu_safety_are_fail_closed() -> None:
    """Profiles must match the single admitted architecture and fit at 80 percent."""
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
    lease = dataclasses.replace(LEASE, model_profile="roberta")
    assert check_model_safety(
        lease, (GPU,), metadata, free_disk_bytes=1
    ).available_bytes == int(GPU.free_memory_bytes * 0.8)
    with pytest.raises(SafetyError, match="match"):
        check_model_safety(
            dataclasses.replace(lease, model_profile="bert"), (GPU,), metadata
        )
    with pytest.raises(SafetyError, match="unknown"):
        check_model_safety(
            dataclasses.replace(lease, model_profile="unknown"), (GPU,), metadata
        )
