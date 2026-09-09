"""Tests for the volunteer worker's broker-facing orchestration."""

import dataclasses
import json
import os
from pathlib import Path

import pytest

from euroeval_worker import evaluator, runtime
from euroeval_worker.auth import authenticate
from euroeval_worker.hardware import NoGpuError, discover_gpus
from euroeval_worker.safety import ModelMetadata, SafetyError, check_model_safety
from euroeval_worker.state import StateStore
from euroeval_worker.types import (
    AuthPoll,
    AuthStart,
    Claim,
    EEERecord,
    Gpu,
    HardwareReport,
    Lease,
)

REVISION = "a" * 40
LEASE = Lease(
    lease_id="lease-1",
    issue_number=42,
    model_id="org/model",
    model_revision=REVISION,
    language="da",
    euroeval_version="18.0.0",
    image_digest="sha256:image",
    expires_at="2099-01-01T00:00:00Z",
)
GPU = Gpu("A100", "GPU-1", 10 * 1024**3, 20 * 1024**3, "8.0")
HARDWARE = HardwareReport("x86_64", 64, 100, "550", "12.4", "2.7", (GPU,))


class AuthClient:
    """Minimal device-flow broker fake."""

    def start_auth(self) -> AuthStart:
        """Return test device-flow details."""
        return AuthStart("session", "CODE", "https://example.test", 60, 0)

    def poll_auth(self, session_id: str) -> AuthPoll:
        """Approve the test device flow."""
        assert session_id == "session"
        return AuthPoll(False, "opaque-credential", "octocat")


def test_auth_persists_only_broker_auth_with_private_permissions(
    tmp_path: Path,
) -> None:
    """Persist the opaque credential and verified login mode 0600."""
    state = StateStore(tmp_path)
    assert authenticate(AuthClient(), state, sleep=lambda _seconds: None) == (
        "opaque-credential",
        "octocat",
    )
    assert json.loads(state.path.read_text()) == {
        "credential": "opaque-credential",
        "github_login": "octocat",
    }
    assert os.stat(state.path).st_mode & 0o777 == 0o600


def test_nvidia_csv_parser_handles_compute_capability_and_fails_without_gpu() -> None:
    """Parse memory and capability values, and fail closed without a GPU."""
    output = "NVIDIA A100, GPU-1, 10240, 20480, 8.0\n"
    gpu = discover_gpus(runner=lambda _command: output)[0]
    assert gpu.free_memory_bytes == 10 * 1024**3
    assert gpu.compute_capability == "8.0"
    with pytest.raises(NoGpuError):
        discover_gpus(runner=lambda _command: "")


def test_safety_rejects_remote_code_and_unpinned_models() -> None:
    """Reject remote code and mutable Hub revisions."""
    metadata = ModelMetadata(
        private=False,
        gated=False,
        auto_map=True,
        files=("config.json", "model.safetensors"),
        safetensors=True,
        estimated_bytes=1,
    )
    with pytest.raises(SafetyError, match="auto_map"):
        check_model_safety(LEASE, (GPU,), metadata)
    with pytest.raises(SafetyError, match="unpinned"):
        check_model_safety(
            dataclasses.replace(LEASE, model_revision="main"), (GPU,), metadata
        )


def test_evaluator_uses_validation_and_remote_code_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pass the worker's safety flags to the existing evaluator."""
    calls: dict[str, object] = {}

    class FakeBenchmarker:
        """Capture the adapter's calls."""

        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs

        def benchmark(self, **kwargs: object) -> list[object]:
            """Return one fake benchmark result."""
            calls["benchmark"] = kwargs
            return [object()]

    monkeypatch.setattr(evaluator, "Benchmarker", FakeBenchmarker)
    monkeypatch.setattr(
        evaluator,
        "benchmark_result_to_eee_dict",
        lambda result: {"evaluation_id": "one", "result": 1},
    )
    records = evaluator.EuroEvalEvaluator(tmp_path).evaluate(
        lease=LEASE, output_path=tmp_path / "isolated.jsonl"
    )
    assert records[0].sha256 == records[0].sha256
    assert calls["benchmark"] == {
        "model": f"org/model@{REVISION}",
        "language": "da",
        "progress_bar": False,
        "save_results": False,
        "trust_remote_code": False,
        "evaluate_test_split": False,
        "requires_safetensors": True,
        "gpu_memory_utilization": 0.8,
        "force": True,
        "raise_errors": True,
    }
    assert len((tmp_path / "isolated.jsonl").read_text().splitlines()) == 1


class Broker:
    """Broker fake covering claim, lease, and result lifecycle."""

    def __init__(self) -> None:
        """Initialise broker state."""
        self.claims = 0
        self.submissions = 0
        self.releases: list[str] = []
        self.finalised = False

    def start_auth(self) -> AuthStart:
        """Return test auth details."""
        return AuthStart("session", "CODE", "https://example.test", 60, 1)

    def poll_auth(self, session_id: str) -> AuthPoll:
        """Return approved test auth."""
        return AuthPoll(False, "cred", "login")

    def claim(self, credential: str, hardware: HardwareReport) -> Claim:
        """Return the test lease."""
        self.claims += 1
        return Claim(LEASE)

    def heartbeat(self, credential: str, lease_id: str) -> None:
        """Accept a test heartbeat."""

    def submit_result(self, credential: str, lease: Lease, result: EEERecord) -> None:
        """Fail once to verify digest-stable retry."""
        self.submissions += 1
        if self.submissions == 1:
            raise RuntimeError("temporary broker error")

    def finalise(self, credential: str, lease_id: str) -> None:
        """Accept finalisation."""
        self.finalised = True

    def release(self, credential: str, lease_id: str, reason: str) -> None:
        """Record lease release."""
        self.releases.append(reason)


class OneRecordEvaluator:
    """Evaluator fake producing one isolated result."""

    def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
        """Return one stable record."""
        return [EEERecord({"id": "one"}, "digest-one")]


def test_worker_retries_idempotently_and_finalises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Retry a result without changing its digest, then finalise."""
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(runtime, "check_model_safety", lambda lease, gpus: None)
    broker = Broker()
    worker = runtime.Worker(
        client=broker,
        state=StateStore(tmp_path),
        evaluator=OneRecordEvaluator(),
        hardware_factory=lambda: HARDWARE,
    )
    worker.run(once=True)
    assert broker.submissions == 2
    assert broker.finalised
    assert not broker.releases


def test_no_gpu_exits_before_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Do not claim work when hardware discovery reports no GPU."""
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    broker = Broker()
    worker = runtime.Worker(
        client=broker,
        state=StateStore(tmp_path),
        hardware_factory=lambda: (_ for _ in ()).throw(NoGpuError("no GPU")),
    )
    with pytest.raises(NoGpuError):
        worker.run(once=True)
    assert broker.claims == 0


def test_evaluation_failure_releases_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Release a lease when evaluation fails."""
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(runtime, "check_model_safety", lambda lease, gpus: None)
    broker = Broker()

    class FailingEvaluator:
        """Evaluator that fails before producing a record."""

        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            """Raise a representative evaluation failure."""
            raise RuntimeError("evaluation failed")

    worker = runtime.Worker(
        client=broker,
        state=StateStore(tmp_path),
        evaluator=FailingEvaluator(),
        hardware_factory=lambda: HARDWARE,
    )
    with pytest.raises(RuntimeError):
        worker.run(once=True)
    assert broker.releases == ["worker_interrupted_or_failed"]


def test_lease_loss_releases_without_finalising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A lost heartbeat never permits finalisation."""
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(runtime, "check_model_safety", lambda lease, gpus: None)

    class LostHeartbeat:
        """Heartbeat fake reporting lease loss."""

        def __init__(self, client: object, credential: str, lease: Lease) -> None:
            pass

        def start(self) -> None:
            """Start the fake heartbeat."""

        def check(self) -> None:
            """Report a lost lease."""
            raise runtime.LeaseLost("lost")

        def stop(self) -> None:
            """Stop the fake heartbeat."""

    monkeypatch.setattr(runtime, "Heartbeat", LostHeartbeat)
    broker = Broker()
    worker = runtime.Worker(
        client=broker,
        state=StateStore(tmp_path),
        evaluator=OneRecordEvaluator(),
        hardware_factory=lambda: HARDWARE,
    )
    with pytest.raises(runtime.LeaseLost):
        worker.run(once=True)
    assert broker.releases == ["worker_interrupted_or_failed"]
    assert not broker.finalised


def test_keyboard_interrupt_releases_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cancellation releases an active lease."""
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(runtime, "check_model_safety", lambda lease, gpus: None)
    broker = Broker()

    class InterruptedEvaluator:
        """Evaluator interrupted by Ctrl-C."""

        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            """Simulate Ctrl-C."""
            raise KeyboardInterrupt

    worker = runtime.Worker(
        client=broker,
        state=StateStore(tmp_path),
        evaluator=InterruptedEvaluator(),
        hardware_factory=lambda: HARDWARE,
    )
    with pytest.raises(KeyboardInterrupt):
        worker.run(once=True)
    assert broker.releases == ["worker_interrupted_or_failed"]
