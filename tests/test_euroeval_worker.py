"""Tests for the volunteer worker's broker-facing orchestration."""

import dataclasses
import json
import os
from pathlib import Path

import pytest

from euroeval_worker import evaluator, runtime
from euroeval_worker.auth import authenticate
from euroeval_worker.broker import BrokerClient, BrokerError
from euroeval_worker.hardware import NoGpuError, discover_gpus
from euroeval_worker.safety import ModelMetadata, SafetyError, check_model_safety
from euroeval_worker.state import PendingRecord, StateStore
from euroeval_worker.types import (
    AuthPoll,
    AuthStart,
    Claim,
    EEERecord,
    Gpu,
    HardwareReport,
    Lease,
    canonical_json,
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
    worker_version="worker-1",
    selected_gpu_uuid="GPU-1",
    selected_gpu_index=0,
)
GPU = Gpu("A100", "GPU-1", 10 * 1024**3, 20 * 1024**3, "8.0", 0)
HARDWARE = HardwareReport(
    "x86_64",
    64,
    100,
    "550",
    "12.4",
    "2.7",
    (GPU,),
    selected_gpu_index=GPU.index,
    selected_gpu_uuid=GPU.uuid,
)


class AuthClient:
    """Minimal device-flow broker fake."""

    def start_auth(self) -> AuthStart:
        """Return test device-flow details."""
        return AuthStart("session", "CODE", "https://example.test", 60, 0)

    def poll_auth(self, session_id: str) -> AuthPoll:
        """Approve the test device flow.

        Returns:
            Approved test credentials.
        """
        assert session_id == "session"
        return AuthPoll(False, "opaque-credential", "octocat")


def test_broker_protocol_payload_is_canonical() -> None:
    """The Python client emits the same flat v1 envelope as the broker."""
    calls: list[tuple[str, str, dict[str, object]]] = []

    def request(
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        """Capture a request and return a no-work response.

        Returns:
            The canonical no-work envelope.
        """
        assert payload is not None
        calls.append((method, url, payload))
        return {"protocol_version": "volunteer-worker/v1", "status": "no_work"}

    client = BrokerClient(
        "https://broker.test", request=request, worker_version="worker-1"
    )
    assert client.claim("credential", HARDWARE).lease is None
    assert calls[0][2]["protocol_version"] == "volunteer-worker/v1"
    assert calls[0][2]["worker_version"] == "worker-1"
    hardware_payload = calls[0][2]["hardware"]
    assert isinstance(hardware_payload, dict)
    assert set(hardware_payload) == {
        "architecture",
        "ram_bytes",
        "free_disk_bytes",
        "driver_version",
        "cuda_version",
        "pytorch_version",
        "gpu_memory_utilisation",
        "selected_gpu_index",
        "selected_gpu_uuid",
        "gpus",
    }
    assert hardware_payload["gpu_memory_utilisation"] == 0.8
    assert hardware_payload["selected_gpu_index"] == GPU.index
    assert hardware_payload["selected_gpu_uuid"] == GPU.uuid


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
    output = "1, NVIDIA A100, GPU-1, 10240, 20480, 8.0\n"
    gpu = discover_gpus(runner=lambda _command: output)[0]
    assert gpu.free_memory_bytes == 10 * 1024**3
    assert gpu.compute_capability == "8.0"
    assert gpu.index == 1
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

    def heartbeat(self, credential: str, lease_id: str) -> str:
        """Accept a test heartbeat.

        Returns:
            The current lease expiry.
        """
        return LEASE.expires_at

    def submit_result(self, credential: str, lease: Lease, result: EEERecord) -> None:
        """Fail once to verify digest-stable retry.

        Raises:
            RuntimeError: On the first submission.
        """
        self.submissions += 1
        if self.submissions == 1:
            raise RuntimeError("temporary broker error")

    def finalise(self, credential: str, lease_id: str) -> str:
        """Accept finalisation and return its stable identifier.

        Returns:
            The stable submission identifier.
        """
        self.finalised = True
        return "submission-1"

    def release(self, credential: str, lease_id: str, reason: str) -> None:
        """Record lease release."""
        self.releases.append(reason)


class OneRecordEvaluator:
    """Evaluator fake producing one isolated result."""

    def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
        """Return one stable record."""
        return [EEERecord({"id": "one"})]


def test_worker_retries_idempotently_and_finalises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Retry a result without changing its digest, then finalise."""
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )
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


def test_busy_gpu_is_not_selected_or_exposed_to_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Choose the free GPU and restore CUDA visibility after evaluation."""
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )
    busy = Gpu("A100", "GPU-0", 1, 10, "8.0", 0)
    free = Gpu("A100", "GPU-1", 9, 10, "8.0", 1)
    hardware = HardwareReport("x86_64", 64, 100, "550", "12.4", "2.7", (busy, free))
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    observed: dict[str, object] = {}

    class CapturingBroker(Broker):
        """Capture the selected claim hardware."""

        def claim(self, credential: str, hardware: HardwareReport) -> Claim:
            """Capture hardware and return work.

            Returns:
                The test lease.
            """
            observed["hardware"] = hardware
            return Claim(
                dataclasses.replace(
                    LEASE,
                    selected_gpu_uuid=hardware.selected_gpu_uuid,
                    selected_gpu_index=hardware.selected_gpu_index,
                )
            )

    class CapturingEvaluator:
        """Capture CUDA visibility during model setup."""

        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            """Return one result after observing the environment."""
            observed["cuda"] = os.environ.get("CUDA_VISIBLE_DEVICES")
            return [EEERecord({"id": "one"})]

    broker = CapturingBroker()
    runtime.Worker(
        client=broker,
        state=StateStore(tmp_path),
        evaluator=CapturingEvaluator(),
        hardware_factory=lambda: hardware,
    ).run(once=True)
    claimed = observed["hardware"]
    assert isinstance(claimed, HardwareReport)
    assert claimed.selected_gpu_index == 1
    assert claimed.selected_gpu_uuid == "GPU-1"
    assert observed["cuda"] == "GPU-1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == original


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
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )
    broker = Broker()

    class FailingEvaluator:
        """Evaluator that fails before producing a record."""

        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            """Raise a representative evaluation failure.

            Raises:
                RuntimeError: Always, to exercise release handling.
            """
            raise RuntimeError("evaluation failed")

    worker = runtime.Worker(
        client=broker,
        state=StateStore(tmp_path),
        evaluator=FailingEvaluator(),
        hardware_factory=lambda: HARDWARE,
    )
    with pytest.raises(RuntimeError):
        worker.run(once=True)
    assert broker.releases == []
    assert StateStore(tmp_path).load_active() is not None


def test_lease_loss_releases_without_finalising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A lost heartbeat never permits finalisation."""
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )

    class LostHeartbeat:
        """Heartbeat fake reporting lease loss."""

        def __init__(self, client: object, credential: str, lease: Lease) -> None:
            pass

        def start(self) -> None:
            """Start the fake heartbeat."""

        def check(self) -> None:
            """Report a lost lease.

            Raises:
                LeaseLost: Always, for this test double.
            """
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
    assert broker.releases == []
    assert not broker.finalised
    assert list((tmp_path / "archive").glob("*.json"))


def test_keyboard_interrupt_releases_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cancellation releases an active lease."""
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )
    broker = Broker()

    class InterruptedEvaluator:
        """Evaluator interrupted by Ctrl-C."""

        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            """Simulate Ctrl-C.

            Raises:
                KeyboardInterrupt: Always, for this test double.
            """
            raise KeyboardInterrupt

    worker = runtime.Worker(
        client=broker,
        state=StateStore(tmp_path),
        evaluator=InterruptedEvaluator(),
        hardware_factory=lambda: HARDWARE,
    )
    with pytest.raises(KeyboardInterrupt):
        worker.run(once=True)
    assert broker.releases == []
    assert StateStore(tmp_path).load_active() is not None


def test_broker_client_drives_canonical_http_lifecycle(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Drive authentication, lease, result and finalisation over fake HTTP."""
    calls: list[tuple[str, dict[str, object]]] = []

    def request(
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        """Return one canonical response for each broker endpoint."""
        assert method == "POST"
        assert payload is not None
        path = url.removeprefix("https://broker.test/")
        calls.append((path, payload))
        if path == "auth/start":
            return {
                "protocol_version": "volunteer-worker/v1",
                "session_id": "session",
                "user_code": "CODE",
                "verification_uri": "https://example.test",
                "expires_in": 60,
                "interval": 0,
            }
        if path == "auth/poll":
            return {
                "protocol_version": "volunteer-worker/v1",
                "status": "authorised",
                "credential": "secret-credential",
                "github_login": "volunteer",
            }
        if path == "claim":
            return {
                "protocol_version": "volunteer-worker/v1",
                "lease_id": LEASE.lease_id,
                "issue_number": LEASE.issue_number,
                "language": LEASE.language,
                "model_id": LEASE.model_id,
                "model_revision": LEASE.model_revision,
                "euroeval_version": LEASE.euroeval_version,
                "image_digest": LEASE.image_digest,
                "worker_version": LEASE.worker_version,
                "expires_at": LEASE.expires_at,
                "selected_gpu_uuid": LEASE.selected_gpu_uuid,
                "selected_gpu_index": LEASE.selected_gpu_index,
            }
        if path == "heartbeat":
            return {
                "protocol_version": "volunteer-worker/v1",
                "lease_id": LEASE.lease_id,
                "expires_at": LEASE.expires_at,
            }
        if path == "result":
            return {"protocol_version": "volunteer-worker/v1", "status": "uploaded"}
        return {
            "protocol_version": "volunteer-worker/v1",
            "status": "ready",
            "submission_id": "submission-42",
        }

    client = BrokerClient("https://broker.test", request=request)
    state = StateStore(tmp_path)
    credential, login = authenticate(client, state, sleep=lambda _delay: None)
    assert (credential, login) == ("secret-credential", "volunteer")
    assert client.claim(credential, HARDWARE).lease == LEASE
    client.heartbeat(credential, LEASE.lease_id)
    record = EEERecord(record_json=canonical_json({"one": 1.0, "tiny": 1e-7}))
    client.submit_result(credential, LEASE, record)
    assert client.finalise(credential, LEASE.lease_id) == "submission-42"
    result_payload = next(payload for path, payload in calls if path == "result")
    assert result_payload["record_json"] == record.record_json
    assert result_payload["digest"] == record.digest
    assert "secret-credential" not in json.dumps(calls)
    assert "secret-credential" not in caplog.text


def test_auth_honours_slow_down_retry_after(tmp_path: Path) -> None:
    """Use the broker's slow-down interval rather than a tight polling loop."""
    delays: list[float] = []

    class SlowAuth(AuthClient):
        """Device flow that first requests slower polling."""

        def __init__(self) -> None:
            self.polls = 0

        def poll_auth(self, session_id: str) -> AuthPoll:
            """Return slow-down and then approval."""
            self.polls += 1
            if self.polls == 1:
                return AuthPoll(True, retry_after=7)
            return AuthPoll(False, "credential", "volunteer")

    assert authenticate(SlowAuth(), StateStore(tmp_path), sleep=delays.append) == (
        "credential",
        "volunteer",
    )
    assert delays == [7]


def test_eee_record_rejects_non_finite_values() -> None:
    """Prevent NaN and Infinity from entering a digest-stable envelope."""
    with pytest.raises(ValueError):
        canonical_json({"value": float("nan")})
    with pytest.raises(ValueError):
        EEERecord(record_json='{"value": Infinity}')


def test_interruption_resumes_pre_evaluation_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Persist a lease before evaluation and resume it without reclaiming work."""
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )

    class Interrupted:
        """Evaluator interrupted before producing output."""

        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            """Interrupt the first worker.

            Raises:
                KeyboardInterrupt:
                    Always, to simulate SIGINT.
            """
            raise KeyboardInterrupt

    broker = Broker()
    with pytest.raises(KeyboardInterrupt):
        runtime.Worker(
            client=broker,
            state=StateStore(tmp_path),
            evaluator=Interrupted(),
            hardware_factory=lambda: HARDWARE,
        ).run(once=True)
    assert StateStore(tmp_path).load_active() is not None

    worker = runtime.Worker(
        client=broker,
        state=StateStore(tmp_path),
        evaluator=OneRecordEvaluator(),
        hardware_factory=lambda: HARDWARE,
    )
    worker.run(once=True)
    assert broker.claims == 1
    assert worker.last_submission_id == "submission-1"
    assert StateStore(tmp_path).load_active() is None


def test_partial_acknowledgements_survive_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Retry only unacknowledged records after a failed submission."""
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )

    class TwoRecords:
        """Evaluator producing two deterministic records."""

        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            """Return two records."""
            return [EEERecord({"id": "one"}), EEERecord({"id": "two"})]

    class FailSecond(Broker):
        """Broker that loses the connection on the second record."""

        def __init__(self) -> None:
            super().__init__()
            self.failures = 0

        def submit_result(
            self, credential: str, lease: Lease, result: EEERecord
        ) -> None:
            """Fail all retries for the second record on the first run.

            Raises:
                BrokerError:
                    When the second record is first submitted.
            """
            if result.record["id"] == "two" and self.failures < 3:
                self.failures += 1
                raise BrokerError("temporary", status=503, retry_after=1)
            super().submit_result(credential, lease, result)

    broker = FailSecond()
    worker = runtime.Worker(
        client=broker,
        state=StateStore(tmp_path),
        evaluator=TwoRecords(),
        hardware_factory=lambda: HARDWARE,
    )
    with pytest.raises(BrokerError):
        worker.run(once=True)
    active = StateStore(tmp_path).load_active()
    assert active is not None
    assert [record.acknowledged for record in active.records] == [True, False]
    submissions_before = broker.submissions
    worker.run(once=True)
    assert broker.submissions > submissions_before
    assert StateStore(tmp_path).load_active() is None


def test_restart_submits_durable_records_without_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resume durable results without regenerating timestamped evaluation bytes."""
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    state = StateStore(tmp_path)
    state.save_active(
        lease=LEASE,
        records=(PendingRecord.from_record(EEERecord({"timestamp": "original"})),),
        github_login="login",
    )

    class MustNotEvaluate:
        """Evaluator that proves restart bypasses evaluation."""

        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            """Fail if the durable result was not used.

            Raises:
                AssertionError:
                    If restart attempts to regenerate the durable result.
            """
            raise AssertionError("restart regenerated an already durable result")

    broker = Broker()
    runtime.Worker(
        client=broker,
        state=state,
        evaluator=MustNotEvaluate(),
        hardware_factory=lambda: HARDWARE,
    ).run(once=True)
    assert broker.finalised
    assert state.load_active() is None


def test_finalise_response_loss_is_idempotently_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep acknowledged results when the finalise response is lost."""
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )

    class LostFinalise(Broker):
        """Broker that loses one finalise response."""

        def finalise(self, credential: str, lease_id: str) -> str:
            """Lose the first response, then return the identifier.

            Returns:
                The submission identifier after the simulated lost response.

            Raises:
                BrokerError:
                    On the simulated lost response.
            """
            if self.finalised:
                return super().finalise(credential, lease_id)
            self.finalised = True
            raise BrokerError("response lost", status=None)

    broker = LostFinalise()
    state = StateStore(tmp_path)
    worker = runtime.Worker(
        client=broker,
        state=state,
        evaluator=OneRecordEvaluator(),
        hardware_factory=lambda: HARDWARE,
    )
    with pytest.raises(BrokerError):
        worker.run(once=True)
    assert state.load_active() is not None
    worker.run(once=True)
    assert state.load_active() is None


def test_cached_credential_is_reauthenticated_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Clear one revoked cached credential and complete device auth once."""
    authentications = iter([("old", "login"), ("new", "login")])
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: next(authentications)
    )

    class Revoked(Broker):
        """Broker rejecting only the first claim credential."""

        def claim(self, credential: str, hardware: HardwareReport) -> Claim:
            """Reject the cached credential once.

            Returns:
                The claimed lease for a valid credential.

            Raises:
                BrokerError:
                    When the cached credential is revoked.
            """
            if credential == "old":
                raise BrokerError("revoked", status=401)
            return super().claim(credential, hardware)

    broker = Revoked()
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )
    runtime.Worker(
        client=broker,
        state=StateStore(tmp_path),
        evaluator=OneRecordEvaluator(),
        hardware_factory=lambda: HARDWARE,
    ).run(once=True)
    assert broker.claims == 1


def test_expired_active_lease_is_archived_before_new_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Never submit stale pending state alongside a newly claimed lease."""
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )
    state = StateStore(tmp_path)
    state.save_active(
        dataclasses.replace(LEASE, expires_at="2000-01-01T00:00:00Z"),
        github_login="login",
    )
    broker = Broker()
    runtime.Worker(
        client=broker,
        state=state,
        evaluator=OneRecordEvaluator(),
        hardware_factory=lambda: HARDWARE,
    ).run(once=True)
    assert list((tmp_path / "archive").glob("*.json"))
    assert state.load_active() is None
