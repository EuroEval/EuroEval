"""Tests for the volunteer worker's broker-facing orchestration."""

import dataclasses
import email.message
import functools
import http.client
import importlib.metadata
import io
import json
import os
import threading
import typing as t
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from euroeval.benchmark_config_factory import prepare_dataset_configs
from euroeval.data_models import DatasetConfig
from euroeval.enums import ShotMode
from euroeval.languages import get_all_languages
from euroeval_worker import cli, evaluator, hardware, runtime
from euroeval_worker.auth import authenticate
from euroeval_worker.broker import BrokerClient, BrokerError
from euroeval_worker.hardware import (
    METAL_GPU_UUID,
    NoGpuError,
    discover_gpus,
    discover_hardware,
    unified_free_bytes,
)
from euroeval_worker.safety import ModelMetadata, SafetyError, check_model_safety
from euroeval_worker.state import PendingRecord, StateStore
from euroeval_worker.types import (
    AuthPoll,
    AuthStart,
    CanaryInstruction,
    Claim,
    EEERecord,
    ExpectedScope,
    Gpu,
    HardwareReport,
    Lease,
    ModelEvidence,
    canonical_json,
    lease_from_dict,
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
    model_type="encoder",
    worker_version="worker-1",
    selected_gpu_uuid="GPU-1",
    selected_gpu_index=0,
    model_metadata=ModelEvidence(
        pipeline_tag="fill-mask", architectures=("RobertaModel",), model_type="encoder"
    ),
    expected_scope=ExpectedScope(
        policy_version="test-policy",
        language_group="da",
        allowed_identity_suffix_sets=(('["test",false,true]',),),
        count=1,
        warnings=(),
        task_groups=("sequence_classification",),
    ),
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


class _MetalHostCalls(t.TypedDict):
    """Commands and subprocess calls recorded by the fake Apple Silicon host."""

    commands: list[list[str]]
    subprocess: list[tuple[list[str], dict[str, object]]]


def _fake_metal_host(
    monkeypatch: pytest.MonkeyPatch, cap_stdout: str, vm_stat: str
) -> "_MetalHostCalls":
    """Simulate an Apple Silicon host for hardware discovery.

    ``os.sysconf`` raises for ``SC_PHYS_PAGES`` as it does on macOS, so the test
    fails if discovery falls back to the Linux RAM reader.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        cap_stdout:
            Output of the MLX working-set-limit subprocess.
        vm_stat:
            Output of ``vm_stat``.

    Returns:
        The runner commands and the subprocess calls made during discovery.
    """
    calls: _MetalHostCalls = {"commands": [], "subprocess": []}
    monkeypatch.delenv("VLLM_TARGET_DEVICE", raising=False)
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    monkeypatch.setattr(hardware.sys, "platform", "darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware, "_pytorch_version", lambda: "2.13.0")

    def macos_sysconf(name: str) -> int:
        """Reject the configuration names macOS does not provide.

        Args:
            name:
                Requested configuration name.

        Raises:
            ValueError:
                Always, as macOS does for ``SC_PHYS_PAGES``.
        """
        raise ValueError(f"unrecognized configuration name: {name}")

    monkeypatch.setattr(hardware.os, "sysconf", macos_sysconf)

    def fake_run(command: list[str], **kwargs: object) -> object:
        """Record the cap subprocess and return its canned output.

        Args:
            command:
                Command supplied to the subprocess.
            **kwargs:
                Subprocess options supplied by discovery.

        Returns:
            A completed-process stand-in.
        """
        calls["subprocess"].append((command, kwargs))
        return type("Completed", (), {"stdout": cap_stdout})()

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)
    # A fresh probe cache per test, restored afterwards.
    monkeypatch.setattr(
        hardware, "_probe_metal", functools.cache(hardware._probe_metal.__wrapped__)
    )

    def runner(command: list[str]) -> str:
        """Answer the macOS commands used for discovery.

        Args:
            command:
                Command supplied to the test.

        Returns:
            Canned command output.

        Raises:
            AssertionError:
                If discovery runs an NVIDIA command on macOS.
        """
        calls["commands"].append(command)
        if command[0] == "nvidia-smi":
            raise AssertionError("nvidia-smi must not run on macOS")
        if command[0] == "vm_stat":
            return vm_stat
        if command == ["sysctl", "-n", "hw.memsize"]:
            return f"{64 * 1024**3}\n"
        return "Apple M3 Max\n"

    monkeypatch.setattr(hardware, "_run", runner)
    return calls


def _metal_probe(cap: int, **backends: bool) -> str:
    """Build the Apple GPU probe output, with every backend available by default.

    Args:
        cap:
            MLX's recommended working-set limit in bytes.
        **backends:
            Overrides for the ``metal``, ``vllm_metal_plugin`` and ``mps`` flags.

    Returns:
        The probe's JSON output.
    """
    flags = {"metal": True, "vllm_metal_plugin": True, "mps": True, **backends}
    return json.dumps({"cap": cap, **flags})


def _unsafe_first_lease(lease: Lease, **_kwargs: object) -> None:
    """Reject the first test lease as unsafe and accept any other lease.

    Args:
        lease:
            Lease under the safety check.
        **_kwargs:
            Ignored safety-check arguments.

    Raises:
        SafetyError:
            For the first test lease.
    """
    if lease.lease_id == LEASE.lease_id:
        raise SafetyError("model backend compatibility was not verified")


def test_auth_honours_slow_down_retry_after(tmp_path: Path) -> None:
    """Use the broker's slow-down interval rather than a tight polling loop.

    Args:
        tmp_path:
            Temporary directory for files created by the test.
    """
    delays: list[float] = []

    class SlowAuth(AuthClient):
        """Device flow that first requests slower polling."""

        def __init__(self) -> None:
            """Initialise the poll counter."""
            self.polls = 0

        def poll_auth(self, session_id: str) -> AuthPoll:
            """Return slow-down and then approval.

            Args:
                session_id:
                    Identifier of the device-flow session to poll.

            Returns:
                A slow-down response on the first poll, then approved credentials.
            """
            self.polls += 1
            if self.polls == 1:
                return AuthPoll(True, retry_after=7)
            return AuthPoll(False, "credential", "volunteer")

    assert authenticate(SlowAuth(), StateStore(tmp_path), sleep=delays.append) == (
        "credential",
        "volunteer",
    )
    assert delays == [7]


def test_auth_persists_only_broker_auth_with_private_permissions(
    tmp_path: Path,
) -> None:
    """Persist the opaque credential and verified login mode 0600.

    Args:
        tmp_path:
            Temporary directory for files created by the test.
    """
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


class AuthClient:
    """Minimal device-flow broker fake."""

    def poll_auth(self, session_id: str) -> AuthPoll:
        """Approve the test device flow.

        Args:
            session_id:
                Session id supplied to the test.

        Returns:
            Approved test credentials.
        """
        assert session_id == "session"
        return AuthPoll(False, "opaque-credential", "octocat")

    def start_auth(self) -> AuthStart:
        """Return test device-flow details.

        Returns:
            Device-flow session, user code, verification URL and polling intervals.
        """
        return AuthStart("session", "CODE", "https://example.test", 60, 0)


def test_broker_client_drives_canonical_http_lifecycle(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Drive authentication, lease, result and finalisation over fake HTTP.

    Args:
        tmp_path:
            Temporary directory for files created by the test.
        caplog:
            Fixture for capturing log records during the test.
    """
    calls: list[tuple[str, dict[str, object]]] = []

    def request(
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        """Return one canonical response for each broker endpoint.

        Args:
            method:
                Method supplied to the test.
            url:
                Url supplied to the test.
            headers:
                Headers supplied to the test.
            payload:
                Optional JSON request body sent to the broker.

        Returns:
            The matching fake broker response as a JSON object.
        """
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
            assert LEASE.model_metadata is not None
            assert LEASE.expected_scope is not None
            return {
                "protocol_version": "volunteer-worker/v1",
                "lease_id": LEASE.lease_id,
                "issue_number": LEASE.issue_number,
                "language": LEASE.language,
                "model_id": LEASE.model_id,
                "model_revision": LEASE.model_revision,
                "model_type": LEASE.model_type,
                "euroeval_version": LEASE.euroeval_version,
                "image_digest": LEASE.image_digest,
                "worker_version": LEASE.worker_version,
                "expires_at": LEASE.expires_at,
                "selected_gpu_uuid": LEASE.selected_gpu_uuid,
                "selected_gpu_index": LEASE.selected_gpu_index,
                "model_metadata": dataclasses.asdict(LEASE.model_metadata),
                "expected_scope": dataclasses.asdict(LEASE.expected_scope),
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


@pytest.mark.parametrize(
    "body", [b'{"code":"worker_update_required","credential":"secret"}', b"not json"]
)
def test_broker_error_preserves_optional_safe_metadata(
    monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    """Expose structured HTTP status without requiring valid JSON error bodies.

    Args:
        monkeypatch:
            Dependency replacement fixture.
        body:
            Simulated error response bytes.
    """

    def reject(request: urllib.request.Request, timeout: int) -> None:
        """Reject a claim with an optional structured error body.

        Args:
            request:
                HTTP request.
            timeout:
                Network timeout.

        Raises:
            urllib.error.HTTPError:
                Always, to simulate the broker rejection.
        """
        raise urllib.error.HTTPError(
            request.full_url, 426, "update", email.message.Message(), io.BytesIO(body)
        )

    monkeypatch.setattr(urllib.request, "urlopen", reject)
    client = BrokerClient(server="https://example.test")
    with pytest.raises(BrokerError) as caught:
        client.claim(credential="cred", hardware=HARDWARE)
    assert caught.value.status == 426
    assert caught.value.code == (
        "worker_update_required" if body.startswith(b"{") else None
    )
    assert "secret" not in str(caught.value.body)
    assert BrokerError("legacy fake error").status is None
    assert BrokerError("legacy fake error").code is None


def test_broker_normalises_legacy_scope_and_rejects_malformed_alternatives() -> None:
    """Legacy scopes become one set while ambiguous alternatives fail closed."""
    legacy = dataclasses.asdict(LEASE)
    legacy["protocol_version"] = "volunteer-worker/v1"
    legacy["contamination_canary"] = dataclasses.asdict(
        CanaryInstruction(
            status="required",
            protocol_version="canary/v1",
            corpus_revision="revision",
            corpus_sha256="a" * 64,
        )
    )
    legacy["expected_scope"] = {
        "policy_version": "test-policy",
        "language_group": "da",
        "identity_suffixes": ['["test",false,true]'],
        "count": 1,
        "warnings": [],
        "task_groups": ["sequence_classification"],
    }
    decoded = lease_from_dict(data=legacy)
    assert decoded.expected_scope is not None
    assert decoded.expected_scope.allowed_identity_suffix_sets == (
        ('["test",false,true]',),
    )
    assert decoded.contamination_canary == CanaryInstruction(
        status="required",
        protocol_version="canary/v1",
        corpus_revision="revision",
        corpus_sha256="a" * 64,
    )
    malformed = dataclasses.asdict(LEASE)
    malformed["protocol_version"] = "volunteer-worker/v1"
    malformed["expected_scope"] = {
        "policy_version": "test-policy",
        "language_group": "da",
        "allowed_identity_suffix_sets": [
            ['["test",false,true]'],
            ['["test",false,true]'],
        ],
        "warnings": [],
        "task_groups": ["sequence_classification"],
    }
    with pytest.raises(ValueError, match="alternatives"):
        lease_from_dict(data=malformed)


def test_broker_normalises_truncated_http_response_as_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose truncated response transport failures as broker errors.

    Args:
        monkeypatch:
            Fixture replacing the HTTP transport.
    """

    def truncated_response(request: urllib.request.Request, timeout: int) -> None:
        raise http.client.IncompleteRead(b"partial")

    monkeypatch.setattr(urllib.request, "urlopen", truncated_response)
    client = BrokerClient("https://broker.example")

    with pytest.raises(BrokerError) as captured:
        client.claim(credential="credential", hardware=HARDWARE)

    assert captured.value.status is None


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

        Args:
            method:
                Method supplied to the test.
            url:
                Url supplied to the test.
            headers:
                Headers supplied to the test.
            payload:
                Payload supplied to the test.

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
    assert calls[0][2]["capabilities"] == ["canonical_coverage_v1"]
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


def test_busy_gpu_is_not_selected_or_exposed_to_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Choose the free GPU and restore CUDA visibility after evaluation.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
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

            Args:
                credential:
                    Credential supplied to the test.
                hardware:
                    Hardware supplied to the test.

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
            """Return one result after observing the environment.

            Args:
                lease:
                    Lease supplied to the test.
                output_path:
                    Path where evaluation results would be written.

            Returns:
                A single deterministic evaluation record.
            """
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


def test_cached_credential_is_reauthenticated_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Clear one revoked cached credential and complete device auth once.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    authentications = iter([("old", "login"), ("new", "login")])
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: next(authentications)
    )

    class Revoked(Broker):
        """Broker rejecting only the first claim credential."""

        def claim(self, credential: str, hardware: HardwareReport) -> Claim:
            """Reject the cached credential once.

            Args:
                credential:
                    Credential supplied to the test.
                hardware:
                    Hardware supplied to the test.

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


class OneRecordEvaluator:
    """Evaluator fake producing one isolated result."""

    def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
        """Return one stable record.

        Args:
            lease:
                Lease supplied to the test.
            output_path:
                Path where evaluation results would be written.

        Returns:
            A single deterministic evaluation record.
        """
        return [EEERecord({"id": "one"})]


def test_canary_corpus_outage_does_not_block_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finalise ordinary results when private corpus delivery is unavailable.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )
    lease = dataclasses.replace(
        LEASE,
        model_type="generative",
        model_metadata=ModelEvidence(
            pipeline_tag="text-generation",
            architectures=("LlamaForCausalLM",),
            model_type="generative",
        ),
        contamination_canary=CanaryInstruction(
            status="required",
            protocol_version="private-completion-canary/v1",
            corpus_revision="revision",
            corpus_sha256="a" * 64,
        ),
    )

    class OutageBroker(Broker):
        """Broker fake that fails to deliver the private canary corpus."""

        def claim(self, credential: str, hardware: HardwareReport) -> Claim:
            """Claim for this test.

            Args:
                credential:
                    Credential supplied to the test.
                hardware:
                    Hardware supplied to the test.

            Returns:
                A claim containing the test lease.
            """
            return Claim(lease)

        def fetch_canary_corpus(self, credential: str, lease: Lease) -> str:
            """Fetch canary corpus for this test.

            Args:
                credential:
                    Credential supplied to the test.
                lease:
                    Lease whose canary corpus is requested.

            Raises:
                BrokerError:
                    If the simulated operation fails.
            """
            raise BrokerError("temporarily unavailable", status=503)

        def submit_result(
            self, credential: str, lease: Lease, result: EEERecord
        ) -> None:
            """Submit result for this test.

            Args:
                credential:
                    Credential supplied to the test.
                lease:
                    Lease supplied to the test.
                result:
                    Result supplied to the test.
            """
            self.submissions += 1

    broker = OutageBroker()
    runtime.Worker(
        client=broker,
        state=StateStore(tmp_path),
        evaluator=OneRecordEvaluator(),
        hardware_factory=lambda: HARDWARE,
    ).run(once=True)

    assert broker.submissions == 1
    assert broker.finalised
    assert not broker.releases


@pytest.mark.parametrize("status", [0, 10, 75, 130, 1])
def test_cli_propagates_worker_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int
) -> None:
    """Pass the actual managed result through the CLI.

    Args:
        monkeypatch:
            Dependency replacement fixture.
        tmp_path:
            State directory.
        status:
            Worker result to propagate.
    """
    monkeypatch.setattr(
        runtime.Worker, "run", lambda self, once, launcher_managed: status
    )
    assert (
        cli.main(
            [
                "--launcher-managed",
                "--image-digest",
                LEASE.image_digest,
                "--state-dir",
                str(tmp_path),
            ]
        )
        == status
    )


def test_continuous_worker_recovers_after_submission_retries_from_saved_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resume exact pending records after bounded upload retries are exhausted.

    Args:
        monkeypatch:
            Fixture replacing network dependencies and sleep.
        tmp_path:
            Private worker state directory.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    first = PendingRecord.from_record(EEERecord({"id": "acknowledged"}))
    second = PendingRecord.from_record(EEERecord({"id": "pending"}))
    first = dataclasses.replace(first, acknowledged=True)
    state = StateStore(tmp_path)
    state.save_active(lease=LEASE, records=(first, second), github_login="login")

    class FlakyUploadBroker(Broker):
        """Transiently reject the pending record before accepting its retry."""

        def __init__(self) -> None:
            super().__init__()
            self.uploaded: list[EEERecord] = []

        def claim(self, credential: str, hardware: HardwareReport) -> Claim:
            """Report no new work after the saved lease is finalised.

            Args:
                credential:
                    Credential supplied to the test.
                hardware:
                    Hardware report used to request work.

            Returns:
                An empty claim so the test reaches its controlled shutdown.
            """
            self.claims += 1
            return Claim(None)

        def submit_result(
            self, credential: str, lease: Lease, result: EEERecord
        ) -> None:
            self.submissions += 1
            if self.submissions <= 3:
                raise BrokerError(
                    "transient result failure",
                    status=(408, 425, 425)[self.submissions - 1],
                )
            self.uploaded.append(result)

    class MustNotEvaluate:
        """Ensure restart uses its persisted records."""

        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            raise AssertionError("saved results must not be reevaluated")

    broker = FlakyUploadBroker()

    def stop_after_finalisation(delay: float) -> None:
        if broker.finalised:
            raise KeyboardInterrupt

    monkeypatch.setattr(runtime.time, "sleep", stop_after_finalisation)
    with pytest.raises(KeyboardInterrupt):
        runtime.Worker(
            client=broker,
            state=state,
            evaluator=MustNotEvaluate(),
            hardware_factory=lambda: HARDWARE,
        ).run()

    assert broker.submissions == 4
    assert broker.uploaded == [second.to_record()]
    assert broker.finalised
    assert state.load_active() is None


def test_continuous_worker_retries_evaluation_and_completes_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Continue after a failed evaluation and process subsequent work.

    Args:
        monkeypatch:
            Fixture replacing network dependencies and the delay function.
        tmp_path:
            Private worker state directory.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )
    delay_calls: list[float] = []

    def stop_after_completion(delay: float) -> None:
        """Stop after work is finalised and the broker reports no more work.

        Raises:
            KeyboardInterrupt:
                To terminate the test worker after successful work.
        """
        delay_calls.append(delay)
        if broker.finalised:
            raise KeyboardInterrupt

    monkeypatch.setattr(runtime.time, "sleep", stop_after_completion)

    class OneFailureEvaluator:
        """Fail once, then provide a result on the retry."""

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            """Return a result after the first simulated engine failure.

            Raises:
                RuntimeError:
                    On the first evaluator invocation.
            """
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("engine initialisation failed")
            return [EEERecord({"id": "recovered"})]

    class OneLeaseBroker(Broker):
        """Offer one lease, then report no additional work."""

        def claim(self, credential: str, hardware: HardwareReport) -> Claim:
            """Return the lease once and then no work."""
            self.claims += 1
            return Claim(LEASE if self.claims == 1 else None)

    broker = OneLeaseBroker()
    broker.submissions = 1
    evaluator_instance = OneFailureEvaluator()
    with pytest.raises(KeyboardInterrupt):
        runtime.Worker(
            client=broker,
            state=StateStore(tmp_path),
            evaluator=evaluator_instance,
            hardware_factory=lambda: HARDWARE,
        ).run()

    assert evaluator_instance.calls == 2
    assert broker.finalised
    assert len(delay_calls) >= 2
    assert StateStore(tmp_path).load_active() is None


def test_continuous_worker_retries_exhausted_release_and_claims_new_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Retry exhausted-work release, then process a distinct lease.

    Args:
        monkeypatch:
            Fixture replacing network dependencies and sleep.
        tmp_path:
            Private worker state directory.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )

    second_lease = dataclasses.replace(LEASE, lease_id="lease-2", issue_number=43)

    class RetryReleaseBroker(Broker):
        """Fail one release temporarily and offer a second distinct lease."""

        def claim(self, credential: str, hardware: HardwareReport) -> Claim:
            self.claims += 1
            return Claim(
                LEASE
                if self.claims == 1
                else second_lease
                if self.claims == 2
                else None
            )

        def release(self, credential: str, lease_id: str, reason: str) -> None:
            if not self.releases:
                self.releases.append("temporary failure")
                raise BrokerError("temporarily busy", status=503)
            self.releases.append(reason)

    class FirstLeaseFails:
        """Exhaust the first lease and successfully evaluate the next one."""

        calls = 0

        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            self.calls += 1
            if lease.lease_id == LEASE.lease_id:
                raise RuntimeError("evaluation unavailable")
            return [EEERecord({"id": "second-lease"})]

    broker = RetryReleaseBroker()
    evaluator_instance = FirstLeaseFails()

    def stop_after_finalisation(delay: float) -> None:
        if broker.finalised:
            raise KeyboardInterrupt

    monkeypatch.setattr(runtime.time, "sleep", stop_after_finalisation)
    state = StateStore(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        runtime.Worker(
            client=broker,
            state=state,
            evaluator=evaluator_instance,
            hardware_factory=lambda: HARDWARE,
        ).run()

    assert evaluator_instance.calls == 4
    assert broker.releases == ["temporary failure", "evaluation_failed"]
    assert broker.finalised
    assert broker.claims >= 3
    assert state.load_active() is None
    archived = list(state.archive_dir.glob("active-lease-*.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text())["evaluation_failures"] == 3


def test_continuous_worker_skips_unsafe_model_and_claims_new_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Release a lease that fails the safety check, then evaluate the next lease.

    Args:
        monkeypatch:
            Fixture replacing network dependencies and sleep.
        tmp_path:
            Private worker state directory.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(runtime, "check_model_safety", _unsafe_first_lease)
    broker = _second_lease_broker()
    evaluated: list[str] = []

    class RecordingEvaluator:
        """Record which leases reach evaluation."""

        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            """Record the lease and return one result.

            Args:
                lease:
                    Lease supplied to the test.
                output_path:
                    Path where evaluation results would be written.

            Returns:
                A single deterministic evaluation record.
            """
            evaluated.append(lease.lease_id)
            return [EEERecord(record_json={"id": lease.lease_id})]

    def stop_after_finalisation(delay: float) -> None:
        """Stop the continuous loop once the second lease is finalised.

        Args:
            delay:
                Requested sleep duration.

        Raises:
            KeyboardInterrupt:
                Once the broker has finalised work.
        """
        if broker.finalised:
            raise KeyboardInterrupt

    monkeypatch.setattr(runtime.time, "sleep", stop_after_finalisation)
    state = StateStore(directory=tmp_path)
    with pytest.raises(KeyboardInterrupt):
        runtime.Worker(
            client=broker,
            state=state,
            evaluator=RecordingEvaluator(),
            hardware_factory=lambda: HARDWARE,
        ).run()
    assert broker.releases == ["unsafe_model"]
    assert evaluated == ["lease-2"]
    assert broker.finalised
    assert state.load_active() is None


def _second_lease_broker() -> "Broker":
    """Build a broker offering the first lease, a distinct second lease, then nothing.

    Returns:
        The broker fake.
    """
    second = dataclasses.replace(LEASE, lease_id="lease-2", issue_number=43)

    class SecondLeaseBroker(Broker):
        """Offer two distinct leases in sequence."""

        def claim(self, credential: str, hardware: HardwareReport) -> Claim:
            """Offer the next lease in sequence.

            Args:
                credential:
                    Credential supplied to the test.
                hardware:
                    Hardware supplied to the test.

            Returns:
                The next lease, or no lease once both were offered.
            """
            self.claims += 1
            return Claim(
                lease=LEASE
                if self.claims == 1
                else second
                if self.claims == 2
                else None
            )

    return SecondLeaseBroker()


def test_discover_hardware_does_not_import_torch_in_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report GB10 data while avoiding CUDA initialisation in the worker process.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    # Exercise the NVIDIA path on any runner, including a Mac.
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(
        hardware,
        "_discover_spark_memory",
        lambda gpu: dataclasses.replace(
            gpu, free_memory_bytes=10, total_memory_bytes=20
        ),
    )
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: 128 * 1024**3)
    monkeypatch.setattr(hardware, "_pytorch_version", lambda: None)
    gpu_output = "0, NVIDIA GB10, GPU-test, 0, 0, 12.1\n"
    report = discover_hardware(
        runner=lambda command: (
            gpu_output
            if any(argument.startswith("--query-gpu=") for argument in command)
            else "Driver Version: 1.0 CUDA Version: 12.1"
        )
    )
    assert report.pytorch_version is None
    assert report.selected_gpu_uuid == "GPU-test"


def test_discover_hardware_uses_supplied_state_filesystem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Measure free space on the supplied cache rather than the working directory.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    state_dir = tmp_path / "mounted-cache"
    state_dir.mkdir()
    measured: list[Path] = []
    disk_usage = hardware.shutil.disk_usage
    # Exercise the NVIDIA path on any runner, including a Mac.
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: 128 * 1024**3)
    monkeypatch.setattr(hardware, "_pytorch_version", lambda: None)

    def record_disk_path(path: Path) -> object:
        """Record disk path for this test.

        Args:
            path:
                Path supplied to the test.

        Returns:
            The value returned by the test helper.
        """
        measured.append(path)
        return disk_usage(path)

    monkeypatch.setattr(hardware.shutil, "disk_usage", record_disk_path)
    gpu_output = "0, A100, GPU-test, 100, 200, 8.0\n"
    discover_hardware(
        runner=lambda command: (
            gpu_output
            if any(argument.startswith("--query-gpu=") for argument in command)
            else "Driver Version: 1.0 CUDA Version: 12.1"
        ),
        state_dir=state_dir,
    )
    assert measured == [state_dir]


def test_eee_record_rejects_non_finite_values() -> None:
    """Prevent NaN and Infinity from entering a digest-stable envelope."""
    with pytest.raises(ValueError):
        canonical_json({"value": float("nan")})
    with pytest.raises(ValueError):
        EEERecord(record_json='{"value": Infinity}')


def test_encoder_canary_uses_ordinary_result_path_without_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Submit the encoder's auxiliary record without delivering canary prompts.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )
    instruction = CanaryInstruction(
        status="required",
        protocol_version="private-completion-canary/v1",
        corpus_revision="revision",
        corpus_sha256="a" * 64,
        reason="encoder",
    )
    lease = dataclasses.replace(LEASE, contamination_canary=instruction)

    class EncoderEvaluator:
        """Return an ordinary result and an encoder canary placeholder."""

        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            """Evaluate for this test.

            Args:
                lease:
                    Lease supplied to the test.
                output_path:
                    Output path supplied to the test.

            Returns:
                The value returned by the test helper.
            """
            return [
                EEERecord({"id": "ordinary"}),
                EEERecord({"id": "canary", "status": "not_applicable"}),
            ]

    class EncoderBroker(Broker):
        """Broker fake that provides an encoder lease and accepts its results."""

        def claim(self, credential: str, hardware: HardwareReport) -> Claim:
            """Claim for this test.

            Args:
                credential:
                    Credential supplied to the test.
                hardware:
                    Hardware supplied to the test.

            Returns:
                A claim containing the test lease.
            """
            return Claim(lease)

        def fetch_canary_corpus(self, credential: str, lease: Lease) -> str:
            """Fetch canary corpus for this test.

            Args:
                credential:
                    Credential supplied to the test.
                lease:
                    Encoder lease for which corpus delivery is forbidden.

            Raises:
                AssertionError:
                    If the simulated operation fails.
            """
            raise AssertionError("encoder must not receive the canary corpus")

        def submit_result(
            self, credential: str, lease: Lease, result: EEERecord
        ) -> None:
            """Submit result for this test.

            Args:
                credential:
                    Credential supplied to the test.
                lease:
                    Lease supplied to the test.
                result:
                    Result supplied to the test.
            """
            self.submissions += 1

    broker = EncoderBroker()
    runtime.Worker(
        client=broker,
        state=StateStore(tmp_path),
        evaluator=EncoderEvaluator(),
        hardware_factory=lambda: HARDWARE,
    ).run(once=True)

    assert broker.submissions == 2
    assert broker.finalised


def test_evaluation_failure_keeps_active_lease_for_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Preserve the lease and propagate evaluation errors in once mode.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
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

            Args:
                lease:
                    Lease supplied to the test.
                output_path:
                    Path where evaluation results would be written.

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
    with pytest.raises(RuntimeError, match="evaluation failed"):
        worker.run(once=True)
    assert broker.releases == []
    active = StateStore(tmp_path).load_active()
    assert active is not None
    assert active.evaluation_failures == 0
    assert not active.release_pending


class Broker:
    """Broker fake covering claim, lease, and result lifecycle."""

    def __init__(self) -> None:
        """Initialise broker state."""
        self.claims = 0
        self.submissions = 0
        self.releases: list[str] = []
        self.finalised = False

    def claim(self, credential: str, hardware: HardwareReport) -> Claim:
        """Return the test lease.

        Args:
            credential:
                Credential supplied to the test.
            hardware:
                Hardware report used to request work.

        Returns:
            A claim containing the module-level test lease.
        """
        self.claims += 1
        return Claim(LEASE)

    def finalise(self, credential: str, lease_id: str) -> str:
        """Accept finalisation and return its stable identifier.

        Args:
            credential:
                Credential supplied to the test.
            lease_id:
                Lease id supplied to the test.

        Returns:
            The stable submission identifier.
        """
        self.finalised = True
        return "submission-1"

    def heartbeat(self, credential: str, lease_id: str) -> str:
        """Accept a test heartbeat.

        Args:
            credential:
                Credential supplied to the test.
            lease_id:
                Lease id supplied to the test.

        Returns:
            The current lease expiry.
        """
        return LEASE.expires_at

    def poll_auth(self, session_id: str) -> AuthPoll:
        """Return approved test auth.

        Args:
            session_id:
                Identifier of the device-flow session to poll.

        Returns:
            An approved authentication response for the test account.
        """
        return AuthPoll(False, "cred", "login")

    def release(self, credential: str, lease_id: str, reason: str) -> None:
        """Record lease release.

        Args:
            credential:
                Credential supplied to the test.
            lease_id:
                Lease id supplied to the test.
            reason:
                Reason supplied to the test.
        """
        self.releases.append(reason)

    def start_auth(self) -> AuthStart:
        """Return test auth details.

        Returns:
            Device-flow session, user code, verification URL and polling intervals.
        """
        return AuthStart("session", "CODE", "https://example.test", 60, 1)

    def submit_result(self, credential: str, lease: Lease, result: EEERecord) -> None:
        """Fail once to verify digest-stable retry.

        Args:
            credential:
                Credential supplied to the test.
            lease:
                Lease supplied to the test.
            result:
                Result supplied to the test.

        Raises:
            RuntimeError: On the first submission.
        """
        self.submissions += 1
        if self.submissions == 1:
            raise RuntimeError("temporary broker error")


def test_evaluator_runs_only_missing_scope_identities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Run only the remaining dataset, split and shot combination.

    Args:
        monkeypatch:
            Fixture for replacing evaluator dependencies.
        tmp_path:
            Temporary directory for isolated output.
    """
    calls: list[dict[str, object]] = []
    dataset = type("Dataset", (), {"name": "remaining"})()

    class FakeBenchmarker:
        """Capture per-identity benchmark requests."""

        def __init__(self, **kwargs: object) -> None:
            """Accept benchmark configuration.

            Args:
                kwargs:
                    Configured evaluator parameters.
            """

        def benchmark(self, **kwargs: object) -> list[object]:
            """Record the scoped benchmark invocation.

            Args:
                kwargs:
                    Benchmark invocation parameters.

            Returns:
                A single placeholder benchmark result.
            """
            calls.append(kwargs)
            return [object()]

    scope = ExpectedScope(
        policy_version="test-policy",
        language_group="da",
        allowed_identity_suffix_sets=(('["remaining",true,true]',),),
        count=None,
        warnings=(),
        task_groups=("sequence_classification",),
        original_identity_suffix_sets=(
            ('["covered",true,true]', '["remaining",true,true]'),
        ),
        covered_identity_suffixes=('["covered",true,true]',),
    )
    lease = dataclasses.replace(LEASE, expected_scope=scope)
    monkeypatch.setattr(evaluator, "Benchmarker", FakeBenchmarker)
    monkeypatch.setattr(
        evaluator, "_official_dataset_configs", lambda language: [dataset]
    )
    monkeypatch.setattr(
        evaluator,
        "benchmark_result_to_eee_dict",
        lambda result: {"evaluation_id": "remaining", "result": 1},
    )

    records = evaluator.EuroEvalEvaluator(tmp_path).evaluate(
        lease=lease, output_path=tmp_path / "narrowed.jsonl"
    )

    assert len(records) == 1
    assert len(calls) == 1
    assert calls[0]["dataset"] == [dataset]
    assert calls[0]["few_shot"] is ShotMode.FEW_SHOT
    assert calls[0]["evaluate_test_split"] is False


def test_evaluator_uses_validation_and_remote_code_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pass the worker's safety flags to the existing evaluator.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    calls: dict[str, object] = {}

    class FakeBenchmarker:
        """Capture the adapter's calls."""

        def __init__(self, **kwargs: object) -> None:
            """Init   for this test.

            Args:
                kwargs:
                    Kwargs supplied to the test.
            """
            calls["init"] = kwargs

        def benchmark(self, **kwargs: object) -> list[object]:
            """Return one fake benchmark result.

            Args:
                kwargs:
                    Benchmark options supplied by the evaluator.

            Returns:
                A list containing one placeholder benchmark result.
            """
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
    benchmark_kwargs = calls["benchmark"]
    assert isinstance(benchmark_kwargs, dict)
    assert benchmark_kwargs == {
        "model": f"org/model@{REVISION}",
        "language": "da",
        "progress_bar": False,
        "save_results": False,
        "task": None,
        "dataset": benchmark_kwargs["dataset"],
        "trust_remote_code": False,
        "evaluate_test_split": False,
        "requires_safetensors": True,
        "gpu_memory_utilization": 0.8,
        "few_shot": ShotMode.AUTO,
        "force": True,
        "raise_errors": True,
    }
    assert benchmark_kwargs["dataset"]
    assert all(
        config.task.name != "contamination-detection"
        for config in benchmark_kwargs["dataset"]
    )
    assert len((tmp_path / "isolated.jsonl").read_text().splitlines()) == 1


def test_expired_active_lease_is_archived_before_new_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Never submit stale pending state alongside a newly claimed lease.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
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


def test_failure_recording_preserves_latest_renewal_and_acknowledgements(
    tmp_path: Path,
) -> None:
    """Record failure metadata without replacing newer lease or result state.

    Args:
        tmp_path:
            Private worker state directory.
    """
    state = StateStore(tmp_path)
    result = EEERecord({"id": "durable"})
    pending = PendingRecord.from_record(result)
    state.save_active(lease=LEASE, records=(pending,))
    renewed = dataclasses.replace(LEASE, expires_at="2099-01-02T00:00:00Z")
    state.renew_active(renewed)
    state.save_records((dataclasses.replace(pending, acknowledged=True),))

    active = state.record_evaluation_failure(lease_id=LEASE.lease_id, max_failures=3)

    assert active.lease == renewed
    assert active.records[0].acknowledged
    assert active.evaluation_failures == 1
    assert not active.release_pending
    assert state.load_active() == active


def test_finalise_response_loss_is_idempotently_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep acknowledged results when the finalise response is lost.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
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

            Args:
                credential:
                    Credential supplied to the test.
                lease_id:
                    Lease id supplied to the test.

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


@pytest.mark.parametrize("evaluation_fails", [False, True])
def test_metal_gpu_is_not_cuda_pinned_during_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evaluation_fails: bool
) -> None:
    """Leave CUDA visibility untouched for the Apple GPU, even when evaluation fails.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
        evaluation_fails:
            Whether the evaluator raises during the lease.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-preexisting")
    metal = Gpu(
        name="Apple M3 Max",
        uuid=METAL_GPU_UUID,
        free_memory_bytes=9,
        total_memory_bytes=10,
        compute_capability=None,
        index=0,
    )
    hardware = HardwareReport(
        architecture="arm64",
        ram_bytes=64,
        free_disk_bytes=100,
        driver_version=None,
        cuda_version=None,
        pytorch_version="2.13",
        gpus=(metal,),
    )
    observed: dict[str, object] = {}

    class MetalBroker(Broker):
        """Lease the Apple GPU selected in the claim."""

        def claim(self, credential: str, hardware: HardwareReport) -> Claim:
            """Return work pinned to the claimed GPU.

            Args:
                credential:
                    Credential supplied to the test.
                hardware:
                    Hardware supplied to the test.

            Returns:
                The test lease.
            """
            return Claim(
                lease=dataclasses.replace(
                    LEASE,
                    selected_gpu_uuid=hardware.selected_gpu_uuid,
                    selected_gpu_index=hardware.selected_gpu_index,
                )
            )

    class CapturingEvaluator:
        """Capture CUDA visibility during evaluation."""

        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            """Return one result, or fail, after observing the environment.

            Args:
                lease:
                    Lease supplied to the test.
                output_path:
                    Path where evaluation results would be written.

            Returns:
                A single deterministic evaluation record.

            Raises:
                RuntimeError:
                    When the test asks the evaluation to fail.
            """
            observed["cuda"] = os.environ.get("CUDA_VISIBLE_DEVICES")
            if evaluation_fails:
                raise RuntimeError("evaluation failed")
            return [EEERecord(record_json={"id": "one"})]

    worker = runtime.Worker(
        client=MetalBroker(),
        state=StateStore(directory=tmp_path),
        evaluator=CapturingEvaluator(),
        hardware_factory=lambda: hardware,
    )
    if evaluation_fails:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            worker.run(once=True)
    else:
        worker.run(once=True)
    assert observed["cuda"] == "GPU-preexisting"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-preexisting"


def test_repeated_unsafe_work_backs_off_until_a_lease_gets_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Escalate the backoff when the broker keeps offering the same unsafe work.

    A successful claim resets the shared recovery backoff, so the safety skips must
    keep their own count. Otherwise the worker would retry every 30 seconds.

    Args:
        monkeypatch:
            Fixture replacing network dependencies and sleep.
        tmp_path:
            Private worker state directory.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(runtime, "check_model_safety", _unsafe_first_lease)
    second = dataclasses.replace(LEASE, lease_id="lease-2", issue_number=43)

    class RepeatingBroker(Broker):
        """Offer the unsafe lease six times, a runnable lease, then unsafe once more."""

        def claim(self, credential: str, hardware: HardwareReport) -> Claim:
            """Offer the next lease in sequence.

            Args:
                credential:
                    Credential supplied to the test.
                hardware:
                    Hardware supplied to the test.

            Returns:
                The next lease in the sequence, then nothing.
            """
            self.claims += 1
            if self.claims == 7:
                return Claim(lease=second)
            return Claim(lease=LEASE if self.claims <= 8 else None)

    broker = RepeatingBroker()
    delays: list[float] = []

    def record_sleep(delay: float) -> None:
        """Record each sleep and stop after the skip that follows the runnable lease.

        Args:
            delay:
                Requested sleep duration.

        Raises:
            KeyboardInterrupt:
                Once a skip after the finalised lease has slept.
        """
        delays.append(delay)
        if broker.finalised and broker.claims >= 8:
            raise KeyboardInterrupt

    monkeypatch.setattr(runtime.time, "sleep", record_sleep)
    worker = runtime.Worker(
        client=broker,
        state=StateStore(directory=tmp_path),
        evaluator=OneRecordEvaluator(),
        hardware_factory=lambda: HARDWARE,
    )
    with pytest.raises(KeyboardInterrupt):
        worker.run()
    backoffs = [delay for delay in delays if delay >= 30]
    assert backoffs == [30.0, 60.0, 120.0, 240.0, 300.0, 300.0, 30.0]
    assert broker.releases == ["unsafe_model"] * 7
    assert broker.finalised


def test_resumed_lease_success_resets_the_safety_skip_backoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Restart the skip backoff at 30 seconds after a resumed lease finishes.

    The runnable lease first fails to finalise, so it is finished on the resumed
    path. The next unsafe lease must then wait 30 seconds again.

    Args:
        monkeypatch:
            Fixture replacing network dependencies and sleep.
        tmp_path:
            Private worker state directory.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(runtime, "check_model_safety", _unsafe_first_lease)
    second = dataclasses.replace(LEASE, lease_id="lease-2", issue_number=43)

    class ResumingBroker(Broker):
        """Offer two unsafe leases, a runnable one, then unsafe again."""

        finalise_attempts = 0

        def claim(self, credential: str, hardware: HardwareReport) -> Claim:
            """Offer the next lease in sequence.

            Args:
                credential:
                    Credential supplied to the test.
                hardware:
                    Hardware supplied to the test.

            Returns:
                The next lease in the sequence, then nothing.
            """
            self.claims += 1
            if self.claims == 3:
                return Claim(lease=second)
            return Claim(lease=LEASE if self.claims <= 4 else None)

        def finalise(self, credential: str, lease_id: str) -> str:
            """Fail the first finalisation transiently, then accept it.

            Args:
                credential:
                    Credential supplied to the test.
                lease_id:
                    Lease being finalised.

            Returns:
                The stable submission identifier.

            Raises:
                BrokerError:
                    On the first attempt.
            """
            self.finalise_attempts += 1
            if self.finalise_attempts == 1:
                raise BrokerError("temporarily busy", status=503)
            return super().finalise(credential=credential, lease_id=lease_id)

    broker = ResumingBroker()
    delays: list[float] = []

    def record_sleep(delay: float) -> None:
        """Record each sleep and stop after the skip that follows the resume.

        Args:
            delay:
                Requested sleep duration.

        Raises:
            KeyboardInterrupt:
                Once a skip after the finalised lease has slept.
        """
        delays.append(delay)
        if broker.finalised and broker.claims >= 4:
            raise KeyboardInterrupt

    monkeypatch.setattr(runtime.time, "sleep", record_sleep)
    with pytest.raises(KeyboardInterrupt):
        runtime.Worker(
            client=broker,
            state=StateStore(directory=tmp_path),
            evaluator=OneRecordEvaluator(),
            hardware_factory=lambda: HARDWARE,
        ).run()
    assert broker.finalise_attempts == 2
    assert broker.releases == ["unsafe_model"] * 3
    backoffs = [delay for delay in delays if delay >= 30]
    assert backoffs[:2] == [30.0, 60.0]
    assert backoffs[-1] == 30.0


def test_resumed_unsafe_lease_does_not_stop_continuous_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep a continuous worker running when a resumed lease fails the safety check.

    Args:
        monkeypatch:
            Fixture replacing network dependencies and sleep.
        tmp_path:
            Private worker state directory.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(runtime, "check_model_safety", _unsafe_first_lease)
    state = StateStore(directory=tmp_path)
    state.save_active(lease=LEASE, github_login="login")
    broker = _second_lease_broker()
    broker.claims = 1

    def stop_after_finalisation(delay: float) -> None:
        """Stop the continuous loop once the next lease is finalised.

        Args:
            delay:
                Requested sleep duration.

        Raises:
            KeyboardInterrupt:
                Once the broker has finalised work.
        """
        if broker.finalised:
            raise KeyboardInterrupt

    monkeypatch.setattr(runtime.time, "sleep", stop_after_finalisation)
    with pytest.raises(KeyboardInterrupt):
        runtime.Worker(
            client=broker,
            state=state,
            evaluator=OneRecordEvaluator(),
            hardware_factory=lambda: HARDWARE,
        ).run()
    assert broker.releases == ["unsafe_model"]
    assert broker.finalised
    assert state.load_active() is None


@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize("launcher_managed", [False, True])
def test_unsafe_model_still_stops_once_and_launcher_managed_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    launcher_managed: bool,
    resumed: bool,
) -> None:
    """Preserve the existing exit behaviour outside plain continuous mode.

    Args:
        monkeypatch:
            Fixture replacing network dependencies.
        tmp_path:
            Private worker state directory.
        launcher_managed:
            Whether the worker runs under the Docker launcher.
        resumed:
            Whether the unsafe lease is resumed from saved state.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(runtime, "check_model_safety", _unsafe_first_lease)
    state = StateStore(directory=tmp_path)
    if resumed:
        state.save_active(
            lease=LEASE,
            github_login="login",
            managed_image_digest=LEASE.image_digest if launcher_managed else None,
        )
    broker = _second_lease_broker()
    worker = runtime.Worker(
        client=broker,
        state=state,
        evaluator=OneRecordEvaluator(),
        hardware_factory=lambda: HARDWARE,
        image_digest=LEASE.image_digest if launcher_managed else None,
    )
    if launcher_managed:
        assert worker.run(launcher_managed=True) == 1
    else:
        with pytest.raises(SafetyError):
            worker.run(once=True)
    assert broker.releases == ["unsafe_model"]
    assert broker.claims == (0 if resumed else 1)


VM_STAT = (
    "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
    "Pages free:                                  1216055.\n"
    "Pages inactive:                              1194506.\n"
    "Pages speculative:                             50000.\n"
)


def test_apple_silicon_probes_the_gpu_once_per_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Run the GPU probe subprocess once, but re-read available memory each time.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    calls = _fake_metal_host(
        monkeypatch=monkeypatch,
        cap_stdout=_metal_probe(cap=55_662_788_608),
        vm_stat=VM_STAT,
    )
    discover_hardware(state_dir=tmp_path)
    discover_hardware(state_dir=tmp_path)
    assert len(calls["subprocess"]) == 1
    assert sum(command[0] == "vm_stat" for command in calls["commands"]) == 2


@pytest.mark.parametrize(
    ("variable", "value", "refused"),
    [
        ("VLLM_TARGET_DEVICE", "cpu", True),
        ("VLLM_TARGET_DEVICE", "CPU", True),
        ("VLLM_PLUGINS", "", True),
        ("VLLM_PLUGINS", "gguf_metal", True),
        ("VLLM_PLUGINS", "metal,gguf_metal", False),
    ],
)
def test_apple_silicon_refuses_when_vllm_is_steered_to_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    variable: str,
    value: str,
    refused: bool,
) -> None:
    """Refuse to claim when an environment variable makes vLLM select the CPU.

    The variable is set after a successful discovery, so the check must not depend on
    the cached GPU probe.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
        variable:
            The vLLM environment variable to set.
        value:
            Its value.
        refused:
            Whether discovery must refuse to report the GPU.
    """
    _fake_metal_host(
        monkeypatch=monkeypatch,
        cap_stdout=_metal_probe(cap=55_662_788_608),
        vm_stat=VM_STAT,
    )
    assert discover_hardware(state_dir=tmp_path).selected_gpu_uuid == METAL_GPU_UUID
    monkeypatch.setenv(variable, value)
    if refused:
        with pytest.raises(NoGpuError, match=f"{variable}"):
            discover_hardware(state_dir=tmp_path)
    else:
        assert discover_hardware(state_dir=tmp_path).selected_gpu_uuid == METAL_GPU_UUID


def test_apple_silicon_reports_metal_gpu_with_unified_memory_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Report one Metal GPU whose free memory is capped by available host memory.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    cap = 55_662_788_608
    calls = _fake_metal_host(
        monkeypatch=monkeypatch, cap_stdout=_metal_probe(cap=cap), vm_stat=VM_STAT
    )
    report = discover_hardware(state_dir=tmp_path)
    available = (1216055 + 1194506) * 16384
    (gpu,) = report.gpus
    assert gpu.uuid == METAL_GPU_UUID
    assert gpu.name == "Apple M3 Max"
    assert gpu.total_memory_bytes == cap
    assert gpu.free_memory_bytes == available - 1024**3
    assert report.selected_gpu_uuid == METAL_GPU_UUID
    assert report.driver_version is None
    assert report.cuda_version is None
    assert report.ram_bytes == 64 * 1024**3
    assert ["sysctl", "-n", "hw.memsize"] in calls["commands"]
    assert all(command[0] != "nvidia-smi" for command in calls["commands"])
    ((command, options),) = calls["subprocess"]
    assert command[:2] == [hardware.sys.executable, "-c"]
    assert "max_recommended_working_set_size" in command[2]
    assert "mx.metal.is_available()" in command[2]
    assert "'vllm.platform_plugins'" in command[2]
    assert "torch.backends.mps.is_available()" in command[2]
    assert options["check"] is True
    timeout = options["timeout"]
    assert isinstance(timeout, (int, float))
    assert timeout > 0


def test_apple_silicon_retries_a_failed_gpu_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Probe again after a failed probe instead of remembering the failure.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    cap = 55_662_788_608
    _fake_metal_host(monkeypatch=monkeypatch, cap_stdout="not json", vm_stat=VM_STAT)
    with pytest.raises(NoGpuError, match="could not probe the Apple GPU"):
        discover_hardware(state_dir=tmp_path)
    monkeypatch.setattr(
        hardware.subprocess,
        "run",
        lambda command, **kwargs: type(
            "Completed", (), {"stdout": _metal_probe(cap=cap)}
        )(),
    )
    (gpu,) = discover_hardware(state_dir=tmp_path).gpus
    assert gpu.total_memory_bytes == cap


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("metal", "MLX cannot use the Apple GPU"),
        ("vllm_metal_plugin", "vllm-metal's platform plugin is not installed"),
        ("mps", "PyTorch cannot use the Apple GPU"),
    ],
)
def test_apple_silicon_without_a_gpu_backend_refuses_to_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: str, message: str
) -> None:
    """Stop before claiming when evaluation would fall back to the CPU.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
        missing:
            The GPU backend the probe reports as unavailable.
        message:
            Expected start of the error message.
    """
    _fake_metal_host(
        monkeypatch=monkeypatch,
        cap_stdout=_metal_probe(cap=55_662_788_608, **{missing: False}),
        vm_stat=VM_STAT,
    )
    with pytest.raises(NoGpuError, match=message):
        discover_hardware(state_dir=tmp_path)


def test_apple_silicon_without_host_memory_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Refuse to claim work when available host memory cannot be estimated.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    _fake_metal_host(
        monkeypatch=monkeypatch,
        cap_stdout=_metal_probe(cap=55_662_788_608),
        vm_stat="unexpected output",
    )
    with pytest.raises(NoGpuError, match="unified-memory availability"):
        discover_hardware(state_dir=tmp_path)


def test_apple_silicon_without_metal_cap_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Refuse to claim work when the Metal working-set limit cannot be read.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    _fake_metal_host(monkeypatch=monkeypatch, cap_stdout="not json", vm_stat=VM_STAT)
    with pytest.raises(NoGpuError, match="could not probe the Apple GPU"):
        discover_hardware(state_dir=tmp_path)


def test_gb10_contention_caps_reported_free_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Respect host contention even when CUDA reports substantially more free.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    monkeypatch.setattr(
        hardware.subprocess,
        "run",
        lambda *_args, **_kwargs: type(
            "Completed",
            (),
            {"stdout": '{"uuid":"GPU-test","free":14000000000,"total":130000000000}'},
        )(),
    )
    monkeypatch.setattr(hardware, "_host_available_bytes", lambda: 2 * 1024**3)
    gpu = discover_gpus(
        runner=lambda _command: "0, NVIDIA GB10, GPU-test, N/A, N/A, 12.1\n"
    )[0]
    assert gpu.free_memory_bytes == 1024**3


def test_gb10_discovery_fails_closed_on_unverified_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not report availability if device identity cannot be verified.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    monkeypatch.setattr(
        hardware.subprocess,
        "run",
        lambda *_args, **_kwargs: type(
            "Completed",
            (),
            {"stdout": '{"uuid":"GPU-other","free":10000000000,"total":130000000000}'},
        )(),
    )
    monkeypatch.setattr(hardware, "_host_available_bytes", lambda: 8 * 1024**3)
    gpu = discover_gpus(
        runner=lambda _command: "0, NVIDIA GB10, GPU-test, N/A, N/A, 12.1\n"
    )[0]
    assert gpu.free_memory_bytes == 0
    assert gpu.total_memory_bytes == 0


def test_gb10_malformed_and_failed_fallback_reports_no_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore malformed rows and fail closed when isolated CUDA probing fails.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    monkeypatch.setattr(
        hardware.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            hardware.subprocess.CalledProcessError(1, "python")
        ),
    )
    output = (
        "bad row\n"
        "0, NVIDIA GB10, GPU-test, N/A, N/A, 12.1\n"
        "1, NVIDIA A100, GPU-discrete, 1024, 4096, 8.0\\n"
    )
    gpus = discover_gpus(runner=lambda _command: output)
    assert len(gpus) == 2
    assert gpus[0].free_memory_bytes == 0
    assert gpus[0].total_memory_bytes == 0
    assert gpus[1].free_memory_bytes == 1024**3
    assert gpus[1].total_memory_bytes == 4 * 1024**3


def test_gb10_n_a_memory_uses_verified_cuda_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use isolated CUDA measurements and cap free memory by host availability.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    output = "0, NVIDIA GB10, GPU-test, N/A, N/A, 12.1\n"
    measured: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        """Fake run for this test.

        Args:
            command:
                Command supplied to the test.
            kwargs:
                Kwargs supplied to the test.

        Returns:
            The value returned by the test helper.
        """
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        measured.append(str(environment.get("CUDA_VISIBLE_DEVICES")))
        return type(
            "Completed",
            (),
            {"stdout": '{"uuid":"test","free":14000000000,"total":130000000000}'},
        )()

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)
    monkeypatch.setattr(hardware, "_host_available_bytes", lambda: 4 * 1024**3)
    gpu = discover_gpus(runner=lambda _command: output)[0]

    assert gpu.free_memory_bytes == 3 * 1024**3
    assert gpu.total_memory_bytes == 130000000000
    assert measured == ["GPU-test"]


@pytest.mark.parametrize("managed", [False, True])
def test_inspect_worker_state_is_readonly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], managed: bool
) -> None:
    """Inspect legacy and managed state without authentication or mutation.

    Args:
        tmp_path:
            State directory.
        capsys:
            Captured CLI output.
        managed:
            Whether persisted state contains managed provenance.
    """
    state = StateStore(directory=tmp_path)
    state.save_active(
        lease=LEASE, managed_image_digest=LEASE.image_digest if managed else None
    )
    if not managed:
        raw = json.loads(state.active_path.read_text())
        del raw["managed_image_digest"]
        state.active_path.write_text(json.dumps(raw))
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert cli.main(["--inspect-state", "--state-dir", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": "volunteer-worker-state/v1",
        "active": True,
        "finalised": False,
        "managed_image_digest": LEASE.image_digest if managed else None,
        "lease_image_digest": LEASE.image_digest,
    }
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }


def test_interruption_resumes_pre_evaluation_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Persist a lease before evaluation and resume it without reclaiming work.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
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

            Args:
                lease:
                    Lease supplied to the test.
                output_path:
                    Path where evaluation results would be written.

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


def test_keyboard_interrupt_releases_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cancellation releases an active lease.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
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

            Args:
                lease:
                    Lease supplied to the test.
                output_path:
                    Path where evaluation results would be written.

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


def test_lease_loss_releases_without_finalising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A lost heartbeat never permits finalisation.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )

    class LostHeartbeat:
        """Heartbeat fake reporting lease loss."""

        def __init__(self, client: object, credential: str, lease: Lease) -> None:
            """Init   for this test.

            Args:
                client:
                    Client supplied to the test.
                credential:
                    Credential supplied to the test.
                lease:
                    Lease supplied to the test.
            """
            pass

        def check(self) -> None:
            """Report a lost lease.

            Raises:
                LeaseLost: Always, for this test double.
            """
            raise runtime.LeaseLost("lost")

        def start(self) -> None:
            """Start the fake heartbeat."""

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


@pytest.mark.parametrize(
    "outcome,expected", [("idle", 10), ("update", 75), ("fatal", 1)]
)
def test_managed_claim_handshake_and_status(
    tmp_path: Path, outcome: str, expected: int
) -> None:
    """Advertise installed facts and stop before evaluation on idle or rejection.

    Args:
        tmp_path:
            State directory.
        outcome:
            Simulated claim outcome.
        expected:
            Managed process status.
    """
    state = StateStore(directory=tmp_path)
    state.save_auth(credential="cred", github_login="login")
    claims: list[dict[str, object]] = []

    def request(
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        """Inspect a managed claim and return its simulated outcome.

        Args:
            method:
                HTTP method.
            url:
                Claim endpoint.
            headers:
                Request headers.
            payload:
                Managed claim payload.

        Returns:
            An idle response.

        Raises:
            BrokerError:
                For the selected structured rejection.
        """
        assert url.endswith("claim")
        assert payload is not None
        claims.append(payload)
        assert "managed_image_v1" in payload["capabilities"]
        assert payload["image_digest"] == LEASE.image_digest
        assert payload["euroeval_version"] == importlib.metadata.version("euroeval")
        assert payload["worker_version"]
        hardware_payload = payload["hardware"]
        assert isinstance(hardware_payload, dict)
        assert hardware_payload["architecture"] == HARDWARE.architecture
        if outcome != "idle":
            raise BrokerError(
                "rejected",
                status=426 if outcome == "update" else 400,
                code="worker_update_required",
            )
        return {"protocol_version": "volunteer-worker/v1", "status": "no_work"}

    worker = runtime.Worker(
        client=BrokerClient(server="https://example.test", request=request),
        state=state,
        hardware_factory=lambda: HARDWARE,
        image_digest=LEASE.image_digest,
    )
    assert worker.run(launcher_managed=True) == expected
    assert len(claims) == 1
    assert state.load_active() is None


def test_managed_fresh_lease_stops_heartbeat_before_clearing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hand off only after finalisation, heartbeat join and durable state clearing.

    Args:
        monkeypatch:
            Dependency replacement fixture.
        tmp_path:
            State directory.
    """
    monkeypatch.setattr(runtime, "check_model_safety", lambda **kwargs: None)
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)

    class HandoffStore(StateStore):
        """Assert completion ordering at the public storage seam."""

        def clear_active(self) -> None:
            """Clear only a finalised lease with no running heartbeat."""
            active = self.load_active()
            assert active is not None
            assert active.finalised_submission_id == "submission-1"
            assert active.managed_image_digest == LEASE.image_digest
            assert not any(
                thread.name == "worker-heartbeat" and thread.is_alive()
                for thread in threading.enumerate()
            )
            super().clear_active()

    state = HandoffStore(directory=tmp_path)
    state.save_auth(credential="cred", github_login="login")
    broker = Broker()
    worker = runtime.Worker(
        client=broker,
        state=state,
        evaluator=OneRecordEvaluator(),
        hardware_factory=lambda: HARDWARE,
        image_digest=LEASE.image_digest,
    )
    assert worker.run(launcher_managed=True) == 0
    assert broker.claims == 1
    assert broker.finalised
    assert state.load_active() is None


@pytest.mark.parametrize("failure_stage", ["upload", "finalise", None])
@pytest.mark.parametrize("network_loss", [False, True])
def test_managed_handoff_resumes_exact_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str | None,
    network_loss: bool,
) -> None:
    """Preserve provenance and acknowledgements through response loss and handoff.

    Args:
        monkeypatch:
            Dependency replacement fixture.
        tmp_path:
            Private state directory.
        failure_stage:
            Operation whose response is lost before a process restart.
        network_loss:
            Whether response loss is a retryable transport error or an interrupt.
    """
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    state = StateStore(directory=tmp_path)
    state.save_auth(credential="cred", github_login="login")
    records = tuple(
        PendingRecord.from_record(EEERecord({"id": item})) for item in (1, 2)
    )
    state.save_active(
        lease=LEASE,
        records=records,
        github_login="login",
        managed_image_digest=LEASE.image_digest,
    )
    uploads: list[str] = []
    failed = False

    def request(
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        """Emulate idempotent broker endpoints and one interrupted response.

        Args:
            method:
                HTTP method.
            url:
                Endpoint URL.
            headers:
                Request headers.
            payload:
                Exact result envelope.

        Returns:
            Broker protocol response.

        Raises:
            KeyboardInterrupt:
                When the selected response is interrupted.
            BrokerError:
                When the selected response is lost over the transport.
        """
        nonlocal failed
        assert payload is not None
        assert not url.endswith("claim")
        if url.endswith("result"):
            uploads.append(str(payload["record_json"]))
            if failure_stage == "upload" and len(uploads) == 2 and not failed:
                failed = True
                if network_loss:
                    raise BrokerError("response lost", status=503)
                raise KeyboardInterrupt
        if url.endswith("finalise") and failure_stage == "finalise" and not failed:
            failed = True
            if network_loss:
                raise BrokerError("response lost", status=503)
            raise KeyboardInterrupt
        return {
            "protocol_version": "volunteer-worker/v1",
            "submission_id": "submission-1",
        }

    client = BrokerClient(server="https://example.test", request=request)
    worker = runtime.Worker(
        client=client,
        state=state,
        hardware_factory=lambda: HARDWARE,
        image_digest=LEASE.image_digest,
    )
    if failure_stage is not None and not network_loss:
        assert worker.run(launcher_managed=True) == 130
        active = state.load_active()
        assert active is not None
        assert active.managed_image_digest == LEASE.image_digest
        assert tuple(record.record_json for record in active.records) == tuple(
            record.record_json for record in records
        )
        assert tuple(record.digest for record in active.records) == tuple(
            record.digest for record in records
        )
        assert active.records[0].acknowledged
    assert worker.run(launcher_managed=True) == 0
    assert state.load_active() is None
    assert not any(
        thread.name == "worker-heartbeat" and thread.is_alive()
        for thread in threading.enumerate()
    )
    assert uploads.count(records[0].record_json) == 1


@pytest.mark.parametrize("finalised", [False, True])
def test_managed_provenance_survives_all_state_writes(
    tmp_path: Path, finalised: bool
) -> None:
    """Keep immutable provenance across renew, failure, record and completion writes.

    Args:
        tmp_path:
            State directory.
        finalised:
            Whether to persist completion before rejecting an image mismatch.
    """
    state = StateStore(directory=tmp_path)
    state.save_auth(credential="cred", github_login="login")
    state.save_active(
        lease=LEASE, github_login="login", managed_image_digest=LEASE.image_digest
    )
    state.renew_active(
        lease=dataclasses.replace(LEASE, expires_at="2099-02-01T00:00:00Z")
    )
    state.record_evaluation_failure(lease_id=LEASE.lease_id, max_failures=3)
    state.save_records(records=(PendingRecord.from_record(EEERecord({"id": 1})),))
    active = state.load_active()
    assert active is not None
    state.save_active(
        lease=active.lease,
        records=active.records,
        github_login="login",
        evaluation_failures=active.evaluation_failures,
    )
    if finalised:
        state.mark_finalised(submission_id="submission-1")
    before = state.active_path.read_bytes()
    active = state.load_active()
    assert active is not None
    assert active.managed_image_digest == LEASE.image_digest
    assert active.evaluation_failures == 1
    worker = runtime.Worker(
        client=Broker(),
        state=state,
        image_digest="sha256:other",
        hardware_factory=lambda: HARDWARE,
    )
    assert worker.run(launcher_managed=True) == 1
    assert state.active_path.read_bytes() == before


def test_no_gpu_exits_before_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Do not claim work when hardware discovery reports no GPU.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
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


def test_nvidia_csv_parser_handles_compute_capability_and_fails_without_gpu() -> None:
    """Parse memory and capability values, and fail closed without a GPU."""
    output = "1, NVIDIA A100, GPU-1, 10240, 20480, 8.0\n"
    gpu = discover_gpus(runner=lambda _command: output)[0]
    assert gpu.free_memory_bytes == 10 * 1024**3
    assert gpu.compute_capability == "8.0"
    assert gpu.index == 1
    with pytest.raises(NoGpuError):
        discover_gpus(runner=lambda _command: "")


def test_partial_acknowledgements_survive_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Retry only unacknowledged records after a failed submission.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )

    class TwoRecords:
        """Evaluator producing two deterministic records."""

        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            """Return two records.

            Args:
                lease:
                    Lease supplied to the test.
                output_path:
                    Path where evaluation results would be written.

            Returns:
                Two deterministic records with distinct identifiers.
            """
            return [EEERecord({"id": "one"}), EEERecord({"id": "two"})]

    class FailSecond(Broker):
        """Broker that loses the connection on the second record."""

        def __init__(self) -> None:
            """Initialise broker state and the second-record failure counter."""
            super().__init__()
            self.failures = 0

        def submit_result(
            self, credential: str, lease: Lease, result: EEERecord
        ) -> None:
            """Fail all retries for the second record on the first run.

            Args:
                credential:
                    Credential supplied to the test.
                lease:
                    Lease supplied to the test.
                result:
                    Result supplied to the test.

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


def test_pending_release_failure_does_not_reevaluate_or_clear_intent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep an exhausted lease pending when broker release is unavailable.

    Args:
        monkeypatch:
            Fixture replacing authentication.
        tmp_path:
            Private worker state directory.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    state = StateStore(tmp_path)
    state.save_active(
        lease=LEASE, evaluation_failures=3, release_pending=True, github_login="login"
    )

    class BusyReleaseBroker(Broker):
        """Broker whose issue mutex is temporarily occupied."""

        def release(self, credential: str, lease_id: str, reason: str) -> None:
            """Reject release with its structured temporary contention code.

            Raises:
                BrokerError:
                    To simulate temporary issue mutex contention.
            """
            raise BrokerError("issue busy", status=503, code="ISSUE_BUSY")

    class MustNotEvaluate:
        """Evaluator tracking accidental retries of exhausted work."""

        calls = 0

        def evaluate(self, lease: Lease, output_path: Path) -> list[EEERecord]:
            """Fail if the exhausted lease reaches evaluation.

            Raises:
                AssertionError:
                    If the worker reevaluates exhausted work.
            """
            self.calls += 1
            raise AssertionError("exhausted work must be released, not evaluated")

    evaluator_instance = MustNotEvaluate()
    with pytest.raises(BrokerError, match="issue busy"):
        runtime.Worker(
            client=BusyReleaseBroker(),
            state=state,
            evaluator=evaluator_instance,
            hardware_factory=lambda: HARDWARE,
        ).run(once=True)

    active = state.load_active()
    assert active is not None and active.release_pending
    assert active.evaluation_failures == 3
    assert evaluator_instance.calls == 0


def test_repeated_transient_claim_failures_keep_backoff_finite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep long-running retry backoff bounded without exponent overflow.

    Args:
        monkeypatch:
            Fixture replacing authentication and sleep.
        tmp_path:
            Private worker state directory.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    delays: list[float] = []

    def stop_after_many_retries(delay: float) -> None:
        delays.append(delay)
        if len(delays) == 1_100:
            raise KeyboardInterrupt

    monkeypatch.setattr(runtime.time, "sleep", stop_after_many_retries)

    class TemporarilyUnavailableBroker(Broker):
        """Broker returning a retryable request timeout."""

        def claim(self, credential: str, hardware: HardwareReport) -> Claim:
            """Raise a transient broker failure.

            Raises:
                BrokerError:
                    To simulate a temporary timeout response.
            """
            raise BrokerError("temporarily unavailable", status=408)

    with pytest.raises(KeyboardInterrupt):
        runtime.Worker(
            client=TemporarilyUnavailableBroker(),
            state=StateStore(tmp_path),
            evaluator=OneRecordEvaluator(),
            hardware_factory=lambda: HARDWARE,
        ).run()

    assert len(delays) == 1_100
    assert all(0 <= delay <= 300 for delay in delays)


def test_required_canary_preserves_partial_ordinary_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Evaluate only missing ordinary identities alongside the required canary.

    Args:
        monkeypatch:
            Pytest fixture for replacing evaluator dependencies.
        tmp_path:
            Temporary directory for isolated output.
    """
    calls: list[dict[str, object]] = []
    initialisations: list[dict[str, object]] = []
    danish = get_all_languages()["da"]
    actual_datasets = prepare_dataset_configs(
        task=None,
        dataset=None,
        languages=[danish],
        custom_datasets_file=Path(""),
        api_key=None,
        cache_dir=tmp_path,
        trust_remote_code=False,
        run_with_cli=False,
    )
    ordinary_datasets = [config for config in actual_datasets if not config.unofficial]
    assert len(ordinary_datasets) >= 2
    covered, missing = ordinary_datasets[:2]

    class FakeBenchmarker:
        """Resolve captured calls through EuroEval's real dataset selector."""

        def __init__(self, **kwargs: object) -> None:
            """Keep configured selector defaults for benchmark calls.

            Args:
                kwargs:
                    Configured evaluator parameters.
            """
            self.defaults = kwargs
            initialisations.append(kwargs)

        def benchmark(self, **kwargs: object) -> list[object]:
            """Resolve an invocation to real dataset configurations.

            Args:
                kwargs:
                    Benchmark invocation parameters.

            Returns:
                Result markers for the resolved datasets.
            """
            calls.append(kwargs)
            task = t.cast(
                str | list[str] | None, kwargs.get("task", self.defaults.get("task"))
            )
            dataset = t.cast(
                str | list[DatasetConfig] | None,
                kwargs.get("dataset", self.defaults.get("dataset")),
            )
            resolved = prepare_dataset_configs(
                task=task,
                dataset=dataset,
                languages=[danish],
                custom_datasets_file=Path(""),
                api_key=None,
                cache_dir=tmp_path,
                trust_remote_code=False,
                run_with_cli=False,
            )
            if task == ["contamination-detection"]:
                return ["canary"]
            return [config.name for config in resolved]

    scope = ExpectedScope(
        policy_version="test-policy",
        language_group="da",
        allowed_identity_suffix_sets=(
            (json.dumps([missing.name, True, True], separators=(",", ":")),),
        ),
        count=None,
        warnings=(),
        task_groups=("sequence_classification",),
        original_identity_suffix_sets=(
            tuple(
                sorted(
                    json.dumps([config.name, True, True], separators=(",", ":"))
                    for config in (covered, missing)
                )
            ),
        ),
        covered_identity_suffixes=(
            json.dumps([covered.name, True, True], separators=(",", ":")),
        ),
    )
    lease = dataclasses.replace(
        LEASE,
        expected_scope=scope,
        contamination_canary=CanaryInstruction(
            status="required",
            protocol_version="private-completion-canary/v1",
            corpus_revision="revision",
            corpus_sha256="a" * 64,
        ),
    )
    monkeypatch.setattr(evaluator, "Benchmarker", FakeBenchmarker)
    monkeypatch.setattr(
        evaluator,
        "benchmark_result_to_eee_dict",
        lambda result: {"evaluation_id": result, "result": 1},
    )

    output_path = tmp_path / "partial-canary.jsonl"
    records = evaluator.EuroEvalEvaluator(tmp_path).evaluate(
        lease=lease, output_path=output_path
    )

    assert len(initialisations) == 1
    assert initialisations[0]["task"] is None
    assert initialisations[0]["dataset"] is None
    assert len(calls) == 2
    ordinary_call, canary_call = calls
    ordinary_datasets = ordinary_call["dataset"]
    assert isinstance(ordinary_datasets, list)
    assert [config.name for config in ordinary_datasets] == [missing.name]
    assert ordinary_call.get("task") is None
    assert ordinary_call["few_shot"] is ShotMode.FEW_SHOT
    assert ordinary_call["evaluate_test_split"] is False
    assert canary_call["dataset"] is None
    assert canary_call["task"] == ["contamination-detection"]
    assert [json.loads(record.record_json)["evaluation_id"] for record in records] == [
        missing.name,
        "canary",
    ]
    assert [
        json.loads(line)["evaluation_id"]
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ] == [missing.name, "canary"]


def test_restart_submits_durable_records_without_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resume durable results without regenerating timestamped evaluation bytes.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
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

            Args:
                lease:
                    Lease supplied to the test.
                output_path:
                    Output path supplied to the test.

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


def test_state_round_trip_preserves_scope_alternatives_and_canary(
    tmp_path: Path,
) -> None:
    """Restart state preserves both trusted alternatives and canary instructions.

    Args:
        tmp_path:
            Temporary directory for files created by the test.
    """
    lease = dataclasses.replace(
        LEASE,
        contamination_canary=CanaryInstruction(
            status="required",
            protocol_version="canary/v1",
            corpus_revision="revision",
            corpus_sha256="a" * 64,
        ),
    )
    state = StateStore(tmp_path)

    state.save_active(lease=lease, evaluation_failures=2, release_pending=True)
    active = state.load_active()

    assert active is not None
    assert active.lease == lease
    assert active.evaluation_failures == 2
    assert active.release_pending


def test_unified_free_bytes_reserves_headroom_and_never_goes_negative() -> None:
    """Cap device memory by host availability minus 1 GiB, floored at zero."""
    gib = 1024**3
    assert (
        unified_free_bytes(device_free_bytes=50 * gib, host_available_bytes=10 * gib)
        == 9 * gib
    )
    assert (
        unified_free_bytes(device_free_bytes=4 * gib, host_available_bytes=10 * gib)
        == 4 * gib
    )
    assert (
        unified_free_bytes(device_free_bytes=4 * gib, host_available_bytes=gib // 2)
        == 0
    )


def test_worker_caps_vllm_utilisation_to_fresh_free_gpu_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Scale vLLM's total-memory reservation to current free memory.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )
    initial = HARDWARE
    contended_gpu = dataclasses.replace(
        GPU, free_memory_bytes=96 * 1024**3, total_memory_bytes=130 * 1024**3
    )
    fresh = dataclasses.replace(initial, gpus=(contended_gpu,))
    reports = iter((initial, fresh))
    observed: dict[str, object] = {}

    class CapturingEvaluator:
        """Capture the fresh GPU memory utilisation passed to evaluation."""

        def evaluate(
            self,
            lease: Lease,
            output_path: Path,
            gpu_memory_utilisation: float | None = None,
        ) -> list[EEERecord]:
            """Evaluate for this test.

            Args:
                lease:
                    Lease supplied to the test.
                output_path:
                    Output path supplied to the test.
                gpu_memory_utilisation (optional):
                    Gpu memory utilisation supplied to the test. Defaults to None.

            Returns:
                The value returned by the test helper.
            """
            observed["utilisation"] = gpu_memory_utilisation
            return [EEERecord({"id": "one"})]

    broker = Broker()
    runtime.Worker(
        client=broker,
        state=StateStore(tmp_path),
        evaluator=CapturingEvaluator(),
        hardware_factory=lambda: next(reports),
    ).run(once=True)

    assert observed["utilisation"] == pytest.approx(0.8 * 96 / 130)
    assert broker.finalised


def test_worker_measures_disk_on_its_state_filesystem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pass the state/model-cache path to default hardware discovery.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    state_dir = tmp_path / "mounted-cache"
    state_dir.mkdir()
    measured: list[Path] = []
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )

    def discover(*, state_dir: Path) -> HardwareReport:
        """Discover for this test.

        Args:
            state_dir:
                State dir supplied to the test.

        Returns:
            The value returned by the test helper.
        """
        measured.append(state_dir)
        return HARDWARE

    monkeypatch.setattr(runtime, "discover_hardware", discover)

    class NoWorkBroker(Broker):
        """Broker fake that reports no available lease."""

        def claim(self, credential: str, hardware: HardwareReport) -> Claim:
            """Claim for this test.

            Args:
                credential:
                    Credential supplied to the test.
                hardware:
                    Hardware supplied to the test.

            Returns:
                The value returned by the test helper.
            """
            return Claim(None)

    runtime.Worker(client=NoWorkBroker(), state=StateStore(state_dir)).run(once=True)

    assert measured == [state_dir]


def test_worker_retries_idempotently_and_finalises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Retry a result without changing its digest, then finalise.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
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


def test_worker_update_required_is_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit update rejection exits immediately rather than retrying claims.

    Args:
        monkeypatch:
            Fixture replacing authentication and retry sleeps.
        tmp_path:
            Private worker state directory.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("cred", "login")
    )
    delays: list[float] = []
    monkeypatch.setattr(runtime.time, "sleep", delays.append)
    calls: list[str] = []

    def reject_claim(request: urllib.request.Request, timeout: int) -> None:
        """Reject the claim with the broker's terminal HTTP update response.

        Args:
            request:
                Outgoing broker claim request.
            timeout:
                Network timeout configured by the broker client.

        Raises:
            urllib.error.HTTPError:
                Always, to simulate the upgrade-required HTTP response.
        """
        calls.append(request.full_url)
        response = {
            "protocol_version": "volunteer-worker/v1",
            "error": "Worker update required; update the worker image",
            "code": "worker_update_required",
        }
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=426,
            msg="Upgrade Required",
            hdrs=email.message.Message(),
            fp=io.BytesIO(json.dumps(response).encode()),
        )

    monkeypatch.setattr(urllib.request, "urlopen", reject_claim)
    worker = runtime.Worker(
        client=BrokerClient("https://broker.test"),
        state=StateStore(tmp_path),
        hardware_factory=lambda: HARDWARE,
    )
    with pytest.raises(BrokerError, match="update the worker image") as caught:
        worker.run(once=False)
    assert caught.value.status == 426
    assert caught.value.code == "worker_update_required"
    assert calls == ["https://broker.test/claim"]
    assert delays == []
    assert StateStore(tmp_path).load_active() is None
