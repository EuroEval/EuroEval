"""Tests for assignee-authoritative volunteer worker behaviour."""

import json
from pathlib import Path

import pytest

import src.scripts.volunteer_worker_operations as operations
from euroeval_worker import runtime
from euroeval_worker.broker import BrokerError
from euroeval_worker.state import StateStore
from euroeval_worker.types import (
    AuthPoll,
    AuthStart,
    Claim,
    EEERecord,
    ExpectedScope,
    Gpu,
    HardwareReport,
    Lease,
    ModelEvidence,
)

LEASE = Lease(
    lease_id="lease-assignee",
    issue_number=7,
    model_id="org/model",
    model_revision="a" * 40,
    language="da",
    euroeval_version="18.0.0",
    image_digest="sha256:image",
    expires_at="2099-01-01T00:00:00Z",
    model_type="encoder",
    worker_version="worker-1",
    selected_gpu_uuid="GPU-1",
    selected_gpu_index=0,
    model_metadata=ModelEvidence(
        pipeline_tag="fill-mask", architectures=("Model",), model_type="encoder"
    ),
    expected_scope=ExpectedScope(
        policy_version="test",
        language_group="da",
        allowed_identity_suffix_sets=(('["dataset",false,true]',),),
        count=1,
        warnings=(),
        task_groups=("sequence_classification",),
    ),
)
HARDWARE = HardwareReport(
    architecture="x86_64",
    ram_bytes=64,
    free_disk_bytes=100,
    driver_version="550",
    cuda_version="12.4",
    pytorch_version="2.7",
    gpus=(Gpu("A100", "GPU-1", 10, 20, "8.0", 0),),
)


def test_assignment_loss_archives_state_and_skips_finalisation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fenced lease is removed from active state and is never finalised.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("credential", "volunteer")
    )
    monkeypatch.setattr(
        runtime, "check_model_safety", lambda lease, gpus, free_disk_bytes=None: None
    )
    broker = Broker(
        BrokerError(
            "The lease contributor is no longer assigned",
            status=409,
            code="lease_assignment_lost",
        )
    )
    worker = runtime.Worker(
        client=broker,
        state=StateStore(tmp_path),
        evaluator=type(
            "Evaluator",
            (),
            {"evaluate": lambda self, lease, output_path: [EEERecord({"id": 1})]},
        )(),
        hardware_factory=lambda: HARDWARE,
    )

    with pytest.raises(BrokerError, match="no longer assigned"):
        worker.run(once=True)

    state = StateStore(tmp_path)
    assert state.load_active() is None
    assert list(state.archive_dir.glob("active-lease-*.json"))
    assert broker.submissions == 1
    assert broker.finalisations == 0


class Broker:
    """Minimal broker double for assignment-fence tests."""

    def __init__(self, error: BrokerError) -> None:
        """Initialise the double with a broker error to raise.

        Args:
            error:
                Error supplied to the test.
        """
        self.error = error
        self.claims = 0
        self.submissions = 0
        self.finalisations = 0

    def claim(self, credential: str, hardware: HardwareReport) -> Claim:
        """Return a lease unless the test is exercising claim rejection.

        Args:
            credential:
                Credential supplied to the test.
            hardware:
                Hardware supplied to the test.

        Returns:
            The test lease.

        Raises:
            The configured broker error if the test is exercising claim rejection.
        """
        del credential, hardware
        self.claims += 1
        if self.error.code == "github_login_not_assignable":
            raise self.error
        return Claim(lease=LEASE)

    def finalise(self, credential: str, lease_id: str) -> str:
        """Track unexpected finalisation attempts.

        Args:
            credential:
                Credential supplied to the test.
            lease_id:
                Lease id supplied to the test.

        Returns:
            An intentionally unexpected submission identifier.
        """
        del credential, lease_id
        self.finalisations += 1
        return "unexpected"

    def heartbeat(self, credential: str, lease_id: str) -> str:
        """Return the existing expiry.

        Args:
            credential:
                Credential supplied to the test.
            lease_id:
                Lease id supplied to the test.

        Returns:
            The existing lease expiry.
        """
        del credential, lease_id
        return LEASE.expires_at

    def poll_auth(self, session_id: str) -> AuthPoll:
        """Satisfy the authentication protocol type.

        Args:
            session_id:
                Session id supplied to the test.

        Raises:
            NotImplementedError: This test double does not implement authentication.
        """
        del session_id
        raise NotImplementedError

    def release(self, credential: str, lease_id: str, reason: str) -> None:
        """Satisfy the worker protocol.

        Args:
            credential:
                Credential supplied to the test.
            lease_id:
                Lease id supplied to the test.
            reason:
                Reason supplied to the test.
        """
        del credential, lease_id, reason

    def start_auth(self) -> AuthStart:
        """Satisfy the authentication protocol type.

        Raises:
            NotImplementedError: This test double does not implement authentication.
        """
        raise NotImplementedError

    def submit_result(self, credential: str, lease: Lease, result: EEERecord) -> None:
        """Reject the upload with the broker's assignment fence error.

        Args:
            credential:
                Credential supplied to the test.
            lease:
                Lease supplied to the test.
            result:
                Result supplied to the test.

        Raises:
            The configured assignment-fence error.
        """
        del credential, lease, result
        self.submissions += 1
        raise self.error


def test_claim_retries_transient_error_without_reauthentication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A transient claim error is retried with the existing credential.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for worker state.
    """
    broker = ClaimRetryBroker(
        errors=[BrokerError("temporary", status=503, retry_after=4)]
    )
    authentications: list[bool] = []
    delays: list[float] = []
    monkeypatch.setattr(
        runtime,
        "authenticate",
        lambda client, state: (
            authentications.append(True) or "credential",
            "volunteer",
        ),
    )
    monkeypatch.setattr(runtime.time, "sleep", delays.append)
    worker = runtime.Worker(
        client=broker, state=StateStore(tmp_path), hardware_factory=lambda: HARDWARE
    )

    worker.run(once=True)

    assert broker.claims == 2
    assert authentications == [True]
    assert delays == [4]


class ClaimRetryBroker(Broker):
    """Broker double that fails a configured number of claims."""

    def __init__(self, errors: list[BrokerError]) -> None:
        """Initialise the claim sequence.

        Args:
            errors:
                Errors to return before reporting no work.
        """
        super().__init__(error=BrokerError("unused"))
        self.errors = errors

    def claim(self, credential: str, hardware: HardwareReport) -> Claim:
        """Raise the next configured error, then return no work.

        Args:
            credential:
                Credential supplied by the worker.
            hardware:
                Hardware report supplied by the worker.

        Returns:
            A no-work claim after configured errors are exhausted.
        """
        del credential, hardware
        self.claims += 1
        if self.errors:
            raise self.errors.pop(0)
        return Claim(lease=None)


def test_claim_retry_exhaustion_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Claim retry exhaustion surfaces the final transient error.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for worker state.
    """
    broker = ClaimRetryBroker(
        errors=[BrokerError("unavailable", status=503) for _ in range(5)]
    )
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("credential", "volunteer")
    )
    delays: list[float] = []
    monkeypatch.setattr(runtime.time, "sleep", delays.append)
    worker = runtime.Worker(
        client=broker, state=StateStore(tmp_path), hardware_factory=lambda: HARDWARE
    )

    with pytest.raises(BrokerError, match="unavailable"):
        worker.run(once=True)

    assert broker.claims == 5
    assert delays == [1, 2, 4, 8]


def test_operations_inventory_has_no_removed_login() -> None:
    """Maintainer automation does not require or probe a fixed GitHub login."""
    source = Path(operations.__file__).read_text(encoding="utf-8")
    assert "WORKER_COORDINATOR_LOGIN" not in source
    assert "WORKER_COORDINATOR_SECRET" in operations.REQUIRED_ENVIRONMENT
    assert "WORKER_COORDINATOR_LOGIN" not in operations.REQUIRED_ENVIRONMENT


def test_permanent_claim_error_fails_without_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A permanent claim rejection is not retried.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for worker state.
    """
    broker = ClaimRetryBroker(errors=[BrokerError("invalid", status=422)])
    monkeypatch.setattr(
        runtime, "authenticate", lambda client, state: ("credential", "volunteer")
    )
    worker = runtime.Worker(
        client=broker, state=StateStore(tmp_path), hardware_factory=lambda: HARDWARE
    )

    with pytest.raises(BrokerError, match="invalid"):
        worker.run(once=True)

    assert broker.claims == 1


def test_structured_broker_error_preserves_code_message_and_redaction() -> None:
    """Broker errors expose API fields without retaining credential fields."""
    error = BrokerError(
        "generic",
        status=422,
        body={
            "error": "Use a GitHub-assignable login",
            "code": "github_login_not_assignable",
            "credential": "do-not-leak",
        },
    )

    assert error.code == "github_login_not_assignable"
    assert error.message == "Use a GitHub-assignable login"
    assert str(error) == error.message
    assert "credential" not in json.dumps(error.body)
    assert "do-not-leak" not in repr(error)


def test_unassignable_login_is_permanent_and_not_no_work(tmp_path: Path) -> None:
    """An unassignable OAuth login is surfaced without a claim retry.

    Args:
        tmp_path:
            Temporary directory for files created by the test.
    """
    error = BrokerError(
        "not assignable", status=422, code="github_login_not_assignable"
    )
    broker = Broker(error)
    worker = runtime.Worker(
        client=broker, state=StateStore(tmp_path), hardware_factory=lambda: HARDWARE
    )

    with pytest.raises(BrokerError, match="not assignable") as raised:
        worker._claim_with_reauthentication("credential", HARDWARE)

    assert raised.value.code == "github_login_not_assignable"
    assert broker.claims == 1
