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
        identity_suffixes=("[]",),
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


class Broker:
    """Minimal broker double for assignment-fence tests."""

    def __init__(self, error: BrokerError) -> None:
        """Initialise the double with a broker error to raise."""
        self.error = error
        self.claims = 0
        self.submissions = 0
        self.finalisations = 0

    def poll_auth(self, session_id: str) -> AuthPoll:
        """Satisfy the authentication protocol type."""
        del session_id
        raise NotImplementedError

    def start_auth(self) -> AuthStart:
        """Satisfy the authentication protocol type."""
        raise NotImplementedError

    def claim(self, credential: str, hardware: HardwareReport) -> Claim:
        """Return a lease unless the test is exercising claim rejection."""
        del credential, hardware
        self.claims += 1
        if self.error.code == "github_login_not_assignable":
            raise self.error
        return Claim(lease=LEASE)

    def heartbeat(self, credential: str, lease_id: str) -> str:
        """Return the existing expiry."""
        del credential, lease_id
        return LEASE.expires_at

    def submit_result(self, credential: str, lease: Lease, result: EEERecord) -> None:
        """Reject the upload with the broker's assignment fence error."""
        del credential, lease, result
        self.submissions += 1
        raise self.error

    def finalise(self, credential: str, lease_id: str) -> str:
        """Track unexpected finalisation attempts.

        Returns:
            An intentionally unexpected submission identifier.
        """
        del credential, lease_id
        self.finalisations += 1
        return "unexpected"

    def release(self, credential: str, lease_id: str, reason: str) -> None:
        """Satisfy the worker protocol."""
        del credential, lease_id, reason


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
    """An unassignable OAuth login is surfaced without a claim retry."""
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


def test_assignment_loss_archives_state_and_skips_finalisation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fenced lease is removed from active state and is never finalised."""
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


def test_operations_inventory_has_no_removed_login() -> None:
    """Maintainer automation does not require or probe a fixed GitHub login."""
    source = Path(operations.__file__).read_text(encoding="utf-8")
    assert "WORKER_COORDINATOR_LOGIN" not in source
    assert "WORKER_COORDINATOR_SECRET" in operations.REQUIRED_ENVIRONMENT
    assert "WORKER_COORDINATOR_LOGIN" not in operations.REQUIRED_ENVIRONMENT
