"""The volunteer worker state machine."""

import collections.abc as c
import dataclasses
import datetime
import hashlib
import logging
import threading
import time

from .auth import authenticate
from .broker import BrokerError, BrokerProtocol
from .evaluator import EuroEvalEvaluator, Evaluator
from .hardware import discover_hardware
from .safety import SafetyError, check_model_safety
from .state import ActiveLease, PendingRecord, StateStore
from .types import Claim, EEERecord, HardwareReport, Lease

logger = logging.getLogger(__name__)


class LeaseLost(RuntimeError):
    """Raised when the broker stops accepting heartbeats."""


class Heartbeat:
    """Renew a lease in a daemon thread and expose failures to the worker."""

    def __init__(self, client: BrokerProtocol, credential: str, lease: Lease) -> None:
        """Initialise a heartbeat for a lease."""
        self.client = client
        self.credential = credential
        self.lease = lease
        self.failed: Exception | None = None
        self._stop = threading.Event()
        self._started = False
        self._thread = threading.Thread(
            target=self._run, name="worker-heartbeat", daemon=True
        )

    def start(self) -> None:
        """Start the heartbeat loop."""
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        """Stop and join the heartbeat loop."""
        self._stop.set()
        if self._started:
            self._thread.join(timeout=2)

    def check(self) -> None:
        """Raise the background failure, if any.

        Raises:
            LeaseLost:
                If a heartbeat failed.
        """
        if self.failed is not None:
            raise LeaseLost("broker heartbeat failed") from self.failed

    def _run(self) -> None:
        while not self._stop.wait(timeout=30):
            try:
                self.client.heartbeat(
                    credential=self.credential, lease_id=self.lease.lease_id
                )
            except Exception as error:  # noqa: BLE001 - thread reports all failures
                self.failed = error
                self._stop.set()
                return


class Worker:
    """Authenticate, claim, evaluate, and submit one lease at a time."""

    def __init__(
        self,
        client: BrokerProtocol,
        state: StateStore,
        evaluator: Evaluator | None = None,
        gpu_memory_utilisation: float = 0.8,
        hardware_factory: c.Callable[[], HardwareReport] = discover_hardware,
    ) -> None:
        """Initialise a worker.

        Args:
            client:
                Broker protocol client.
            state:
                Private state and retry storage.
            evaluator (optional):
                Evaluation adapter. Defaults to the EuroEval adapter.
            gpu_memory_utilisation (optional):
                Fraction of GPU memory offered to the evaluator. Defaults to 0.8.
            hardware_factory (optional):
                Hardware discovery function. Defaults to ``discover_hardware``.
        """
        self.client = client
        self.state = state
        self.evaluator = evaluator or EuroEvalEvaluator(
            cache_dir=state.directory, gpu_memory_utilisation=gpu_memory_utilisation
        )
        self.hardware_factory = hardware_factory
        self.last_submission_id: str | None = None

    def run(self, once: bool = False) -> None:
        """Run until interrupted, or process one broker response with ``once``.

        Raises:
            BrokerError:
                If authentication or a lease request fails.
            LeaseLost:
                If a resumable lease is no longer owned by this worker.
            RuntimeError:
                If the active lease disappears while credentials are refreshed.
        """
        credential, _login = authenticate(client=self.client, state=self.state)
        while True:
            hardware = self.hardware_factory()
            active = self.state.load_active()
            if active is not None and not _lease_is_valid(active.lease):
                logger.warning(
                    "Discarding expired local lease %s", active.lease.lease_id
                )
                self.state.archive_active()
                active = None
            if active is not None:
                try:
                    self._process_lease(
                        credential=credential,
                        lease=active.lease,
                        hardware=hardware,
                        active=active,
                    )
                except BrokerError as error:
                    if error.status != 401:
                        if _lease_lost(error):
                            self.state.archive_active()
                        raise
                    self.state.clear_auth()
                    credential, _login = authenticate(
                        client=self.client, state=self.state
                    )
                    self._process_lease(
                        credential=credential,
                        lease=active.lease,
                        hardware=hardware,
                        active=active,
                    )
                except LeaseLost as error:
                    if _lease_lost(error):
                        self.state.archive_active()
                    raise
                if once:
                    return
                continue

            claim = self._claim_with_reauthentication(
                credential=credential, hardware=hardware
            )
            if claim[0] != credential:
                credential = claim[0]
            response = claim[1]
            if response.lease is None:
                logger.info("No volunteer evaluation work is currently available")
                if once:
                    return
                time.sleep(30)
                continue
            try:
                self._process_lease(
                    credential=credential,
                    lease=response.lease,
                    hardware=hardware,
                    active=None,
                )
            except BrokerError as error:
                if error.status != 401:
                    if _lease_lost(error):
                        self.state.archive_active()
                    raise
                self.state.clear_auth()
                credential, _login = authenticate(client=self.client, state=self.state)
                resumed = self.state.load_active()
                if resumed is None:
                    raise RuntimeError(
                        "active lease disappeared during re-authentication"
                    )
                self._process_lease(
                    credential=credential,
                    lease=resumed.lease,
                    hardware=hardware,
                    active=resumed,
                )
            except LeaseLost as error:
                if _lease_lost(error):
                    self.state.archive_active()
                raise
            if once:
                return

    def _claim_with_reauthentication(
        self, credential: str, hardware: HardwareReport
    ) -> tuple[str, Claim]:
        """Claim once, replacing one revoked cached credential at most once.

        Returns:
            The credential used and the broker claim.

        Raises:
            BrokerError:
                If the claim fails after one re-authentication attempt.
        """
        try:
            return credential, self.client.claim(
                credential=credential, hardware=hardware
            )
        except BrokerError as error:
            if error.status != 401:
                raise
            self.state.clear_auth()
            new_credential, _login = authenticate(client=self.client, state=self.state)
            return new_credential, self.client.claim(
                credential=new_credential, hardware=hardware
            )

    def _process_lease(
        self,
        credential: str,
        lease: Lease,
        hardware: HardwareReport,
        active: ActiveLease | None,
    ) -> None:
        if active is None:
            self.state.save_active(lease=lease)
        heartbeat = Heartbeat(client=self.client, credential=credential, lease=lease)
        completed = False
        try:
            try:
                check_model_safety(
                    lease=lease,
                    gpus=hardware.gpus,
                    free_disk_bytes=hardware.free_disk_bytes,
                )
            except SafetyError:
                self.client.release(
                    credential=credential,
                    lease_id=lease.lease_id,
                    reason="unsafe_model",
                )
                self.state.archive_active()
                raise
            heartbeat.start()
            output = self.state.directory / "results" / f"{lease.lease_id}.jsonl"
            evaluated = self.evaluator.evaluate(lease=lease, output_path=output)
            heartbeat.check()
            try:
                records = self._durable_records(active=active, evaluated=evaluated)
            except RuntimeError:
                self.client.release(
                    credential=credential,
                    lease_id=lease.lease_id,
                    reason="incompatible_resume",
                )
                self.state.archive_active()
                raise
            self._submit_records(
                credential=credential, lease=lease, records=records, heartbeat=heartbeat
            )
            submission_id = self.client.finalise(
                credential=credential, lease_id=lease.lease_id
            )
            if isinstance(submission_id, str) and submission_id:
                self.state.save_submission_id(submission_id)
                self.last_submission_id = submission_id
                logger.info("Volunteer submission completed: %s", submission_id)
            self.state.clear_active()
            completed = True
        except (KeyboardInterrupt, SystemExit):
            raise
        finally:
            heartbeat.stop()
            if not completed:
                logger.info("Preserving active lease %s for restart", lease.lease_id)

    def _durable_records(
        self, active: ActiveLease | None, evaluated: list[EEERecord]
    ) -> tuple[PendingRecord, ...]:
        """Persist evaluation output, or verify it against a restart snapshot.

        Returns:
            Durable result entries, retaining prior acknowledgement flags.

        Raises:
            RuntimeError:
                If resumed evaluation differs from the durable snapshot.
        """
        for record in evaluated:
            digest = hashlib.sha256(record.record_json.encode("utf-8")).hexdigest()
            if digest != record.digest:
                raise RuntimeError("evaluation record digest does not match its JSON")
        fresh = tuple(PendingRecord.from_record(record) for record in evaluated)
        if active is None or not active.records:
            self.state.save_records(fresh)
            return fresh
        if len(active.records) != len(fresh) or any(
            old.record_json != new.record_json or old.digest != new.digest
            for old, new in zip(active.records, fresh, strict=True)
        ):
            raise RuntimeError("evaluation output changed while resuming a lease")
        self.state.save_records(active.records)
        return active.records

    def _submit_records(
        self,
        credential: str,
        lease: Lease,
        records: tuple[PendingRecord, ...],
        heartbeat: Heartbeat,
    ) -> None:
        """Submit and atomically acknowledge each exact result."""
        current = list(records)
        for index, pending in enumerate(current):
            if pending.acknowledged:
                continue
            self._submit_record(
                credential=credential, lease=lease, record=pending.to_record()
            )
            current[index] = dataclasses.replace(pending, acknowledged=True)
            self.state.save_records(tuple(current))
            heartbeat.check()

    def _submit_record(self, credential: str, lease: Lease, record: EEERecord) -> None:
        """Retry one exact result without changing its JSON representation."""
        for attempt in range(3):
            try:
                self.client.submit_result(
                    credential=credential, lease=lease, result=record
                )
                return
            except Exception:
                if attempt == 2:
                    raise
                logger.warning("Result submission failed; retrying idempotently")


def _lease_is_valid(lease: Lease) -> bool:
    """Check a broker ISO-8601 expiry without accepting malformed state.

    Returns:
        Whether the lease has a valid future expiry.
    """
    try:
        expiry = datetime.datetime.fromisoformat(
            lease.expires_at.replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return False
    if expiry.tzinfo is None:
        return False
    return expiry > datetime.datetime.now(datetime.UTC)


def _lease_lost(error: BaseException) -> bool:
    """Identify broker responses that make local work unsafe to resume.

    Returns:
        Whether the error indicates lost broker ownership.
    """
    return isinstance(error, LeaseLost) or (
        isinstance(error, BrokerError) and error.status == 409
    )
