"""The volunteer worker state machine."""

import collections.abc as c
import dataclasses
import datetime
import hashlib
import inspect
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


class AuthenticationIdentityError(BrokerError):
    """Raised when reauthentication returns a different contributor."""

    def __init__(self, expected: str, actual: str) -> None:
        """Describe the contributor mismatch without including credentials."""
        super().__init__(
            f"reauthentication returned GitHub login {actual!r}; expected {expected!r}",
            status=403,
        )


class Heartbeat:
    """Renew a lease in a daemon thread and expose failures to the worker."""

    _interval = 30.0
    _max_backoff = 30.0

    def __init__(
        self,
        client: BrokerProtocol,
        credential: str,
        lease: Lease,
        persist: c.Callable[[Lease], None] | None = None,
        reauthenticate: c.Callable[[str], str] | None = None,
    ) -> None:
        """Initialise a heartbeat for a lease."""
        self.client = client
        self.credential = credential
        self.lease = lease
        self.persist = persist
        self.reauthenticate = reauthenticate
        self.failed: Exception | None = None
        self._stop = threading.Event()
        self._started = False
        self._lease_lock = threading.Lock()
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
        """Raise the background failure, if any."""
        if self.failed is None:
            return
        if isinstance(self.failed, LeaseLost):
            raise self.failed
        raise self.failed

    def _run(self) -> None:
        while not self._stop.wait(timeout=self._interval):
            self._renew()
            if self.failed is not None:
                return

    def _renew(self) -> None:
        attempt = 0
        auth_attempted = False
        while not self._stop.is_set():
            try:
                expires_at = self.client.heartbeat(
                    credential=self.credential, lease_id=self.lease.lease_id
                )
                if not isinstance(expires_at, str) or not expires_at:
                    raise BrokerError("broker heartbeat response omitted expires_at")
                renewed = dataclasses.replace(self.lease, expires_at=expires_at)
                if self.persist is not None:
                    self.persist(renewed)
                with self._lease_lock:
                    self.lease = renewed
                return
            except BrokerError as error:
                if error.status == 401 and not auth_attempted:
                    if self.reauthenticate is None:
                        self._fail(error)
                        return
                    try:
                        self.credential = self.reauthenticate(self.credential)
                    except Exception as refresh_error:  # noqa: BLE001
                        self._fail(refresh_error)
                        return
                    auth_attempted = True
                    continue
                if error.status == 409:
                    self._fail(LeaseLost("broker heartbeat lost the lease"))
                    return
                if not _transient(error):
                    self._fail(error)
                    return
                retry_after = error.retry_after
            except Exception:  # noqa: BLE001 - network clients vary
                retry_after = None
            delay = _retry_delay(attempt, retry_after, self._max_backoff)
            attempt += 1
            if self._stop.wait(delay):
                return

    def _fail(self, error: Exception) -> None:
        self.failed = error
        self._stop.set()


class Worker:
    """Authenticate, claim, evaluate, and submit one lease at a time."""

    _max_submission_attempts = 3
    _max_retry_delay = 30.0

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
        self.gpu_memory_utilisation = gpu_memory_utilisation
        self.evaluator = evaluator or EuroEvalEvaluator(
            cache_dir=state.directory, gpu_memory_utilisation=gpu_memory_utilisation
        )
        self.hardware_factory = hardware_factory
        self.last_submission_id: str | None = None
        self._credential = ""
        self._login = ""
        self._auth_lock = threading.Lock()

    def run(self, once: bool = False) -> None:
        """Run until interrupted, or process one broker response with ``once``.

        Raises:
            BrokerError:
                If the broker rejects an operation.
            AuthenticationIdentityError:
                If a resumed lease is presented to a different contributor.
            LeaseLost:
                If broker ownership is lost.
        """
        self._credential, self._login = authenticate(
            client=self.client, state=self.state
        )
        while True:
            hardware = self.hardware_factory()
            active = self.state.load_active()
            if active is not None and active.github_login not in (None, self._login):
                raise AuthenticationIdentityError(
                    active.github_login or "", self._login
                )
            if active is not None and not _lease_is_valid(active.lease):
                logger.warning(
                    "Discarding expired local lease %s", active.lease.lease_id
                )
                self.state.archive_active()
                active = None
            if active is not None:
                try:
                    self._process_lease(
                        credential=self._credential,
                        lease=active.lease,
                        hardware=hardware,
                        active=active,
                    )
                except (BrokerError, LeaseLost) as error:
                    if _lease_lost(error):
                        self.state.archive_active()
                    raise
                if once:
                    return
                continue

            credential, response = self._claim_with_reauthentication(
                credential=self._credential, hardware=hardware
            )
            self._credential = credential
            if response.lease is None:
                logger.info("No volunteer evaluation work is currently available")
                if once:
                    return
                time.sleep(30)
                continue
            try:
                self._process_lease(
                    credential=self._credential,
                    lease=response.lease,
                    hardware=hardware,
                    active=None,
                )
            except (BrokerError, LeaseLost) as error:
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
            The credential used and the broker response.

        Raises:
            BrokerError:
                If claiming fails after one authentication retry.
        """
        try:
            return credential, self.client.claim(
                credential=credential, hardware=hardware
            )
        except BrokerError as error:
            if error.status != 401:
                raise
            self.state.clear_auth()
            new_credential, login = authenticate(client=self.client, state=self.state)
            self._login = login
            self._credential = new_credential
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
        expected_login = active.github_login or self._login if active else self._login
        if active is None:
            self.state.save_active(lease=lease, github_login=expected_login or None)
        elif active.github_login is None and expected_login:
            self.state.save_active(
                lease=active.lease, records=active.records, github_login=expected_login
            )
        heartbeat_parameters = inspect.signature(Heartbeat).parameters
        if "persist" in heartbeat_parameters:
            heartbeat = Heartbeat(
                client=self.client,
                credential=credential,
                lease=lease,
                persist=self.state.renew_active,
                reauthenticate=lambda failed: self._reauthenticate(
                    failed, expected_login
                ),
            )
        else:
            heartbeat = Heartbeat(
                client=self.client, credential=credential, lease=lease
            )
        completed = False
        try:
            try:
                if (
                    "gpu_memory_utilisation"
                    in inspect.signature(check_model_safety).parameters
                ):
                    check_model_safety(
                        lease=lease,
                        gpus=hardware.gpus,
                        free_disk_bytes=hardware.free_disk_bytes,
                        gpu_memory_utilisation=self.gpu_memory_utilisation,
                    )
                else:
                    check_model_safety(
                        lease=lease,
                        gpus=hardware.gpus,
                        free_disk_bytes=hardware.free_disk_bytes,
                    )
            except SafetyError:
                self.client.release(
                    credential=self._credential,
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
                    credential=self._credential,
                    lease_id=lease.lease_id,
                    reason="incompatible_resume",
                )
                self.state.archive_active()
                raise
            self._submit_records(
                credential=self._credential,
                lease=lease,
                records=records,
                heartbeat=heartbeat,
            )
            submission_id = self._finalise(
                credential=self._credential, lease_id=lease.lease_id
            )
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

    def _reauthenticate(self, failed_credential: str, expected_login: str) -> str:
        """Refresh a credential while proving the contributor did not change.

        Returns:
            A credential belonging to ``expected_login``.

        Raises:
            AuthenticationIdentityError:
                If the refreshed credential belongs to another contributor.
        """
        with self._auth_lock:
            saved = self.state.load_auth()
            if saved is not None and saved[0] != failed_credential:
                if saved[1] != expected_login:
                    raise AuthenticationIdentityError(expected_login, saved[1])
                self._login = saved[1]
                self._credential = saved[0]
                return saved[0]
            self.state.clear_auth()
            credential, login = authenticate(client=self.client, state=self.state)
            if login != expected_login:
                raise AuthenticationIdentityError(expected_login, login)
            self.state.save_auth(credential=credential, github_login=login)
            self._login = login
            self._credential = credential
            return credential

    def _durable_records(
        self, active: ActiveLease | None, evaluated: list[EEERecord]
    ) -> tuple[PendingRecord, ...]:
        """Persist evaluation output, or verify it against a restart snapshot.

        Returns:
            Durable records with existing acknowledgement flags retained.

        Raises:
            RuntimeError:
                If the evaluation output differs from its restart snapshot.
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
        """Retry one exact result under the bounded transient-error policy.

        Raises:
            BrokerError:
                If the broker reports a terminal result error.
        """
        del credential
        auth_attempted = False
        for attempt in range(self._max_submission_attempts):
            try:
                self.client.submit_result(
                    credential=self._credential, lease=lease, result=record
                )
                return
            except BrokerError as error:
                if error.status == 401 and not auth_attempted:
                    self._credential = self._reauthenticate(
                        self._credential, self._login
                    )
                    auth_attempted = True
                    continue
                if (
                    not _submission_transient(error)
                    or attempt + 1 >= self._max_submission_attempts
                ):
                    raise
                retry_after = error.retry_after
            except Exception:  # noqa: BLE001 - network clients vary
                if attempt + 1 >= self._max_submission_attempts:
                    raise
                retry_after = None
            logger.warning("Result submission failed; retrying idempotently")
            time.sleep(_retry_delay(attempt, retry_after, self._max_retry_delay))

    def _finalise(self, credential: str, lease_id: str) -> str:
        """Finalise once, retrying authentication exactly once if required.

        Returns:
            The stable broker submission identifier.

        Raises:
            BrokerError:
                If finalisation fails.
        """
        del credential
        auth_attempted = False
        while True:
            try:
                return self.client.finalise(
                    credential=self._credential, lease_id=lease_id
                )
            except BrokerError as error:
                if error.status != 401 or auth_attempted:
                    raise
                self._credential = self._reauthenticate(self._credential, self._login)
                auth_attempted = True


def _transient(error: BrokerError) -> bool:
    """Return whether a broker error is safe to retry."""
    return error.status is None or error.status == 429 or error.status >= 500


def _submission_transient(error: BrokerError) -> bool:
    """Return whether result upload may be retried.

    Returns:
        Whether the status is in the result retry set.
    """
    return (
        error.status is None or error.status in (408, 425, 429) or error.status >= 500
    )


def _retry_delay(attempt: int, retry_after: float | None, maximum: float) -> float:
    """Calculate bounded exponential backoff, honouring Retry-After.

    Returns:
        A non-negative delay no larger than ``maximum``.
    """
    requested = retry_after if retry_after is not None else 2**attempt
    return min(maximum, max(0.0, requested))


def _lease_is_valid(lease: Lease) -> bool:
    """Check a broker ISO-8601 expiry without accepting malformed state.

    Returns:
        Whether the expiry is a future timezone-aware timestamp.
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
    """Identify errors that make local work unsafe to resume.

    Returns:
        Whether broker ownership has definitely been lost.
    """
    return isinstance(error, LeaseLost) or (
        isinstance(error, BrokerError) and error.status == 409
    )
