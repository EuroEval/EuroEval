"""The volunteer worker state machine."""

import collections.abc as c
import hashlib
import json
import logging
import threading
import time

from .auth import authenticate
from .broker import BrokerError, BrokerProtocol
from .evaluator import EuroEvalEvaluator, Evaluator
from .hardware import discover_hardware
from .safety import SafetyError, check_model_safety
from .state import StateStore
from .types import EEERecord, HardwareReport, Lease

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
            except Exception as error:  # noqa: BLE001 - thread must report all failures
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

    def run(self, once: bool = False) -> None:
        """Run until interrupted, or process one broker response with ``once``.

        Raises:
            BrokerError: If authentication or a lease request fails.
        """
        credential, _login = authenticate(client=self.client, state=self.state)
        while True:
            hardware = self.hardware_factory()
            try:
                claim = self.client.claim(credential=credential, hardware=hardware)
            except BrokerError as error:
                if error.status != 401:
                    raise
                # A broker credential is opaque and revocable. Re-run the device
                # flow once, then let a second failure surface normally.
                self.state.clear_auth()
                credential, _login = authenticate(client=self.client, state=self.state)
                claim = self.client.claim(credential=credential, hardware=hardware)
            if claim.lease is None:
                logger.info("No volunteer evaluation work is currently available")
                if once:
                    return
                time.sleep(30)
                continue
            self._process_lease(
                credential=credential, lease=claim.lease, hardware=hardware
            )
            if once:
                return

    def _process_lease(
        self, credential: str, lease: Lease, hardware: HardwareReport
    ) -> None:
        heartbeat = Heartbeat(client=self.client, credential=credential, lease=lease)
        completed = False
        released = False
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
                released = True
                raise
            heartbeat.start()
            output = self.state.directory / "results" / f"{lease.lease_id}.jsonl"
            records = self.evaluator.evaluate(lease=lease, output_path=output)
            heartbeat.check()
            for record in records:
                self._submit_record(credential=credential, lease=lease, record=record)
                heartbeat.check()
            self.client.finalise(credential=credential, lease_id=lease.lease_id)
            self.state.clear_results()
            completed = True
        except (KeyboardInterrupt, SystemExit):
            raise
        finally:
            heartbeat.stop()
            if not completed and not released:
                try:
                    self.client.release(
                        credential=credential,
                        lease_id=lease.lease_id,
                        reason="worker_interrupted_or_failed",
                    )
                except Exception:  # noqa: BLE001 - preserve the original failure
                    logger.exception("Could not release failed lease")

    def _submit_record(self, credential: str, lease: Lease, record: EEERecord) -> None:
        encoded = json.dumps(
            record.record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        record = EEERecord(
            record=record.record, sha256=hashlib.sha256(encoded).hexdigest()
        )
        self.state.append_result(json.dumps(record.record, sort_keys=True))
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
