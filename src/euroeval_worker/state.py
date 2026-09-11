"""Private worker credentials and restart-safe lease state."""

import dataclasses
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

from .types import EEERecord, JsonObject, Lease

logger = logging.getLogger(__name__)


def _lease_from_state(value: object) -> Lease:
    if not isinstance(value, dict):
        raise ValueError("lease is not an object")
    required = (
        "lease_id",
        "issue_number",
        "model_id",
        "model_revision",
        "language",
        "euroeval_version",
        "image_digest",
        "worker_version",
        "expires_at",
        "model_type",
    )
    if any(key not in value for key in required):
        raise ValueError("lease is incomplete")
    model_type = value["model_type"]
    if model_type not in {"encoder", "generative"}:
        raise ValueError("model_type is malformed")
    selected_gpu_uuid = value.get("selected_gpu_uuid")
    if selected_gpu_uuid is not None and (
        not isinstance(selected_gpu_uuid, str) or not selected_gpu_uuid
    ):
        raise ValueError("selected_gpu_uuid is malformed")
    selected_gpu_index = value.get("selected_gpu_index")
    if selected_gpu_index is not None and (
        isinstance(selected_gpu_index, bool) or not isinstance(selected_gpu_index, int)
    ):
        raise ValueError("selected_gpu_index is malformed")
    gpu_memory_utilisation = value.get("gpu_memory_utilisation", 0.8)
    if (
        isinstance(gpu_memory_utilisation, bool)
        or not isinstance(gpu_memory_utilisation, (int, float))
        or not 0 < gpu_memory_utilisation <= 1
    ):
        raise ValueError("gpu_memory_utilisation is malformed")
    return Lease(
        lease_id=value["lease_id"],
        issue_number=value["issue_number"],
        model_id=value["model_id"],
        model_revision=value["model_revision"],
        language=value["language"],
        euroeval_version=value["euroeval_version"],
        image_digest=value["image_digest"],
        worker_version=value["worker_version"],
        gpu_memory_utilisation=float(gpu_memory_utilisation),
        expires_at=value["expires_at"],
        model_type=model_type,
        selected_gpu_uuid=selected_gpu_uuid,
        selected_gpu_index=selected_gpu_index,
    )


@dataclasses.dataclass(frozen=True)
class PendingRecord:
    """A result and whether the broker has acknowledged it."""

    record_json: str
    digest: str
    acknowledged: bool = False

    @classmethod
    def from_record(cls, record: EEERecord) -> "PendingRecord":
        """Create pending state without re-serialising the result.

        Returns:
            The durable pending representation.
        """
        return cls(record_json=record.record_json, digest=record.digest)

    def to_record(self) -> EEERecord:
        """Return the exact result stored in this state entry."""
        return EEERecord(record_json=self.record_json, digest=self.digest)


@dataclasses.dataclass(frozen=True)
class ActiveLease:
    """A lease and all locally durable result acknowledgements."""

    lease: Lease
    records: tuple[PendingRecord, ...]
    github_login: str | None = None


class StateStore:
    """Persist credentials and one isolated active lease atomically."""

    def __init__(self, directory: Path) -> None:
        """Initialise state beneath ``directory``."""
        self.directory = directory
        self.path = directory / "state.json"
        self.active_path = directory / "active-lease.json"
        self.submission_path = directory / "last-submission.json"
        self.archive_dir = directory / "archive"
        self._lock = threading.RLock()

    def archive_active(self) -> None:
        """Move expired or lost work aside before claiming another lease."""
        if not self.active_path.exists():
            return
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        target = self.archive_dir / f"active-lease-{time.time_ns()}.json"
        self.active_path.replace(target)
        os.chmod(target, 0o600)

    def clear_active(self) -> None:
        """Remove the active lease after successful finalisation."""
        self.active_path.unlink(missing_ok=True)

    def clear_auth(self) -> None:
        """Forget an expired broker credential without touching lease state."""
        self.path.unlink(missing_ok=True)

    def load_auth(self) -> tuple[str, str] | None:
        """Return the saved opaque credential and verified login.

        Raises:
            RuntimeError:
                If the state file is malformed.
        """
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            credential = data["credential"]
            login = data["github_login"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid worker state: {self.path}") from error
        if not isinstance(credential, str) or not isinstance(login, str):
            raise RuntimeError(f"invalid worker state: {self.path}")
        return credential, login

    def load_submission_id(self) -> str | None:
        """Return the most recently reported submission identifier.

        Raises:
            RuntimeError:
                If the stored submission state is malformed.
        """
        if not self.submission_path.exists():
            return None
        try:
            value = json.loads(self.submission_path.read_text(encoding="utf-8"))[
                "submission_id"
            ]
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise RuntimeError(
                f"invalid submission state: {self.submission_path}"
            ) from error
        return value if isinstance(value, str) and value else None

    def renew_active(self, lease: Lease) -> None:
        """Atomically persist a broker-issued lease renewal.

        Raises:
            RuntimeError:
                If the active lease has disappeared or changed identity.
        """
        with self._lock:
            active = self.load_active()
            if active is None or active.lease.lease_id != lease.lease_id:
                raise RuntimeError("cannot renew a missing or different active lease")
            self._atomic_write(
                self.active_path,
                _active_dict(ActiveLease(lease, active.records, active.github_login)),
            )

    def _atomic_write(self, path: Path, value: JsonObject, mode: int = 0o600) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, mode)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
        os.chmod(path, mode)

    def load_active(self) -> ActiveLease | None:
        """Load the active lease, rejecting malformed or mixed state.

        Returns:
            The active lease, or ``None`` when no work is in progress.

        Raises:
            RuntimeError:
                If the active state is malformed.
            ValueError:
                If a stored record is not valid JSON.
        """
        if not self.active_path.exists():
            return None
        try:
            raw = json.loads(self.active_path.read_text(encoding="utf-8"))
            lease = _lease_from_state(raw["lease"])
            raw_records = raw["records"]
            if not isinstance(raw_records, list):
                raise ValueError("records is not a list")
            records = tuple(_pending_from_dict(item) for item in raw_records)
            github_login = raw.get("github_login")
            if github_login is not None and not isinstance(github_login, str):
                raise ValueError("github_login is malformed")
            return ActiveLease(lease=lease, records=records, github_login=github_login)
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise RuntimeError(
                f"invalid active worker state: {self.active_path}"
            ) from error

    def save_auth(self, credential: str, github_login: str) -> None:
        """Save device-flow credentials with restrictive permissions."""
        self._atomic_write(
            self.path,
            {"credential": credential, "github_login": github_login},
            mode=0o600,
        )

    def save_records(self, records: tuple[PendingRecord, ...]) -> None:
        """Atomically replace result and acknowledgement state for the lease.

        Raises:
            RuntimeError:
                If there is no active lease.
        """
        with self._lock:
            active = self.load_active()
            if active is None:
                raise RuntimeError("cannot save records without an active lease")
            self.save_active(
                lease=active.lease, records=records, github_login=active.github_login
            )

    def save_active(
        self,
        lease: Lease,
        records: tuple[PendingRecord, ...] = (),
        github_login: str | None = None,
    ) -> None:
        """Atomically save a lease before evaluation or submission starts."""
        with self._lock:
            self._atomic_write(
                self.active_path,
                _active_dict(ActiveLease(lease, records, github_login)),
            )

    def save_submission_id(self, submission_id: str) -> None:
        """Persist a successful submission identifier for reporting."""
        self._atomic_write(self.submission_path, {"submission_id": submission_id})


def _active_dict(active: ActiveLease) -> JsonObject:
    """Encode active state without touching result JSON text.

    Returns:
        The JSON-compatible active state.
    """
    lease = dataclasses.asdict(active.lease)
    return {
        "lease": lease,
        "records": [dataclasses.asdict(record) for record in active.records],
        "github_login": active.github_login,
    }


def _pending_from_dict(value: object) -> PendingRecord:
    if not isinstance(value, dict):
        raise ValueError("pending record is not an object")
    record_json = value.get("record_json")
    digest = value.get("digest")
    acknowledged = value.get("acknowledged", False)
    if (
        not isinstance(record_json, str)
        or not isinstance(digest, str)
        or not isinstance(acknowledged, bool)
    ):
        raise ValueError("pending record is malformed")
    EEERecord(record_json=record_json, digest=digest)
    expected = hashlib.sha256(record_json.encode("utf-8")).hexdigest()
    if digest != expected:
        raise ValueError("pending record digest does not match its JSON")
    return PendingRecord(record_json, digest, acknowledged)


def default_state_dir() -> Path:
    """Select a writable container cache, then the user's cache directory.

    Returns:
        A writable state directory.

    Raises:
        RuntimeError:
            If no candidate directory is writable.
    """
    configured = os.environ.get("EUROEVAL_WORKER_CACHE")
    candidates = [Path(configured)] if configured else []
    candidates.extend([Path("/cache"), Path.home() / ".cache" / "euroeval-worker"])
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write-test"
            probe.touch()
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise RuntimeError("no writable worker state directory; use --state-dir")
