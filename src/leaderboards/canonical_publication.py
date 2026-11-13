"""Serialize durable writes to canonical EuroEval result records.

A publication session owns the broker's no-TTL lock from before its first remote
read until all mutations and verification are complete. Callers must pass the
session explicitly; lock ownership is never inherited by evaluation subprocesses.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import secrets
import tempfile
import typing as t
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from euroeval.numeric_model_metadata import positive_context_length, positive_integer

from .contamination_canary import is_canary_record
from .result_identity import (
    ResultIdentity,
    dedup_newer_record,
    identity_from_eee_record,
    identity_to_path,
)

DEFAULT_BUCKET = "EuroEval/results"


class BucketApi(t.Protocol):
    """Hugging Face bucket methods used by the publisher."""

    def batch_bucket_files(
        self,
        bucket_id: str,
        add: list[tuple[Path, str]] | None = None,
        delete: list[str] | None = None,
        token: str = "",
    ) -> None:
        """Add or delete bucket paths."""

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[str, Path]],
        *,
        raise_on_missing_files: bool,
        token: str,
    ) -> None:
        """Download selected bucket paths."""


class PublicationError(RuntimeError):
    """A canonical publication could not be completed safely."""


def _contains_canary(content: bytes) -> bool:
    """Identify embedded canary JSON before a generic write can expose it.

    Args:
        content: Object bytes to inspect.

    Returns:
        Whether the bytes decode as a canary result record.
    """
    try:
        record = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(record, dict) and is_canary_record(record)


def _decode_record(content: bytes, path: str) -> dict[str, object]:
    """Decode one JSON result record.

    Args:
        content: Raw JSON bytes.
        path: Label used in validation errors.

    Returns:
        Decoded JSON object.

    Raises:
        PublicationError:
            If the bytes do not contain a JSON object.
    """
    try:
        record = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError(f"Invalid JSON result at {path}") from exc
    if not isinstance(record, dict):
        raise PublicationError(f"Result at {path} must be a JSON object")
    return record


def _metadata_only_transform(
    current: dict[str, object], candidate: dict[str, object]
) -> bool:
    """Check that a result change is limited to model metadata.

    Args:
        current: Existing canonical result record.
        candidate: Proposed replacement result record.

    Returns:
        Whether all non-metadata fields are unchanged.
    """
    if (
        identity_from_eee_record(current) != identity_from_eee_record(candidate)
        or current.get("retrieved_timestamp") != candidate.get("retrieved_timestamp")
        or current.get("eval_library", {}).get("version")
        != candidate.get("eval_library", {}).get("version")
    ):
        return False

    current_copy = copy.deepcopy(current)
    candidate_copy = copy.deepcopy(candidate)
    current_details = _model_additional_details(record=current_copy)
    candidate_details = _model_additional_details(record=candidate_copy)
    for key, validator in (
        ("num_model_parameters", positive_integer),
        ("vocabulary_size", positive_integer),
        ("max_sequence_length", positive_context_length),
    ):
        if (
            validator(current_details.get(key)) is None
            and validator(candidate_details.get(key)) is not None
        ):
            current_details.pop(key, None)
            candidate_details.pop(key, None)
    for record in (current_copy, candidate_copy):
        record.pop("metadata", None)
        model_info = record.get("model_info")
        if isinstance(model_info, dict):
            additional = model_info.get("additional_details")
            if isinstance(additional, dict):
                for key in (
                    "commercially_licensed",
                    "open",
                    "merge",
                    "trained_from_scratch",
                    "generative_type",
                    "model_url",
                    "release_date",
                ):
                    additional.pop(key, None)
                if not additional:
                    model_info.pop("additional_details", None)
    return current_copy == candidate_copy


def _model_additional_details(record: dict[str, object]) -> dict[str, object]:
    """Return model additional details, creating no missing record structure.

    Args:
        record: A decoded result record.

    Returns:
        The model metadata details, or an empty mapping when absent.
    """
    model_info = record.get("model_info")
    if not isinstance(model_info, dict):
        return {}
    details = model_info.get("additional_details")
    return details if isinstance(details, dict) else {}


def _receipt_path(owner: str) -> Path:
    """Return the private recovery receipt path for an owner token."""
    root = Path(os.environ.get("XDG_CACHE_HOME", tempfile.gettempdir()))
    return root / "euroeval-publication-locks" / f"{owner}.json"


def _validated_candidates(
    records: t.Sequence[bytes],
) -> list[tuple[str, ResultIdentity, bytes, dict[str, object]]]:
    """Parse candidates, exclude private canaries, and detect path collisions.

    Args:
        records: Candidate record bytes.

    Returns:
        Validated candidate entries.

    Raises:
        PublicationError:
            If a candidate is invalid, private, or collides with another identity.
    """
    result: dict[str, tuple[ResultIdentity, bytes, dict[str, object]]] = {}
    for content in records:
        record = _decode_record(content, "candidate")
        if is_canary_record(record):
            raise PublicationError("Canary records cannot be published canonically")
        identity = identity_from_eee_record(record)
        path = identity_to_path(identity).as_posix()
        if path in result and result[path][0] != identity:
            raise PublicationError(f"Distinct result identities collide at {path}")
        if path in result and result[path][1] != content:
            raise PublicationError(f"Multiple candidate records target {path}")
        result[path] = (identity, content, record)
    return [(path, *value) for path, value in result.items()]


def _write_receipt(path: Path, owner: str, endpoint: str, bucket: str) -> None:
    """Persist lock recovery details with restrictive file permissions.

    Args:
        path: Receipt file path.
        owner: Opaque lock owner token.
        endpoint: Broker lock endpoint.
        bucket: Canonical bucket identifier.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as receipt:
        json.dump({"owner": owner, "endpoint": endpoint, "bucket": bucket}, receipt)


class LockAdapter(t.Protocol):
    """Explicit broker lock interface for injectable lock adapters."""

    bucket: str
    endpoint: str

    def request(self, action: str, owner: str) -> str:
        """Perform one lock operation."""


class PublicationLock:
    """HTTP adapter for the broker's durable canonical-publication lock."""

    def __init__(
        self, endpoint: str, secret: str, bucket: str = DEFAULT_BUCKET
    ) -> None:
        """Configure broker access.

        Args:
            endpoint: Complete endpoint URL or broker base URL.
            secret: Broker bearer secret.
            bucket (optional): Canonical bucket. Defaults to EuroEval/results.

        Raises:
            ValueError:
                If the URL or secret is missing.
        """
        if not endpoint or not secret:
            raise ValueError("Publication lock URL and secret are required")
        if not endpoint.endswith("/publication-lock"):
            endpoint = f"{endpoint.rstrip('/')}/publication-lock"
        self.endpoint = endpoint
        self.secret = secret
        self.bucket = bucket

    @classmethod
    def from_environment(cls, bucket: str = DEFAULT_BUCKET) -> PublicationLock:
        """Create the configured lock adapter, failing closed if unavailable.

        Args:
            bucket (optional): Canonical bucket. Defaults to EuroEval/results.

        Returns:
            Configured lock adapter.
        """
        endpoint = os.environ.get("CANONICAL_PUBLICATION_URL")
        if not endpoint:
            endpoint = os.environ.get(
                "VOLUNTEER_BROKER_URL", "https://euroeval.com/api/worker"
            )
        secret = os.environ.get("CANONICAL_PUBLICATION_SECRET", "")
        return cls(endpoint=endpoint, secret=secret, bucket=bucket)

    def request(self, action: str, owner: str) -> str:
        """Send one broker lock operation and validate its complete response.

        Args:
            action: acquire, assert, release, or status.
            owner: Opaque session owner token.

        Returns:
            Broker lock status.

        Raises:
            ValueError:
                If the action is unsupported.
            PublicationError:
                If the broker request fails or its response is invalid.
        """
        if action not in {"acquire", "assert", "release", "status"}:
            raise ValueError(f"Unsupported publication lock action: {action}")
        body = json.dumps({"action": action, "bucket": self.bucket, "owner": owner})
        request = urllib.request.Request(
            self.endpoint,
            data=body.encode(),
            headers={
                "Authorization": f"Bearer {self.secret}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.loads(response.read())
                break
            except (
                OSError,
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                if attempt == 1:
                    raise PublicationError("Publication lock request failed") from exc
        expected_status = {"acquire": "owned", "assert": "owned", "release": "released"}
        if not isinstance(payload, dict) or payload.get("bucket") != self.bucket:
            raise PublicationError("Invalid publication lock response")
        status = payload.get("status")
        if status != expected_status.get(action, status):
            raise PublicationError("Publication lock operation was not confirmed")
        if status == "busy":
            raise PublicationError("Canonical publication lock is held")
        if action != "status" and payload.get("owner") != owner:
            raise PublicationError("Publication lock owner response did not match")
        if action == "status" and status not in {"owned", "unlocked"}:
            raise PublicationError("Invalid publication lock status")
        return t.cast(str, status)


class CanonicalPublisher:
    """Publish result records with lock enforcement and durable predecessor history."""

    def __init__(
        self,
        api: BucketApi,
        token: str,
        bucket: str = DEFAULT_BUCKET,
        lock: LockAdapter | None = None,
    ) -> None:
        """Create a publisher over a Hugging Face-compatible adapter.

        Args:
            api: Adapter implementing bucket download and mutation methods.
            token: Hugging Face write token.
            bucket (optional): Canonical bucket. Defaults to EuroEval/results.
            lock (optional): Broker adapter; defaults to environment configuration.

        Raises:
            ValueError:
                If the token or lock configuration is invalid.
        """
        if not token:
            raise ValueError("HF token is required")
        self.api = api
        self.token = token
        self.bucket = bucket
        self.lock = lock or PublicationLock.from_environment(bucket=bucket)
        if self.lock.bucket != bucket:
            raise ValueError("Publication lock bucket does not match publisher bucket")

    @contextmanager
    def session(self) -> t.Iterator[PublicationSession]:
        """Acquire an owned publication session and safely release it.

        Yields:
            An explicit publication session.
        """
        session = PublicationSession(publisher=self)
        session.acquire()
        try:
            yield session
        except BaseException:
            if not session.poisoned:
                session.release()
            raise
        else:
            if not session.poisoned:
                session.release()


class PublicationSession:
    """Explicit, lock-owning interface for canonical bucket operations."""

    def __init__(self, publisher: CanonicalPublisher) -> None:
        """Initialise session identity and private recovery receipt.

        Args:
            publisher: Publisher that owns the bucket adapter and credentials.
        """
        self.publisher = publisher
        self.owner = secrets.token_urlsafe(32)
        self.bucket = publisher.bucket
        self.poisoned = False
        self._receipt = _receipt_path(self.owner)
        self._receipt.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    def acquire(self) -> None:
        """Persist recovery details before acquiring the no-TTL lock."""
        _write_receipt(
            self._receipt, self.owner, self.publisher.lock.endpoint, self.bucket
        )
        self.publisher.lock.request(action="acquire", owner=self.owner)

    def delete_empty(self, paths: t.Sequence[str]) -> None:
        """Delete only objects that still exist and are currently empty.

        Args:
            paths: Bucket-relative object paths to inspect.
        """
        self.assert_owned()
        for path in paths:
            current = self._download(path)
            if current == b"":
                self.assert_owned()
                try:
                    self.publisher.api.batch_bucket_files(
                        bucket_id=self.bucket, delete=[path], token=self.publisher.token
                    )
                except BaseException:
                    self.poisoned = True
                    raise

    def _download(self, path: str) -> bytes | None:
        with tempfile.TemporaryDirectory(prefix="euroeval-publish-") as directory:
            local = Path(directory) / "object"
            try:
                self.publisher.api.download_bucket_files(
                    bucket_id=self.bucket,
                    files=[(path, local)],
                    raise_on_missing_files=False,
                    token=self.publisher.token,
                )
            except FileNotFoundError:
                return None
            return local.read_bytes() if local.exists() else None

    def assert_owned(self) -> None:
        """Confirm this session still owns the broker lock."""
        self.publisher.lock.request(action="assert", owner=self.owner)

    def publish(
        self,
        records: t.Sequence[bytes],
        sources: t.Mapping[ResultIdentity, bytes] | None = None,
    ) -> int:
        """Validate, reconcile, archive predecessors, and publish records.

        Args:
            records: Candidate JSON result records.
            sources (optional): Original bytes keyed by identity for sanctioned
                metadata transformations. Defaults to no source records.

        Returns:
            Number of canonical result objects written.

        Raises:
            PublicationError:
                If candidates conflict, are invalid, or cannot be safely written.
        """
        candidates = _validated_candidates(records=records)
        self.assert_owned()
        writes: list[tuple[str, bytes, bytes | None]] = []
        for path, identity, candidate, record in candidates:
            current = self._download(path)
            if current is None:
                writes.append((path, candidate, None))
                continue
            current_record = _decode_record(current, path)
            if is_canary_record(current_record):
                raise PublicationError(f"Canary record found at canonical path {path}")
            current_identity = identity_from_eee_record(current_record)
            if current_identity != identity:
                raise PublicationError(f"Canonical path collision at {path}")
            if record == current_record:
                continue
            newer = dedup_newer_record(current_record, record)
            if newer is current_record:
                equal_rank = dedup_newer_record(record, current_record) is record
                if not equal_rank:
                    continue
                if (
                    sources is None
                    or sources.get(identity) != current
                    or not _metadata_only_transform(current_record, record)
                ):
                    raise PublicationError(f"Equal-ranked result conflict at {path}")
            writes.append((path, candidate, current))

        # Every predecessor is archived and verified before the first overwrite.
        for path, _, previous in writes:
            if previous is None:
                continue
            archive = f"_history/{hashlib.sha256(previous).hexdigest()}.bin"
            archived = self._download(archive)
            if archived is None:
                self.write_verified(path=archive, content=previous, expected=None)
            elif archived != previous:
                raise PublicationError(f"History hash collision at {path}")

        for path, content, previous in writes:
            self.write_verified(path=path, content=content, expected=previous)
        return len(writes)

    def write_verified(self, path: str, content: bytes, expected: bytes | None) -> None:
        """Write only if the actual current object equals expected, then verify.

        Args:
            path: Bucket-relative object path.
            content: Replacement bytes.
            expected: Expected existing bytes, or None when the object must be absent.

        Raises:
            PublicationError:
                If ownership is lost, current bytes differ, or verification fails.
        """
        self.assert_owned()
        if _contains_canary(content):
            raise PublicationError("Canary records cannot be written canonically")
        current = self._download(path)
        if current != expected:
            raise PublicationError(f"Concurrent change detected for {path}")
        self._upload(path=path, content=content)
        try:
            if self._download(path) != content:
                raise PublicationError(
                    f"Uploaded object verification failed for {path}"
                )
        except BaseException:
            self.poisoned = True
            raise

    def _upload(self, path: str, content: bytes) -> None:
        with tempfile.TemporaryDirectory(prefix="euroeval-publish-") as directory:
            local = Path(directory) / "object"
            local.write_bytes(content)
            try:
                self.publisher.api.batch_bucket_files(
                    bucket_id=self.bucket,
                    add=[(local, path)],
                    token=self.publisher.token,
                )
            except BaseException:
                self.poisoned = True
                raise

    def read(self, path: str) -> bytes | None:
        """Read an exact object from the bucket.

        Args:
            path: Bucket-relative object path.

        Returns:
            Object bytes, or None if absent.
        """
        return self._download(path)

    def release(self) -> None:
        """Release only this owner's lock and remove its recovery receipt."""
        self.publisher.lock.request(action="release", owner=self.owner)
        self._receipt.unlink(missing_ok=True)
