"""Serialize durable writes to canonical EuroEval result records.

A publication session owns the broker's no-TTL lock from before its first remote
read until all mutations and verification are complete. Callers must pass the
session explicitly; lock ownership is never inherited by evaluation subprocesses.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import secrets
import sqlite3
import tempfile
import typing as t
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import hf_xet

from euroeval.numeric_model_metadata import positive_context_length, positive_integer

from .contamination_canary import is_canary_record
from .result_conflicts import (
    POLICY_VERSION,
    ConflictChoice,
    ConflictDecision,
    ConflictReason,
    resolve_result_conflict,
)
from .result_identity import ResultIdentity, identity_from_eee_record, identity_to_path

DEFAULT_BUCKET = "EuroEval/results"
_COMPARISON_BATCH_SIZE = 500
logger = logging.getLogger(__name__)


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


def _receipt_path(owner: str) -> Path:
    """Return the private recovery receipt path for an owner token."""
    root = Path(os.environ.get("XDG_CACHE_HOME", tempfile.gettempdir()))
    return root / "euroeval-publication-locks" / f"{owner}.json"


def _semantic_digest(record: dict[str, object]) -> str:
    """Hash canonical JSON semantics independently of formatting.

    Args:
        record: Decoded result record.

    Returns:
        SHA-256 digest of deterministic JSON bytes.
    """
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        details = _model_additional_details(record=record)
        for key, validator in (
            ("num_model_parameters", positive_integer),
            ("vocabulary_size", positive_integer),
            ("max_sequence_length", positive_context_length),
        ):
            if (
                key in details
                and details[key] not in (None, -1, "-1")
                and validator(details[key]) is None
            ):
                raise PublicationError(f"Invalid numeric metadata {key}")
        identity = identity_from_eee_record(record)
        path = identity_to_path(identity).as_posix()
        if path in result and result[path][0] != identity:
            raise PublicationError(f"Distinct result identities collide at {path}")
        if path in result and result[path][1] != content:
            raise PublicationError(f"Multiple candidate records target {path}")
        result[path] = (identity, content, record)
    return [(path, *value) for path, value in result.items()]


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


def is_metadata_only_transform(
    current: dict[str, object], candidate: dict[str, object]
) -> bool:
    """Check that a result change is limited to sanctioned metadata.

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
    for record in (current_copy, candidate_copy):
        for key in ("id", "name", "eval_id", "eval_timestamp"):
            record.pop(key, None)
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
        _normalise_raw_results(record=record)
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


def _normalise_raw_results(record: dict[str, object]) -> None:
    """Decode embedded raw-result JSON so equivalent serialisations compare equal.

    Args:
        record: A copied result record to normalise in place.
    """
    evaluation_library = record.get("eval_library")
    if not isinstance(evaluation_library, dict):
        return
    details = evaluation_library.get("additional_details")
    if not isinstance(details, dict):
        return
    raw_results = details.get("raw_results")
    if isinstance(raw_results, str):
        try:
            details["raw_results"] = json.loads(raw_results)
        except json.JSONDecodeError:
            return


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
        checkpoint_path: Path | None = None,
    ) -> None:
        """Create a publisher over a Hugging Face-compatible adapter.

        Args:
            api: Adapter implementing bucket download and mutation methods.
            token: Hugging Face write token.
            bucket (optional): Canonical bucket. Defaults to EuroEval/results.
            lock (optional): Broker adapter; defaults to environment configuration.
            checkpoint_path (optional): Incremental publication receipt database.
                Defaults to the EuroEval cache directory.

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
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        self.checkpoint_path = checkpoint_path or (
            cache_root / "euroeval" / "canonical-publication.sqlite3"
        )
        self.scope = hashlib.sha256(
            "\0".join(
                (
                    bucket,
                    self.lock.endpoint,
                    str(getattr(api, "endpoint", "https://huggingface.co")),
                    "canonical-publication-v1",
                    hashlib.sha256(token.encode()).hexdigest(),
                )
            ).encode()
        ).hexdigest()
        if self.lock.bucket != bucket:
            raise ValueError("Publication lock bucket does not match publisher bucket")

    def prepare(
        self,
        records: t.Sequence[bytes],
        sources: t.Mapping[ResultIdentity, bytes],
        *,
        decisions: t.Sequence[ConflictDecision] = (),
    ) -> None:
        """Checkpoint valid predecessor evidence before callers mutate local files.

        Args:
            records: Candidate result records that bind the evidence.
            sources: Exact pre-mutation bytes keyed by result identity.
            decisions (optional): Explicit, byte-bound conflict choices.
        """
        with self.session() as session:
            session.prepare(records=records, sources=sources, decisions=decisions)

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

    def _source_authorises_transform(
        self,
        identity: ResultIdentity,
        current: bytes,
        current_record: dict[str, object],
        record: dict[str, object],
        sources: t.Mapping[ResultIdentity, bytes] | None,
    ) -> bool:
        """Check exact predecessor provenance for a metadata-only update.

        Args:
            identity: Candidate result identity.
            current: Exact current remote bytes.
            current_record: Decoded current record.
            record: Candidate decoded record.
            sources: Caller or checkpoint source evidence.

        Returns:
            Whether exact evidence authorises this transform.
        """
        return (
            sources is not None
            and sources.get(identity) == current
            and is_metadata_only_transform(current_record, record)
        )

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

    def prepare(
        self,
        records: t.Sequence[bytes],
        sources: t.Mapping[ResultIdentity, bytes],
        *,
        decisions: t.Sequence[ConflictDecision] = (),
    ) -> None:
        """Persist applicable source evidence before any local mutation.

        Args:
            records: Candidate result records.
            sources: Exact predecessor bytes keyed by identity.
            decisions (optional): Explicit, byte-bound conflict choices.

        Raises:
            PublicationError: If a supplied conflict decision is stale or invalid.
        """
        candidates = _validated_candidates(records=records)
        self.assert_owned()
        validated = self._validated_sources(candidates=candidates, sources=sources)
        proofs = self._validated_decisions(
            candidates=candidates, sources=validated, decisions=decisions
        )
        if proofs:
            proof_paths = [identity_to_path(identity).as_posix() for identity in proofs]
            current = self._download_many(paths=proof_paths)
            candidate_bytes = {
                identity: content for _, identity, content, _ in candidates
            }
            for identity, proof in proofs.items():
                path = identity_to_path(identity).as_posix()
                remote_predecessor = current.get(path)
                candidate = candidate_bytes[identity]
                remote_hash = (
                    hashlib.sha256(remote_predecessor).hexdigest()
                    if remote_predecessor is not None
                    else None
                )
                if remote_hash not in {proof[1], hashlib.sha256(candidate).hexdigest()}:
                    raise PublicationError(
                        f"Conflict decision predecessor changed at {path}"
                    )
        if validated:
            self._store_sources(sources=validated)
        if proofs:
            self._store_decisions(proofs=proofs)

    def _download_many(self, paths: list[str]) -> dict[str, bytes | None]:
        """Download exact current bucket objects in bounded batches.

        Args:
            paths:
                Bucket-relative object paths to download.

        Returns:
            Current bytes for every requested path; absent objects map to None.

        """
        objects: dict[str, bytes | None] = {}
        if not paths:
            return objects

        with tempfile.TemporaryDirectory(prefix="euroeval-publish-") as directory:
            root = Path(directory)
            for start in range(0, len(paths), _COMPARISON_BATCH_SIZE):
                batch = paths[start : start + _COMPARISON_BATCH_SIZE]
                destinations = {
                    path: root / str(start + offset)
                    for offset, path in enumerate(batch)
                }
                self.publisher.api.download_bucket_files(
                    bucket_id=self.bucket,
                    files=[(path, destinations[path]) for path in batch],
                    raise_on_missing_files=False,
                    token=self.publisher.token,
                )
                objects.update(
                    {
                        path: destination.read_bytes() if destination.exists() else None
                        for path, destination in destinations.items()
                    }
                )
                compared = min(start + len(batch), len(paths))
                logger.info(
                    "Compared %s of %s remote result files.",
                    f"{compared:,}",
                    f"{len(paths):,}",
                )
        return objects

    def _store_decisions(
        self, proofs: t.Mapping[ResultIdentity, tuple[str, str, str, str]]
    ) -> None:
        """Persist explicit proofs in the publisher's isolated scope.

        Args:
            proofs: Candidate/source fingerprints, policy version and selected side.
        """
        with self._database() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (self.publisher.scope, json.dumps(identity), *proof)
                    for identity, proof in proofs.items()
                ],
            )

    @contextmanager
    def _database(self) -> t.Iterator[sqlite3.Connection]:
        """Open and initialise the scoped incremental publication database.

        Yields:
            A configured SQLite connection, closed after the transaction.
        """
        self.publisher.checkpoint_path.parent.mkdir(
            mode=0o700, parents=True, exist_ok=True
        )
        self.publisher.checkpoint_path.parent.chmod(0o700)
        connection = sqlite3.connect(self.publisher.checkpoint_path, timeout=30)
        os.chmod(self.publisher.checkpoint_path, 0o600)
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS receipts "
                "(scope TEXT NOT NULL, path TEXT NOT NULL, fingerprint TEXT NOT NULL, "
                "digest TEXT NOT NULL, PRIMARY KEY(scope, path))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS sources "
                "(scope TEXT NOT NULL, identity TEXT NOT NULL, content BLOB NOT NULL, "
                "PRIMARY KEY(scope, identity))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS decisions "
                "(scope TEXT NOT NULL, identity TEXT NOT NULL, "
                "candidate_sha256 TEXT NOT NULL, current_sha256 TEXT NOT NULL, "
                "policy_version TEXT NOT NULL, choice TEXT NOT NULL, "
                "PRIMARY KEY(scope, identity))"
            )
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _store_sources(self, sources: t.Mapping[ResultIdentity, bytes]) -> None:
        """Durably retain exact caller-provided predecessor evidence.

        Args:
            sources: Source bytes keyed by result identity.
        """
        with self._database() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO sources VALUES (?, ?, ?)",
                [
                    (self.publisher.scope, json.dumps(identity), content)
                    for identity, content in sources.items()
                ],
            )

    def _validated_decisions(
        self,
        candidates: list[tuple[str, ResultIdentity, bytes, dict[str, object]]],
        sources: t.Mapping[ResultIdentity, bytes],
        decisions: t.Sequence[ConflictDecision],
    ) -> dict[ResultIdentity, tuple[str, str, str, str]]:
        """Accept only explicit decisions bound to source and candidate bytes.

        Args:
            candidates: Validated candidate records.
            sources: Exact caller-provided predecessor records.
            decisions: Choices supplied by the collector.

        Returns:
            Publisher-scope-independent proof tuples indexed by result identity.

        Raises:
            PublicationError: If any supplied proof is stale or non-manual.
        """
        candidates_by_identity = {
            identity: (content, record) for _, identity, content, record in candidates
        }
        proofs: dict[ResultIdentity, tuple[str, str, str, str]] = {}
        for decision in decisions:
            evidence = decision.evidence
            identity = evidence.identity
            candidate = candidates_by_identity.get(identity)
            source = sources.get(identity)
            if (
                candidate is None
                or source is None
                or decision.reason is not ConflictReason.EXPLICIT_CHOICE
                or decision.selected
                not in (ConflictChoice.STORED, ConflictChoice.INCOMING)
                or evidence.policy_version != POLICY_VERSION
                or hashlib.sha256(candidate[0]).hexdigest() != evidence.candidate_sha256
                or hashlib.sha256(source).hexdigest() != evidence.current_sha256
                or evidence.choice != decision.selected
                or identity_from_eee_record(_decode_record(source, "decision source"))
                != identity
            ):
                raise PublicationError(
                    "Conflict decision is not bound to these records"
                )
            proof = (
                evidence.candidate_sha256,
                evidence.current_sha256,
                evidence.policy_version,
                decision.selected.value,
            )
            if identity in proofs and proofs[identity] != proof:
                raise PublicationError("Multiple conflict decisions target one result")
            proofs[identity] = proof
        return proofs

    def _validated_sources(
        self,
        candidates: list[tuple[str, ResultIdentity, bytes, dict[str, object]]],
        sources: t.Mapping[ResultIdentity, bytes] | None,
    ) -> dict[ResultIdentity, bytes]:
        """Keep only valid, candidate-identity-bound predecessor evidence.

        Args:
            candidates: Validated candidate records.
            sources: Caller-provided predecessor bytes.

        Returns:
            Safe source evidence keyed by candidate identity.
        """
        candidates_by_identity = {identity for _, identity, _, _ in candidates}
        validated: dict[ResultIdentity, bytes] = {}
        for identity, content in (sources or {}).items():
            if identity not in candidates_by_identity or _contains_canary(content):
                continue
            try:
                record = _decode_record(content, identity_to_path(identity).as_posix())
                if identity not in candidates_by_identity:
                    continue
                if identity_from_eee_record(record) == identity:
                    validated[identity] = content
            except (PublicationError, ValueError, KeyError, TypeError):
                continue
        return validated

    def publish(
        self,
        records: t.Sequence[bytes],
        sources: t.Mapping[ResultIdentity, bytes] | None = None,
        *,
        decisions: t.Sequence[ConflictDecision] = (),
    ) -> int:
        """Validate, reconcile, archive predecessors, and publish records.

        Args:
            records: Candidate JSON result records.
            sources (optional): Original bytes keyed by identity for sanctioned
                metadata transformations. Defaults to no source records.
            decisions (optional): Explicit, byte-bound conflict choices.

        Returns:
            Number of canonical result objects written.

        Raises:
            PublicationError:
                If candidates conflict, are invalid, or cannot be safely written.
        """
        candidates = _validated_candidates(records=records)
        self.assert_owned()
        validated_sources = self._validated_sources(
            candidates=candidates, sources=sources
        )
        proofs = self._validated_decisions(
            candidates=candidates, sources=validated_sources, decisions=decisions
        )
        if validated_sources:
            self._store_sources(sources=validated_sources)
        if proofs:
            self._store_decisions(proofs=proofs)
        decisions_by_identity = self._decisions_with_pending(proofs=proofs)
        sources = self._sources_with_pending(sources=validated_sources)
        logger.info(
            "Fetching fresh remote metadata for %s candidate result files...",
            f"{len(candidates):,}",
        )
        remote_fingerprints = self._remote_fingerprints(candidates=candidates)
        cached = self._cached_receipts()
        warm_paths = {
            path
            for path, _, _, record in candidates
            if remote_fingerprints is not None
            and cached.get(path)
            == (remote_fingerprints.get(path), _semantic_digest(record))
        }
        local_fingerprints = (
            self._local_fingerprints(
                candidates=candidates, sources=sources, skip_paths=warm_paths
            )
            if remote_fingerprints is not None
            else {}
        )
        source_paths = {
            identity_to_path(identity).as_posix(): content
            for identity, content in (sources or {}).items()
        }
        trusted = {
            path
            for path, _, candidate, record in candidates
            if remote_fingerprints is not None
            and remote_fingerprints.get(path) is not None
            and (
                remote_fingerprints[path] == local_fingerprints.get(path)
                or cached.get(path)
                == (remote_fingerprints[path], _semantic_digest(record))
                or (
                    path in source_paths
                    and remote_fingerprints[path]
                    == local_fingerprints.get(f"source:{path}")
                    and _decode_record(source_paths[path], path) == record
                )
            )
        }
        source_proven = {
            path: content
            for path, content in source_paths.items()
            if remote_fingerprints is not None
            and path in remote_fingerprints
            and remote_fingerprints[path] == local_fingerprints.get(f"source:{path}")
        }
        body_comparison_paths = [
            path
            for path, _, _, _ in candidates
            if path not in trusted
            and path not in source_proven
            and (remote_fingerprints is None or path in remote_fingerprints)
        ]
        logger.info(
            "Fresh metadata verified %s candidate files; comparing bodies for "
            "%s of %s.",
            f"{len(trusted | source_proven.keys()):,}",
            f"{len(body_comparison_paths):,}",
            f"{len(candidates):,}",
        )
        current_objects = self._download_many(paths=body_comparison_paths)
        if remote_fingerprints is not None:
            current_objects.update(
                {
                    path: None
                    for path, _, _, _ in candidates
                    if path not in remote_fingerprints
                }
            )
        current_objects.update(source_proven)
        writes, verified_receipts = self._reconcile_candidates(
            candidates=candidates,
            trusted=trusted,
            current_objects=current_objects,
            remote_fingerprints=remote_fingerprints,
            sources=sources,
            decisions=decisions_by_identity,
        )

        if writes:
            logger.info(
                "Remote comparison complete: %s writes required; "
                "archiving predecessors...",
                f"{len(writes):,}",
            )
        else:
            logger.info("Remote comparison complete: no writes required.")
        # Every predecessor is archived and verified before the first overwrite.
        predecessors = [
            (path, previous) for path, _, previous in writes if previous is not None
        ]
        archives = {
            f"_history/{hashlib.sha256(previous).hexdigest()}.bin": previous
            for _, previous in predecessors
        }
        archive_paths = list(archives)
        archive_fingerprints = self._hash_bytes(
            [(path, content) for path, content in archives.items()]
        )
        archived_metadata = (
            self._bucket_fingerprints(paths=archive_paths) if archive_paths else {}
        )
        if archived_metadata is None:
            archived = self._download_many(paths=archive_paths)
        else:
            uncertain_paths = [
                path
                for path in archive_paths
                if path in archived_metadata
                and archived_metadata[path] != archive_fingerprints[path]
            ]
            archived = self._download_many(paths=uncertain_paths)
            archived.update(
                {path: None for path in archive_paths if path not in archived_metadata}
            )
            archived.update(
                {
                    path: archives[path]
                    for path in archive_paths
                    if archived_metadata.get(path) == archive_fingerprints[path]
                }
            )
        missing_archives: list[tuple[str, bytes, bytes | None]] = []
        for path, content in archives.items():
            if archived[path] is None:
                missing_archives.append((path, content, None))
            elif archived[path] != content:
                raise PublicationError(f"History hash collision at {path}")
        self._write_batches(writes=missing_archives)

        self._write_batches(
            writes=writes,
            receipts={
                path: (local_fingerprints[path], _semantic_digest(record))
                for path, _, _, record in candidates
                if path in local_fingerprints
            },
        )
        if remote_fingerprints is not None:
            written_paths = {path for path, _, _ in writes}
            for path, _, _, record in candidates:
                if path in written_paths:
                    # Verified upload chunks already have durable receipts.
                    continue
                if path in trusted and remote_fingerprints.get(path) is not None:
                    verified_receipts[path] = (
                        remote_fingerprints[path],
                        _semantic_digest(record),
                    )
                elif path not in current_objects:
                    verified_receipts.pop(path, None)
            self._store_receipts(
                receipts={
                    path: receipt
                    for path, receipt in verified_receipts.items()
                    if cached.get(path) != receipt
                }
            )
        self._clear_sources(identities={identity for _, identity, _, _ in candidates})
        self._clear_decisions(identities={identity for _, identity, _, _ in candidates})
        return len(writes)

    def _bucket_fingerprints(self, paths: list[str]) -> dict[str, str] | None:
        """Fetch complete fresh metadata, falling back when it is unusable.

        Args:
            paths: Bucket-relative paths to query.

        Returns:
            Fingerprints for present paths, or None when metadata cannot be trusted.
        """
        method = getattr(self.publisher.api, "get_bucket_paths_info", None)
        if method is None:
            return None
        requested_paths = set(paths)
        try:
            entries = [
                entry
                for start in range(0, len(paths), _COMPARISON_BATCH_SIZE)
                for entry in method(
                    self.bucket,
                    paths[start : start + _COMPARISON_BATCH_SIZE],
                    token=self.publisher.token,
                )
            ]
            fingerprints: dict[str, str] = {}
            for entry in entries:
                path = getattr(entry, "path", None)
                fingerprint = getattr(entry, "xet_hash", None)
                if not isinstance(path, str) or path not in requested_paths:
                    return None
                if not isinstance(fingerprint, str) or not fingerprint:
                    return None
                if path in fingerprints:
                    return None
                fingerprints[path] = fingerprint
            return fingerprints
        except Exception:
            return None

    def _cached_receipts(self) -> dict[str, tuple[str, str]]:
        """Load successful semantic publication receipts.

        Returns:
            Path-keyed remote fingerprint and semantic digest pairs.
        """
        with self._database() as connection:
            rows = connection.execute(
                "SELECT path, fingerprint, digest FROM receipts WHERE scope = ?",
                (self.publisher.scope,),
            )
            return {path: (fingerprint, digest) for path, fingerprint, digest in rows}

    def _clear_decisions(self, identities: set[ResultIdentity]) -> None:
        """Remove completed manual proofs from the retry checkpoint.

        Args:
            identities: Result identities that completed reconciliation.
        """
        if not identities:
            return
        with self._database() as connection:
            connection.executemany(
                "DELETE FROM decisions WHERE scope = ? AND identity = ?",
                [
                    (self.publisher.scope, json.dumps(identity))
                    for identity in identities
                ],
            )

    def _clear_sources(self, identities: set[ResultIdentity]) -> None:
        """Remove retry evidence after its candidate has converged.

        Args:
            identities: Candidate identities that completed publication.
        """
        if not identities:
            return
        with self._database() as connection:
            connection.executemany(
                "DELETE FROM sources WHERE scope = ? AND identity = ?",
                [
                    (self.publisher.scope, json.dumps(identity))
                    for identity in identities
                ],
            )

    def _decisions_with_pending(
        self, proofs: t.Mapping[ResultIdentity, tuple[str, str, str, str]]
    ) -> dict[ResultIdentity, tuple[str, str, str, str]]:
        """Load durable proofs, preferring newly validated caller proofs.

        Args:
            proofs: Newly validated decisions.

        Returns:
            All scope-bound pending proofs.
        """
        combined: dict[ResultIdentity, tuple[str, str, str, str]] = {}
        with self._database() as connection:
            rows = connection.execute(
                "SELECT identity, candidate_sha256, current_sha256, policy_version, "
                "choice FROM decisions WHERE scope = ?",
                (self.publisher.scope,),
            )
            for identity, candidate_sha, current_sha, version, choice in rows:
                combined[t.cast(ResultIdentity, tuple(json.loads(identity)))] = (
                    candidate_sha,
                    current_sha,
                    version,
                    choice,
                )
        combined.update(proofs)
        return combined

    def _hash_bytes(self, contents: list[tuple[str, bytes]]) -> dict[str, str]:
        """Hash arbitrary object bytes in bounded bulk Xet calls.

        Args:
            contents: Labels and exact object bytes to hash.

        Returns:
            Xet fingerprints keyed by label.

        Raises:
            PublicationError:
                If Xet returns an incomplete fingerprint batch.
        """
        fingerprints: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="euroeval-xet-hash-") as directory:
            root = Path(directory)
            for start in range(0, len(contents), _COMPARISON_BATCH_SIZE):
                batch = contents[start : start + _COMPARISON_BATCH_SIZE]
                paths = []
                for index, (_, content) in enumerate(batch):
                    local = root / f"{start + index}"
                    local.write_bytes(content)
                    paths.append(str(local))
                infos = hf_xet.hash_files(paths)
                if len(infos) != len(batch):
                    raise PublicationError("Incomplete local Xet fingerprint results")
                fingerprints.update(
                    (label, info.hash)
                    for (label, _), info in zip(batch, infos, strict=True)
                )
        return fingerprints

    def _local_fingerprints(
        self,
        candidates: list[tuple[str, ResultIdentity, bytes, dict[str, object]]],
        sources: t.Mapping[ResultIdentity, bytes] | None = None,
        skip_paths: set[str] | None = None,
    ) -> dict[str, str]:
        """Calculate local Xet fingerprints for candidate and source bytes.

        Args:
            candidates: Validated publication candidates.
            sources (optional): Validated predecessor bytes.
            skip_paths (optional): Candidate paths already proven by fresh receipts.

        Returns:
            Xet fingerprints keyed by canonical path or source label.

        """
        fingerprints: dict[str, str] = {}
        skip_paths = skip_paths or set()
        with tempfile.TemporaryDirectory(prefix="euroeval-xet-hash-") as directory:
            root = Path(directory)
            batch_paths: list[str] = []
            batch_labels: list[str] = []

            def hash_batch() -> None:
                if not batch_paths:
                    return
                paths = batch_paths.copy()
                try:
                    infos = hf_xet.hash_files(paths)
                    if len(infos) != len(paths):
                        raise PublicationError(
                            "Incomplete local Xet fingerprint results"
                        )
                    fingerprints.update(
                        (label, info.hash)
                        for label, info in zip(batch_labels, infos, strict=True)
                    )
                finally:
                    for path in paths:
                        Path(path).unlink(missing_ok=True)
                    batch_paths.clear()
                    batch_labels.clear()

            inputs = (
                (path, content)
                for path, _, content, _ in candidates
                if path not in skip_paths
            )
            inputs = iter(inputs)
            for index, (path, content) in enumerate(inputs):
                local = root / str(index)
                local.write_bytes(content)
                batch_paths.append(str(local))
                batch_labels.append(path)
                if len(batch_paths) == _COMPARISON_BATCH_SIZE:
                    hash_batch()
            for index, (identity, content) in enumerate((sources or {}).items()):
                path = identity_to_path(identity).as_posix()
                if path in skip_paths:
                    continue
                local = root / f"source-{index}"
                local.write_bytes(content)
                batch_paths.append(str(local))
                batch_labels.append(f"source:{path}")
                if len(batch_paths) == _COMPARISON_BATCH_SIZE:
                    hash_batch()
            hash_batch()
        return fingerprints

    def _reconcile_candidates(
        self,
        candidates: list[tuple[str, ResultIdentity, bytes, dict[str, object]]],
        trusted: set[str],
        current_objects: dict[str, bytes | None],
        remote_fingerprints: dict[str, str] | None,
        sources: t.Mapping[ResultIdentity, bytes] | None,
        decisions: t.Mapping[ResultIdentity, tuple[str, str, str, str]],
    ) -> tuple[list[tuple[str, bytes, bytes | None]], dict[str, tuple[str, str]]]:
        """Reconcile untrusted candidates with current remote bytes.

        Args:
            candidates: Validated result candidates.
            trusted: Paths proven equivalent by fresh fingerprints or receipts.
            current_objects: Downloaded current bytes for untrusted candidates.
            remote_fingerprints: Fresh metadata for existing remote paths.
            sources: Exact predecessor bytes retained for retries.
            decisions: Stored byte-bound manual choices.

        Returns:
            Pending writes and newly verified semantic receipts.

        Raises:
            PublicationError:
                If a candidate conflicts with current canonical state.
        """
        writes: list[tuple[str, bytes, bytes | None]] = []
        receipts: dict[str, tuple[str, str]] = {}
        equivalent: list[tuple[str, ResultIdentity, bytes, dict[str, object]]] = []
        for path, identity, candidate, record in candidates:
            if path in trusted:
                continue
            current = current_objects[path]
            if current is None:
                if identity in decisions:
                    raise PublicationError(
                        f"Conflict decision predecessor is missing at {path}"
                    )
                if remote_fingerprints is not None and path in remote_fingerprints:
                    raise PublicationError(
                        f"Remote object missing after metadata: {path}"
                    )
                writes.append((path, candidate, None))
                continue
            current_record = _decode_record(current, path)
            self._validate_current_record(
                current_record=current_record, path=path, identity=identity
            )
            if record == current_record:
                fingerprint = (
                    remote_fingerprints.get(path)
                    if remote_fingerprints is not None
                    else None
                )
                if fingerprint is not None:
                    equivalent.append((path, identity, current, record))
                continue
            proof = decisions.get(identity)
            if proof is not None:
                candidate_sha, current_sha, version, choice = proof
                if (
                    candidate_sha != hashlib.sha256(candidate).hexdigest()
                    or current_sha != hashlib.sha256(current).hexdigest()
                    or version != POLICY_VERSION
                ):
                    raise PublicationError(
                        f"Stored conflict proof no longer matches {path}"
                    )
                if choice == ConflictChoice.STORED.value:
                    continue
                if choice != ConflictChoice.INCOMING.value:
                    raise PublicationError(f"Invalid stored conflict choice for {path}")
                writes.append((path, candidate, current))
                continue
            if (
                is_metadata_only_transform(current_record, record)
                and (sources or {}).get(identity) != current
            ):
                raise PublicationError(
                    f"Unresolved metadata transformation without exact source at {path}"
                )
            decision = resolve_result_conflict(
                current=current_record,
                candidate=record,
                current_bytes=current,
                candidate_bytes=candidate,
            )
            if decision.selected is None:
                raise PublicationError(f"Unresolved result conflict at {path}")
            if decision.selected is ConflictChoice.STORED:
                continue
            writes.append((path, candidate, current))
        fingerprints = self._local_fingerprints(candidates=equivalent)
        for path, _, _, record in equivalent:
            fingerprint = fingerprints[path]
            if (
                remote_fingerprints is not None
                and remote_fingerprints.get(path) == fingerprint
            ):
                receipts[path] = (fingerprint, _semantic_digest(record))
        return writes, receipts

    def _validate_current_record(
        self, current_record: dict[str, object], path: str, identity: ResultIdentity
    ) -> None:
        """Reject unsafe bytes at a canonical result path.

        Args:
            current_record: Decoded remote JSON record.
            path: Canonical path containing the object.
            identity: Expected result identity.

        Raises:
            PublicationError:
                If remote object is canary evidence or a path collision.
        """
        if is_canary_record(current_record):
            raise PublicationError(f"Canary record found at canonical path {path}")
        if identity_from_eee_record(current_record) != identity:
            raise PublicationError(f"Canonical path collision at {path}")

    def _remote_fingerprints(
        self, candidates: list[tuple[str, ResultIdentity, bytes, dict[str, object]]]
    ) -> dict[str, str] | None:
        """Fetch fresh remote Xet fingerprints for candidate paths.

        Args:
            candidates: Validated publication candidates.

        Returns:
            Fingerprints for existing candidates, or None when metadata is unusable.
        """
        return self._bucket_fingerprints(paths=[path for path, _, _, _ in candidates])

    def _sources_with_pending(
        self, sources: t.Mapping[ResultIdentity, bytes] | None
    ) -> t.Mapping[ResultIdentity, bytes] | None:
        """Combine current source evidence with durable retry evidence.

        Args:
            sources: Current source bytes, if supplied.

        Returns:
            Exact source evidence keyed by result identity.
        """
        combined: dict[ResultIdentity, bytes] = {}
        with self._database() as connection:
            rows = connection.execute(
                "SELECT identity, content FROM sources WHERE scope = ?",
                (self.publisher.scope,),
            )
            for identity, content in rows:
                key = tuple(json.loads(identity))
                combined[t.cast(ResultIdentity, key)] = content
        for identity, content in (sources or {}).items():
            combined.setdefault(identity, content)
        return combined or None

    def _store_receipts(self, receipts: t.Mapping[str, tuple[str, str]]) -> None:
        """Persist verified remote receipts in one SQLite transaction.

        Args:
            receipts: Path-keyed Xet fingerprint and semantic digest.
        """
        if not receipts:
            return
        with self._database() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO receipts VALUES (?, ?, ?, ?)",
                [
                    (self.publisher.scope, path, fingerprint, digest)
                    for path, (fingerprint, digest) in receipts.items()
                ],
            )

    def _write_batches(
        self,
        writes: list[tuple[str, bytes, bytes | None]],
        receipts: t.Mapping[str, tuple[str, str]] | None = None,
    ) -> None:
        """Apply bounded nontransactional uploads with exact pre/postconditions.

        Args:
            writes: Path, replacement bytes and expected predecessor tuples.
            receipts (optional): Candidate receipts to persist after verified batches.

        Raises:
            PublicationError:
                If ownership, preconditions or output verification fail.
        """
        receipts = receipts or {}
        for start in range(0, len(writes), _COMPARISON_BATCH_SIZE):
            batch = writes[start : start + _COMPARISON_BATCH_SIZE]
            paths = [path for path, _, _ in batch]
            expected_fingerprints = self._hash_bytes(
                [
                    (f"expected:{path}", expected)
                    for path, _, expected in batch
                    if expected is not None
                ]
            )
            intended_fingerprints = {
                path: receipts[path][0] for path in paths if path in receipts
            }
            intended_fingerprints.update(
                self._hash_bytes(
                    [
                        (path, content)
                        for path, content, _ in batch
                        if path not in intended_fingerprints
                    ]
                )
            )
            current_fingerprints = self._bucket_fingerprints(paths=paths)
            exact_precondition = current_fingerprints is None
            if current_fingerprints is not None:
                for path, _, expected in batch:
                    actual = current_fingerprints.get(path)
                    expected_hash = expected_fingerprints.get(f"expected:{path}")
                    if expected is None and actual is not None:
                        raise PublicationError(f"Concurrent change detected for {path}")
                    if expected is not None and actual != expected_hash:
                        exact_precondition = True
                        break
            if exact_precondition:
                current = self._download_many(paths=paths)
                for path, _, expected in batch:
                    if current[path] != expected:
                        raise PublicationError(f"Concurrent change detected for {path}")
            with tempfile.TemporaryDirectory(prefix="euroeval-publish-") as directory:
                uploads: list[tuple[Path, str]] = []
                for index, (path, content, _) in enumerate(batch):
                    local = Path(directory) / str(index)
                    local.write_bytes(content)
                    uploads.append((local, path))
                self.assert_owned()
                # The bucket has no CAS or transaction: any uncertain mutation
                # retains the no-TTL broker lock for explicit operator recovery.
                try:
                    self.publisher.api.batch_bucket_files(
                        bucket_id=self.bucket, add=uploads, token=self.publisher.token
                    )
                    verified = self._bucket_fingerprints(paths=paths)
                    if verified is None:
                        actual = self._download_many(paths=paths)
                        if any(actual[path] != content for path, content, _ in batch):
                            raise PublicationError(
                                "Uploaded object verification failed"
                            )
                    elif any(
                        verified.get(path) != intended_fingerprints[path]
                        for path in paths
                    ):
                        actual = self._download_many(paths=paths)
                        if any(actual[path] != content for path, content, _ in batch):
                            raise PublicationError(
                                "Uploaded object verification failed"
                            )
                    self.assert_owned()
                    self._store_receipts(
                        receipts={
                            path: (intended_fingerprints[path], receipts[path][1])
                            for path in paths
                            if path in receipts
                        }
                    )
                except BaseException:
                    self.poisoned = True
                    raise
            logger.info(
                "Verified %s of %s uploaded files.", start + len(batch), len(writes)
            )

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
