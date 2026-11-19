"""Tests for the shared canonical publication interface."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import stat
import tempfile
import time
import typing as t
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import hf_xet
import pytest

from leaderboards.cache import Cache
from leaderboards.canonical_publication import (
    CanonicalPublisher,
    PublicationError,
    PublicationLock,
)
from leaderboards.constants import TRAINED_FROM_SCRATCH_PATTERNS
from leaderboards.model_metadata import add_missing_entries
from leaderboards.result_conflicts import (
    ConflictChoice,
    ConflictReason,
    resolve_result_conflict,
)
from leaderboards.result_identity import identity_from_eee_record, identity_to_path


@pytest.mark.parametrize("mode", ["stale", "malformed"])
def test_bulk_metadata_cannot_seed_false_receipts(tmp_path: Path, mode: str) -> None:
    """Malformed metadata aborts; stale metadata forces exact body comparisons."""
    api = BulkApi(lock=CountingLock())
    records = bulk_records(count=501)
    for raw in records:
        path = identity_to_path(identity_from_eee_record(json.loads(raw))).as_posix()
        api.objects[path] = raw
    api.metadata_mode = mode
    publisher = bulk_publisher(tmp_path=tmp_path, api=api)
    with publisher.session() as session:
        assert session.publish(records=records) == 0
    with sqlite3.connect(publisher.checkpoint_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM receipts").fetchone() == (0,)
    assert api.upload_calls == 0
    assert api.downloaded_bytes > 0


def bulk_records(
    count: int, version: str = "1.0.0", *, later_evaluation: bool = False
) -> list[bytes]:
    """Build distinct result identities for bulk publication tests.

    Args:
        count: Number of records.
        version (optional): Evaluation version.
        later_evaluation (optional): Whether to record a later evaluation score.

    Returns:
        Encoded public result records.
    """
    records = []
    for index in range(count):
        record = json.loads(_record(version=version))
        record["model_info"]["id"] = f"org/bulk-{index}"
        if later_evaluation:
            record["eval_timestamp"] = "2026-01-02T00:00:00Z"
            record["evaluation_results"] = {"score": 0.6}
        records.append(json.dumps(record).encode())
    return records


def _record(
    *,
    version: str = "1.0.0",
    score: float = 0.5,
    timestamp: str = "2026-01-01T00:00:00Z",
) -> bytes:
    """Build a minimal public result record.

    Returns:
        JSON encoded result record.
    """
    return json.dumps(
        {
            "model_info": {"id": "org/model"},
            "eval_library": {
                "version": version,
                "additional_details": {
                    "dataset": "task",
                    "few_shot": False,
                    "validation_split": False,
                },
            },
            "retrieved_timestamp": timestamp,
            "eval_timestamp": timestamp,
            "evaluation_results": {"score": score},
            "score": score,
        }
    ).encode()


def test_bulk_precondition_mismatch_never_overwrites(tmp_path: Path) -> None:
    """A body changed between initial comparison and upload cannot be overwritten."""
    api = BulkApi(lock=CountingLock())
    records = bulk_records(count=501)
    path = identity_to_path(identity_from_eee_record(json.loads(records[0]))).as_posix()
    metadata_calls = 0
    metadata = api.get_bucket_paths_info

    def race_metadata(bucket_id: str, paths: list[str], *, token: str) -> list[object]:
        """Change a predecessor before the upload precondition lookup.

        Returns:
            The fresh metadata response after the mutation.
        """
        nonlocal metadata_calls
        metadata_calls += 1
        if metadata_calls == 3:
            api.objects[path] = _record(version="9.0.0")
        return metadata(bucket_id=bucket_id, paths=paths, token=token)

    publisher = bulk_publisher(tmp_path=tmp_path, api=api)
    with patch.object(api, "get_bucket_paths_info", side_effect=race_metadata):
        with pytest.raises(PublicationError, match="Concurrent change"):
            with publisher.session() as session:
                session.publish(records=records)
    assert api.upload_calls == 0
    assert api.lock.owner is None


@pytest.mark.parametrize("count", [1, 499, 500, 501, 1000, 10_000])
def test_bulk_publication_request_budget(tmp_path: Path, count: int) -> None:
    """Remote work is O(batches), including ownership and exact verification."""
    lock = CountingLock()
    api = BulkApi(lock=lock)
    publisher = bulk_publisher(tmp_path=tmp_path, api=api)
    records = bulk_records(count=count)
    with publisher.session() as session:
        assert session.publish(records=records) == count
    batches = (count + 499) // 500
    assert api.upload_calls == batches
    assert len(api.events) == 4 * batches
    assert lock.calls == 3 + 2 * batches
    assert max(len(paths) for _, paths in api.events) <= 500
    assert len(api.objects) == count
    assert api.downloaded_bytes == 0
    assert api.downloaded_files == 0
    assert (len(api.events) + lock.calls) * 2 < 15 * 60
    with sqlite3.connect(publisher.checkpoint_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM receipts").fetchone() == (
            count,
        )
    api.events.clear()
    lock.calls = 0
    # Reconstructed publisher must still fetch fresh metadata, but no bodies.
    with bulk_publisher(tmp_path=tmp_path, api=api).session() as session:
        assert session.publish(records=records) == 0
    assert len(api.events) == batches
    assert all(operation == "metadata" for operation, _ in api.events)
    assert lock.calls == 3


@pytest.mark.parametrize("count", [1, 10_000])
def test_bulk_sleep_adapter_comparison(
    tmp_path: Path, count: int, caplog: pytest.LogCaptureFixture
) -> None:
    """Measure fakes with request latency; the model is not a wall-time guarantee."""
    delay = 0.001
    lock = CountingLock(delay=delay)
    api = BulkApi(lock=lock, delay=delay)
    records = bulk_records(count=count)
    started = time.perf_counter()
    with bulk_publisher(tmp_path=tmp_path, api=api).session() as session:
        assert session.publish(records=records) == count
    bulk_seconds = time.perf_counter() - started
    bulk_requests = len(api.events) + lock.calls
    serial_lock = CountingLock(delay=delay)
    serial_api = BulkApi(lock=serial_lock, delay=delay)
    started = time.perf_counter()
    with bulk_publisher(
        tmp_path=tmp_path / "serial", api=serial_api
    ).session() as session:
        for content in records:
            path = identity_to_path(
                identity_from_eee_record(json.loads(content))
            ).as_posix()
            session.write_verified(path=path, content=content, expected=None)
    serial_seconds = time.perf_counter() - started
    serial_requests = len(serial_api.events) + serial_lock.calls
    assert serial_requests == 4 * count + 2
    assert bulk_requests == 6 * ((count + 499) // 500) + 3
    assert api.downloaded_bytes == 0
    assert api.downloaded_files == 0
    assert bulk_requests * 2 < 900
    caplog.set_level(logging.INFO)
    logging.getLogger(__name__).info(
        "Sleep adapter %s records: bulk %s requests %.3fs; serial %s requests %.3fs; "
        "2s/request model: bulk %ss, serial %ss (not a live timing guarantee)",
        count,
        bulk_requests,
        bulk_seconds,
        serial_requests,
        serial_seconds,
        bulk_requests * 2,
        serial_requests * 2,
    )


def test_canary_is_rejected_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Private canary evidence cannot enter the public object store."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, _ = _publisher(tmp_path)
    canary = json.loads(_record())
    canary["eval_library"]["additional_details"]["dataset"] = "contamination-canary"
    with publisher.session() as session:
        with pytest.raises(PublicationError, match="Canary"):
            session.publish(records=[json.dumps(canary).encode()])
    assert api.objects == {}


class MemoryApi:
    """In-memory subset of the Hugging Face bucket interface."""

    def __init__(self) -> None:
        """Initialise an empty object store."""
        self.objects: dict[str, bytes] = {}
        self.upload_calls = 0
        self.fail_upload = False
        self.interrupt_after_write = False
        self.fail_name: str | None = None
        self.fail_download_batch: int | None = None
        self.mutate_on_download: dict[str, bytes] = {}
        self.download_batches: list[list[str]] = []

    def batch_bucket_files(
        self,
        bucket_id: str,
        add: list[tuple[Path, str]] | None = None,
        delete: list[str] | None = None,
        token: str = "",
    ) -> None:
        """Apply bucket object writes or removals.

        Raises:
            OSError:
                If the adapter is configured to simulate an ambiguous upload.
            KeyboardInterrupt:
                If the adapter simulates cancellation after mutation.
        """
        del bucket_id, token
        self.upload_calls += 1
        if self.fail_upload:
            raise OSError("upload failed")
        for local, name in add or []:
            if name == self.fail_name:
                raise OSError("selected upload failed")
            self.objects[name] = local.read_bytes()
            if self.interrupt_after_write:
                raise KeyboardInterrupt
        for name in delete or []:
            self.objects.pop(name, None)

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[str, Path]],
        *,
        raise_on_missing_files: bool,
        token: str,
    ) -> None:
        """Download selected objects into temporary paths.

        Raises:
            FileNotFoundError:
                If the adapter is configured to fail a selected batch.
        """
        del bucket_id, raise_on_missing_files, token
        self.download_batches.append([name for name, _ in files])
        if len(self.download_batches) == self.fail_download_batch:
            raise FileNotFoundError("remote comparison failed")
        for name, destination in files:
            if name in self.mutate_on_download:
                self.objects[name] = self.mutate_on_download.pop(name)
            if name in self.objects:
                destination.write_bytes(self.objects[name])


class MemoryLock:
    """Deterministic in-memory adapter for the broker lock."""

    def __init__(self) -> None:
        """Initialise an unlocked in-memory broker."""
        self.bucket = "EuroEval/results"
        self.endpoint = "memory://publication-lock"
        self.owner: str | None = None
        self.fail_release = False

    def request(self, action: str, owner: str) -> str:
        """Apply a broker operation atomically.

        Returns:
            The resulting lock state.

        Raises:
            PublicationError:
                If the lock is busy or a release response is unavailable.
        """
        if action == "acquire":
            if self.owner not in {None, owner}:
                raise PublicationError("busy")
            self.owner = owner
            return "owned"
        if action == "assert":
            if self.owner != owner:
                raise PublicationError("busy")
            return "owned"
        if action == "release":
            if self.fail_release:
                raise PublicationError("lost response")
            if self.owner not in {None, owner}:
                raise PublicationError("busy")
            self.owner = None
            return "released"
        return "owned" if self.owner else "unlocked"


class BulkApi(MemoryApi):
    """Batch-aware adapter with deterministic failures and request accounting."""

    def __init__(self, lock: MemoryLock, delay: float = 0) -> None:
        """Initialise an isolated store and optional per-request sleep.

        Args:
            lock: Broker adapter to revoke during a simulated upload.
            delay (optional): Seconds slept per remote request.
        """
        super().__init__()
        self.lock = lock
        self.delay = delay
        self.events: list[tuple[str, list[str]]] = []
        self.fingerprints: dict[bytes, str] = {}
        self.fail_batch: int | None = None
        self.failure = "partial"
        self.metadata_mode = "fresh"
        self.downloaded_bytes = 0
        self.downloaded_files = 0

    def batch_bucket_files(
        self,
        bucket_id: str,
        add: list[tuple[Path, str]] | None = None,
        delete: list[str] | None = None,
        token: str = "",
    ) -> None:
        """Upload a batch with injectable nontransactional failure.

        Args:
            bucket_id: Requested bucket.
            add (optional): Upload pairs.
            delete (optional): Deletion paths.
            token (optional): Test credential.

        Raises:
            OSError: If the batch fails partially or its response is lost.
            KeyboardInterrupt: If interrupted after a partial mutation.
        """
        self.events.append(("upload", [path for _, path in add or []]))
        time.sleep(self.delay)
        failing = self.upload_calls + 1 == self.fail_batch
        selected = add
        if failing and self.failure in {"partial", "interrupt"}:
            selected = (add or [])[:1]
        super().batch_bucket_files(
            bucket_id=bucket_id, add=selected, delete=delete, token=token
        )
        if not failing:
            return
        if self.failure == "interrupt":
            raise KeyboardInterrupt
        if self.failure in {"partial", "lost-response"}:
            raise OSError("uncertain batch")
        path = (add or [])[0][1]
        if self.failure == "missing":
            self.objects.pop(path)
        elif self.failure == "corrupt":
            self.objects[path] = b"corrupt output"
        elif self.failure == "lock-loss":
            self.lock.owner = "another-owner"

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[str, Path]],
        *,
        raise_on_missing_files: bool,
        token: str,
    ) -> None:
        """Record and perform one bounded body request.

        Args:
            bucket_id: Requested bucket.
            files: Object and destination pairs.
            raise_on_missing_files: Missing file policy.
            token: Test credential.
        """
        self.events.append(("download", [path for path, _ in files]))
        self.downloaded_files += sum(path in self.objects for path, _ in files)
        self.downloaded_bytes += sum(
            len(self.objects[path]) for path, _ in files if path in self.objects
        )
        time.sleep(self.delay)
        super().download_bucket_files(
            bucket_id=bucket_id,
            files=files,
            raise_on_missing_files=raise_on_missing_files,
            token=token,
        )

    def get_bucket_paths_info(
        self, bucket_id: str, paths: list[str], *, token: str
    ) -> list[object]:
        """Return bounded fresh metadata, optionally malformed or stale.

        Args:
            bucket_id: Requested bucket.
            paths: Selected object paths.
            token: Test credential.

        Returns:
            Fresh fingerprint entries for present objects.
        """
        del bucket_id, token
        self.events.append(("metadata", paths.copy()))
        time.sleep(self.delay)
        contents = {self.objects[path] for path in paths if path in self.objects}
        missing = contents - self.fingerprints.keys()
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for index, content in enumerate(missing):
                local = Path(directory) / str(index)
                local.write_bytes(content)
                files.append((content, local))
            if files:
                infos = hf_xet.hash_files([str(local) for _, local in files])
                self.fingerprints.update(
                    (content, info.hash)
                    for (content, _), info in zip(files, infos, strict=True)
                )
        if self.metadata_mode == "malformed":
            return [SimpleNamespace(path=paths[0], xet_hash=None)]
        return [
            SimpleNamespace(
                path=path,
                xet_hash=(
                    "stale"
                    if self.metadata_mode == "stale"
                    else self.fingerprints[self.objects[path]]
                ),
            )
            for path in paths
            if path in self.objects
        ]


class CountingLock(MemoryLock):
    """Broker adapter counting remote operations and optional request latency."""

    def __init__(self, delay: float = 0) -> None:
        """Initialise counters.

        Args:
            delay (optional): Seconds slept per remote request.
        """
        super().__init__()
        self.calls = 0
        self.delay = delay

    def request(self, action: str, owner: str) -> str:
        """Count and perform an owned lock operation.

        Args:
            action: Broker operation.
            owner: Session owner.

        Returns:
            Broker state.
        """
        self.calls += 1
        time.sleep(self.delay)
        return super().request(action=action, owner=owner)


def bulk_publisher(tmp_path: Path, api: BulkApi) -> CanonicalPublisher:
    """Construct a publisher with an isolated durable checkpoint.

    Args:
        tmp_path: Isolated test directory.
        api: Remote adapter.

    Returns:
        Publisher using the adapter's broker lock.
    """
    return CanonicalPublisher(
        api=api,
        token="test-token",
        lock=api.lock,
        checkpoint_path=tmp_path / "publication.sqlite3",
    )


def _publisher(tmp_path: Path) -> tuple[CanonicalPublisher, MemoryApi, MemoryLock]:
    """Create an isolated publisher test fixture.

    Returns:
        Publisher, object adapter, and lock adapter.
    """
    api = MemoryApi()
    lock = MemoryLock()
    return CanonicalPublisher(api=api, token="test-token", lock=lock), api, lock


def test_checkpoint_filters_unrelated_sources_and_is_private(tmp_path: Path) -> None:
    """Only candidate-bound evidence is stored in private checkpoint files."""
    checkpoint = tmp_path / "private" / "publication.sqlite3"
    publisher = CanonicalPublisher(
        api=MemoryApi(),
        token="test-token",
        lock=MemoryLock(),
        checkpoint_path=checkpoint,
    )
    candidate = _record()
    identity = identity_from_eee_record(json.loads(candidate))
    other = json.loads(candidate)
    other["model_info"]["id"] = "org/other"
    other_bytes = json.dumps(other).encode()
    canary = other.copy()
    canary["eval_library"]["additional_details"]["dataset"] = "contamination-canary"
    canary_identity = identity_from_eee_record(canary)
    with publisher.session() as session:
        session.prepare(
            records=[candidate, other_bytes],
            sources={
                identity: candidate,
                canary_identity: json.dumps(canary).encode(),
                ("wrong", "task", True, False): candidate,
            },
        )
    connection = sqlite3.connect(checkpoint)
    try:
        rows = connection.execute("SELECT identity FROM sources").fetchall()
    finally:
        connection.close()
    assert rows == [(json.dumps(identity),)]
    assert stat.S_IMODE(checkpoint.stat().st_mode) == 0o600
    assert stat.S_IMODE(checkpoint.parent.stat().st_mode) == 0o700


def test_checkpoint_sources_are_isolated_by_publisher_scope(tmp_path: Path) -> None:
    """A different publisher credential cannot reuse another scope's source proof."""
    api = MemoryApi()
    lock = MemoryLock()
    checkpoint = tmp_path / "publication.sqlite3"
    original = _record()
    identity = identity_from_eee_record(json.loads(original))
    transformed_record = json.loads(original)
    transformed_record["metadata"] = {"open": True}
    transformed_record["model_info"]["additional_details"] = {"open": True}
    transformed = json.dumps(transformed_record).encode()
    path = identity_to_path(identity).as_posix()
    api.objects[path] = original
    first = CanonicalPublisher(
        api=api, token="first-scope", lock=lock, checkpoint_path=checkpoint
    )
    second = CanonicalPublisher(
        api=api, token="second-scope", lock=lock, checkpoint_path=checkpoint
    )

    first.prepare(records=[transformed], sources={identity: original})
    with second.session() as session:
        with pytest.raises(PublicationError, match="Unresolved"):
            session.publish(records=[transformed])
    assert api.objects[path] == original


def test_cold_source_fingerprint_proves_formatting_only_candidate(
    tmp_path: Path,
) -> None:
    """Exact source fingerprints avoid bodies without a prior receipt."""
    api = FingerprintApi()
    lock = MemoryLock()
    checkpoint = tmp_path / "publication.sqlite3"
    original = _record()
    identity = identity_from_eee_record(json.loads(original))
    api.objects[identity_to_path(identity).as_posix()] = original
    publisher = CanonicalPublisher(
        api=api, token="test-token", lock=lock, checkpoint_path=checkpoint
    )
    formatted = json.dumps(json.loads(original), indent=2).encode()
    with publisher.session() as session:
        assert session.publish(records=[formatted], sources={identity: original}) == 0
    assert api.download_calls == 0


class FingerprintApi(MemoryApi):
    """Memory adapter exposing fresh Xet metadata and read counters."""

    def __init__(self) -> None:
        """Initialise the API adapter and operation counters."""
        super().__init__()
        self.metadata_calls = 0
        self.download_calls = 0

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[str, Path]],
        *,
        raise_on_missing_files: bool,
        token: str,
    ) -> None:
        """Count remote body reads before delegating to the memory adapter."""
        self.download_calls += len(files)
        super().download_bucket_files(
            bucket_id=bucket_id,
            files=files,
            raise_on_missing_files=raise_on_missing_files,
            token=token,
        )

    def get_bucket_paths_info(
        self, bucket_id: str, paths: list[str], *, token: str
    ) -> list[object]:
        """Return fresh content fingerprints for requested remote files."""
        self.metadata_calls += 1
        entries: list[object] = []
        for path in paths:
            content = self.objects.get(path)
            if content is not None:
                with tempfile.TemporaryDirectory() as directory:
                    local = Path(directory) / "object"
                    local.write_bytes(content)
                    fingerprint = hf_xet.hash_files([str(local)])[0].hash
                entries.append(SimpleNamespace(path=path, xet_hash=fingerprint))
        return entries


def test_cold_source_fingerprint_receipt_reuses_formatted_candidate(
    tmp_path: Path,
) -> None:
    """A cold exact source proof seeds a reusable semantic receipt."""
    api = FingerprintApi()
    lock = MemoryLock()
    checkpoint = tmp_path / "publication.sqlite3"
    original = json.dumps(json.loads(_record()), indent=2).encode()
    identity = identity_from_eee_record(json.loads(original))
    path = identity_to_path(identity).as_posix()
    api.objects[path] = original
    publisher = CanonicalPublisher(
        api=api, token="test-token", lock=lock, checkpoint_path=checkpoint
    )
    candidate = json.dumps(json.loads(original), separators=(",", ":")).encode()
    with publisher.session() as session:
        assert session.publish(records=[candidate], sources={identity: original}) == 0
    assert api.download_calls == 0
    assert api.upload_calls == 0

    publisher = CanonicalPublisher(
        api=api, token="test-token", lock=lock, checkpoint_path=checkpoint
    )
    reformatted = json.dumps(json.loads(candidate), indent=4).encode()
    remote_fingerprint = hf_xet.hash_files
    with tempfile.TemporaryDirectory() as directory:
        local = Path(directory) / "remote"
        local.write_bytes(original)
        fingerprint = remote_fingerprint([str(local)])[0].hash
    with (
        patch.object(
            api,
            "get_bucket_paths_info",
            return_value=[SimpleNamespace(path=path, xet_hash=fingerprint)],
        ),
        patch("leaderboards.canonical_publication.hf_xet.hash_files") as hash_files,
    ):
        with publisher.session() as session:
            assert session.publish(records=[reformatted]) == 0
        hash_files.assert_not_called()
    assert api.download_calls == 0
    assert api.upload_calls == 0
    assert api.objects[path] == original


def test_duplicate_candidates_archive_only_once(tmp_path: Path) -> None:
    """Repeated identical candidates collapse to one archive and canonical write."""
    api = BulkApi(lock=CountingLock())
    original = _record()
    path = identity_to_path(identity_from_eee_record(json.loads(original))).as_posix()
    api.objects[path] = original
    with bulk_publisher(tmp_path=tmp_path, api=api).session() as session:
        assert (
            session.publish(
                records=[_record(score=0.6, timestamp="2026-01-02T00:00:00Z")] * 1001
            )
            == 1
        )
    assert api.upload_calls == 2
    assert len(api.objects) == 2


def test_equal_rank_metadata_transform_cannot_change_evaluation_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact source bytes do not authorise changing evaluation results."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, _ = _publisher(tmp_path)
    original = _record()
    changed_record = json.loads(original)
    changed_record["score"] = 0.9
    changed = json.dumps(changed_record).encode()
    identity = identity_from_eee_record(json.loads(original))
    with publisher.session() as session:
        session.publish(records=[original])
        with pytest.raises(PublicationError, match="Unresolved"):
            session.publish(records=[changed], sources={identity: original})
    canonical = next(name for name in api.objects if name.endswith(".json"))
    assert api.objects[canonical] == original


def test_equal_rank_metadata_transform_requires_original_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal-rank replacement is allowed only with exact predecessor evidence."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, _ = _publisher(tmp_path)
    original = _record()
    transformed_record = json.loads(original)
    transformed_record["model_info"]["additional_details"] = {"open": True}
    transformed = json.dumps(transformed_record).encode()
    identity = identity_from_eee_record(json.loads(original))
    with publisher.session() as session:
        assert session.publish(records=[original]) == 1
        canonical_path = identity_to_path(identity).as_posix()
        upload_calls = api.upload_calls
        with pytest.raises(PublicationError, match="Unresolved"):
            session.publish(records=[transformed])
        assert api.objects[canonical_path] == original
        assert api.upload_calls == upload_calls
        assert not any(name.startswith("_history/") for name in api.objects)
        assert session.publish(records=[transformed], sources={identity: original}) == 1
    assert api.objects[canonical_path] == transformed
    assert any(name.startswith("_history/") for name in api.objects)


def test_equal_rank_numeric_enrichment_cannot_change_existing_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not treat a changed valid model value as missing-metadata enrichment."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, _, _ = _publisher(tmp_path)
    original_record = json.loads(_record())
    original_record["model_info"]["additional_details"] = {"num_model_parameters": 1000}
    original = json.dumps(original_record).encode()
    changed_record = json.loads(original)
    changed_record["model_info"]["additional_details"]["num_model_parameters"] = 2000
    changed = json.dumps(changed_record).encode()
    identity = identity_from_eee_record(original_record)
    with publisher.session() as session:
        session.publish(records=[original])
        with pytest.raises(PublicationError, match="Unresolved"):
            session.publish(records=[changed], sources={identity: original})


def test_existing_history_is_verified_and_not_uploaded_again(tmp_path: Path) -> None:
    """Content-addressed history is reused, but corrupt history blocks replacement."""
    api = BulkApi(lock=CountingLock())
    original = _record()
    path = identity_to_path(identity_from_eee_record(json.loads(original))).as_posix()
    history = f"_history/{hashlib.sha256(original).hexdigest()}.bin"
    api.objects.update({path: original, history: b"corrupt"})
    publisher = bulk_publisher(tmp_path=tmp_path, api=api)
    replacement = _record(score=0.6, timestamp="2026-01-02T00:00:00Z")
    with pytest.raises(PublicationError, match="History hash collision"):
        with publisher.session() as session:
            session.publish(records=[replacement])
    assert api.upload_calls == 0
    assert api.objects[path] == original
    api.objects[history] = original
    api.events.clear()
    with publisher.session() as session:
        assert session.publish(records=[replacement]) == 1
    assert api.upload_calls == 1
    assert not any(
        operation == "download" and history in paths for operation, paths in api.events
    )


def test_failed_archive_retains_lock_before_canonical_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable predecessor archive blocks overwrite and retains ownership."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, lock = _publisher(tmp_path)
    original = _record()
    with publisher.session() as session:
        session.publish(records=[original])
    path = next(name for name in api.objects if name.endswith(".json"))
    api.fail_name = f"_history/{hashlib.sha256(original).hexdigest()}.bin"
    try:
        with pytest.raises(OSError):
            with publisher.session() as session:
                session.publish(
                    records=[_record(score=0.6, timestamp="2026-01-02T00:00:00Z")]
                )
    finally:
        api.fail_name = None
    assert api.objects[path] == original
    assert lock.owner is not None


def test_interrupt_after_partial_mutation_retains_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation after a remote write keeps the lock for explicit recovery."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, lock = _publisher(tmp_path)
    api.interrupt_after_write = True
    with pytest.raises(KeyboardInterrupt):
        with publisher.session() as session:
            session.write_verified(path="object", content=b"partial", expected=None)
    assert api.objects["object"] == b"partial"
    assert lock.owner is not None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("num_model_parameters", True),
        ("vocabulary_size", 0),
        ("max_sequence_length", 2**63),
    ],
)
def test_invalid_numeric_metadata_enrichment_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: object
) -> None:
    """Reject invalid numeric values in otherwise metadata-only changes."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, _ = _publisher(tmp_path)
    original = _record()
    enriched_record = json.loads(original)
    enriched_record["model_info"]["additional_details"] = {key: value}
    enriched = json.dumps(enriched_record).encode()
    identity = identity_from_eee_record(json.loads(original))
    with publisher.session() as session:
        session.publish(records=[original])
        with pytest.raises(PublicationError, match="Invalid numeric metadata"):
            session.publish(records=[enriched], sources={identity: original})
    canonical_path = identity_to_path(identity).as_posix()
    assert api.objects[canonical_path] == original


def test_json_equal_formatting_change_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Formatting-only candidate changes do not rewrite canonical bytes."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, _ = _publisher(tmp_path)
    original = _record()
    formatted = json.dumps(json.loads(original), indent=2).encode()
    with publisher.session() as session:
        assert session.publish(records=[original]) == 1
        assert session.publish(records=[formatted]) == 0
    canonical = next(name for name in api.objects if name.endswith(".json"))
    assert api.objects[canonical] == original
    assert not any(name.startswith("_history/") for name in api.objects)


def test_later_metadata_enrichment_only_reads_changed_records(tmp_path: Path) -> None:
    """A later enrichment pass still publishes real changes without rereading peers."""
    api = BulkApi(lock=CountingLock())
    publisher = bulk_publisher(tmp_path=tmp_path, api=api)
    originals = bulk_records(count=501)
    with publisher.session() as session:
        session.publish(records=originals)
    enriched = []
    for index, raw in enumerate(originals):
        record = json.loads(raw)
        if index < 5:
            record["model_info"]["additional_details"] = {"open": True}
        enriched.append(json.dumps(record, indent=2).encode())
    sources = {identity_from_eee_record(json.loads(raw)): raw for raw in originals}
    publisher.prepare(records=enriched, sources=sources)
    api.events.clear()
    with bulk_publisher(tmp_path=tmp_path, api=api).session() as session:
        assert session.publish(records=enriched) == 5
    unchanged = {
        identity_to_path(identity).as_posix() for identity in list(sources)[5:]
    }
    assert all(
        not unchanged.intersection(paths)
        for operation, paths in api.events
        if operation != "metadata"
    )
    histories = {
        f"_history/{hashlib.sha256(raw).hexdigest()}.bin" for raw in originals[:5]
    }
    assert histories <= api.objects.keys()
    for raw in enriched[:5]:
        path = identity_to_path(identity_from_eee_record(json.loads(raw))).as_posix()
        assert api.objects[path] == raw
    api.events.clear()
    with publisher.session() as session:
        assert session.publish(records=enriched) == 0
    assert all(operation == "metadata" for operation, _ in api.events)


def test_linear_candidate_validation_and_warm_formatted_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Large source validation and reconstructed formatting retries stay efficient."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    api = FingerprintApi()
    lock = MemoryLock()
    checkpoint = tmp_path / "publication.sqlite3"
    records = []
    sources = {}
    for index in range(1_000):
        record = json.loads(_record())
        record["model_info"]["id"] = f"org/linear-{index}"
        record["eval_library"]["additional_details"]["dataset"] = f"task-{index}"
        content = json.dumps(record).encode()
        identity = identity_from_eee_record(record)
        records.append(content)
        sources[identity] = content
    publisher = CanonicalPublisher(
        api=api, token="test-token", lock=lock, checkpoint_path=checkpoint
    )
    with publisher.session() as session:
        session.prepare(records=records, sources=sources)
        assert session.publish(records=records, sources=sources) == 1_000
    reads = api.download_calls
    publisher = CanonicalPublisher(
        api=api, token="test-token", lock=lock, checkpoint_path=checkpoint
    )
    formatted = [
        json.dumps(json.loads(record), indent=2).encode() for record in records
    ]
    hash_calls: list[int] = []
    real_hash_files = hf_xet.hash_files

    def count_hash_files(paths: list[str]) -> list[object]:
        hash_calls.append(len(paths))
        return real_hash_files(paths)

    with patch(
        "leaderboards.canonical_publication.hf_xet.hash_files", count_hash_files
    ):
        with publisher.session() as session:
            assert session.publish(records=formatted, sources=sources) == 0
    assert api.download_calls == reads
    assert len(hash_calls) == 1_000
    assert set(hash_calls) == {1}


def test_local_hashing_is_bounded_to_comparison_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local candidate hashing never submits an unbounded path list."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    api = FingerprintApi()
    publisher = CanonicalPublisher(api=api, token="test-token", lock=MemoryLock())
    records = []
    for index in range(1_001):
        record = json.loads(_record())
        record["model_info"]["id"] = f"org/hash-{index}"
        record["eval_library"]["additional_details"]["dataset"] = f"task-{index}"
        records.append(json.dumps(record).encode())

    original_hash_files = hf_xet.hash_files

    def counted_hash_files(paths: list[str]) -> list[object]:
        sizes.append(len(paths))
        return original_hash_files(paths)

    sizes: list[int] = []
    with patch(
        "leaderboards.canonical_publication.hf_xet.hash_files", counted_hash_files
    ):
        with publisher.session() as session:
            session.publish(records=records)

    assert sizes
    assert max(sizes) <= 500


def test_lock_retries_lost_acquire_response_with_same_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Broker retries reuse the opaque owner after an ambiguous response."""
    lock = PublicationLock(
        endpoint="https://example.test/publication-lock", secret="secret"
    )
    bodies: list[bytes] = []

    class Response:
        """Minimal successful HTTP response."""

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return (
                b'{"bucket":"EuroEval/results","owner":"same-owner","status":"owned"}'
            )

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> Response:
        """Lose one response, then confirm an idempotent retry.

        Returns:
            Successful simulated response.

        Raises:
            urllib.error.URLError:
                On the first simulated response loss.
        """
        del timeout
        bodies.append(t.cast(bytes, request.data))
        if len(bodies) == 1:
            raise urllib.error.URLError("response lost")
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert lock.request(action="acquire", owner="same-owner") == "owned"
    assert len(bodies) == 2 and bodies[0] == bodies[1]


@pytest.mark.parametrize("mode", ["duplicate", "unexpected-path"])
def test_malformed_metadata_batch_falls_back_to_exact_bodies(
    tmp_path: Path, mode: str
) -> None:
    """Malformed metadata uses exact body checks rather than authorising writes."""
    api = BulkApi(lock=CountingLock())
    records = bulk_records(count=501)
    path = identity_to_path(identity_from_eee_record(json.loads(records[0]))).as_posix()
    entries = [SimpleNamespace(path=path, xet_hash="fingerprint")] * 2
    if mode == "unexpected-path":
        entries = [SimpleNamespace(path="unknown", xet_hash="fingerprint")]
    with patch.object(api, "get_bucket_paths_info", return_value=entries):
        with bulk_publisher(tmp_path=tmp_path, api=api).session() as session:
            assert session.publish(records=records) == len(records)
    assert len(api.objects) == len(records)
    assert api.upload_calls == 2
    assert api.downloaded_bytes > 0


def test_manual_decision_cannot_be_reused_for_changed_candidate(tmp_path: Path) -> None:
    """A manual decision remains bound to the exact approved candidate bytes."""
    api = MemoryApi()
    lock = MemoryLock()
    publisher = CanonicalPublisher(
        api=api,
        token="test-token",
        lock=lock,
        checkpoint_path=tmp_path / "publication.sqlite3",
    )
    original = _record(score=0.9)
    candidate = _record(score=0.1)
    changed_candidate = _record(score=0.2)
    identity = identity_from_eee_record(json.loads(original))
    path = identity_to_path(identity).as_posix()
    api.objects[path] = original
    decision = resolve_result_conflict(
        current=json.loads(original),
        candidate=json.loads(candidate),
        current_bytes=original,
        candidate_bytes=candidate,
        choice=ConflictChoice.INCOMING,
    )

    with pytest.raises(PublicationError, match="not bound"):
        publisher.prepare(
            records=[changed_candidate],
            sources={identity: original},
            decisions=[decision],
        )
    assert api.objects[path] == original


def test_manual_incoming_choice_archives_exact_predecessor(tmp_path: Path) -> None:
    """An explicit incoming choice can replace an older conflicting record."""
    api = MemoryApi()
    lock = MemoryLock()
    publisher = CanonicalPublisher(
        api=api,
        token="test-token",
        lock=lock,
        checkpoint_path=tmp_path / "publication.sqlite3",
    )
    original = _record(score=0.9)
    candidate = _record(score=0.1)
    identity = identity_from_eee_record(json.loads(original))
    path = identity_to_path(identity).as_posix()
    api.objects[path] = original
    decision = resolve_result_conflict(
        current=json.loads(original),
        candidate=json.loads(candidate),
        current_bytes=original,
        candidate_bytes=candidate,
        choice=ConflictChoice.INCOMING,
    )

    publisher.prepare(
        records=[candidate], sources={identity: original}, decisions=[decision]
    )
    with publisher.session() as session:
        assert (
            session.publish(
                records=[candidate], sources={identity: original}, decisions=[decision]
            )
            == 1
        )

    assert api.objects[path] == candidate
    assert (
        api.objects[f"_history/{hashlib.sha256(original).hexdigest()}.bin"] == original
    )


def test_manual_stored_choice_retains_canonical_bytes(tmp_path: Path) -> None:
    """A stored-side decision cannot turn into an incoming overwrite."""
    api = MemoryApi()
    lock = MemoryLock()
    publisher = CanonicalPublisher(
        api=api,
        token="test-token",
        lock=lock,
        checkpoint_path=tmp_path / "publication.sqlite3",
    )
    original = _record(score=0.9)
    candidate = _record(score=0.1)
    identity = identity_from_eee_record(json.loads(original))
    api.objects[identity_to_path(identity).as_posix()] = original
    decision = resolve_result_conflict(
        current=json.loads(original),
        candidate=json.loads(candidate),
        current_bytes=original,
        candidate_bytes=candidate,
        choice=ConflictChoice.STORED,
    )
    with publisher.session() as session:
        assert (
            session.publish(
                records=[candidate], sources={identity: original}, decisions=[decision]
            )
            == 0
        )
    assert api.objects[identity_to_path(identity).as_posix()] == original
    assert not any(path.startswith("_history/") for path in api.objects)


def test_metadata_source_race_does_not_cache_false_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A body changed after metadata lookup is read and rejected on retry."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    api = CountingFingerprintApi()
    lock = MemoryLock()
    checkpoint = tmp_path / "publication.sqlite3"
    original = _record()
    identity = identity_from_eee_record(json.loads(original))
    path = identity_to_path(identity).as_posix()
    api.objects[path] = original
    publisher = CanonicalPublisher(
        api=api, token="test-token", lock=lock, checkpoint_path=checkpoint
    )
    formatted = json.dumps(json.loads(original), indent=2).encode()
    replacement_record = json.loads(original)
    replacement_record["score"] = 0.8
    api.metadata_mutation[path] = json.dumps(replacement_record).encode()

    with publisher.session() as session:
        with pytest.raises(PublicationError, match="Unresolved"):
            session.publish(records=[formatted])

    connection = sqlite3.connect(checkpoint)
    try:
        rows = connection.execute("SELECT path FROM receipts").fetchall()
    finally:
        connection.close()
    assert rows == []
    with publisher.session() as session:
        with pytest.raises(PublicationError, match="Unresolved"):
            session.publish(records=[formatted])
    assert api.download_calls >= 2


class CountingFingerprintApi(FingerprintApi):
    """Fingerprint adapter recording the largest local hash request."""

    def __init__(self) -> None:
        """Initialise hash request tracking."""
        super().__init__()
        self.hash_batch_sizes: list[int] = []
        self.metadata_mutation: dict[str, bytes] = {}

    def get_bucket_paths_info(
        self, bucket_id: str, paths: list[str], *, token: str
    ) -> list[object]:
        """Return metadata and optionally race remote bodies after observing them."""
        entries = super().get_bucket_paths_info(bucket_id, paths, token=token)
        self.objects.update(self.metadata_mutation)
        self.metadata_mutation.clear()
        return entries


def test_missing_lock_configuration_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publisher construction cannot silently fall back to a local lock.

    Args:
        monkeypatch:
            Isolates missing configuration from the local publishing environment.
    """
    monkeypatch.delenv(name="CANONICAL_PUBLICATION_SECRET", raising=False)
    with pytest.raises(ValueError, match="required"):
        CanonicalPublisher(api=MemoryApi(), token="token", lock=None)


def test_missing_object_after_metadata_is_not_treated_as_new(tmp_path: Path) -> None:
    """A missing predecessor body cannot lead to overwriting without its backup."""
    api = BulkApi(lock=CountingLock())
    original = _record()
    path = identity_to_path(identity_from_eee_record(json.loads(original))).as_posix()
    api.objects[path] = original
    metadata = api.get_bucket_paths_info

    def delete_after_metadata(
        bucket_id: str, paths: list[str], *, token: str
    ) -> list[object]:
        """Simulate external deletion after metadata and before body download.

        Args:
            bucket_id: Requested bucket.
            paths: Selected paths.
            token: Test credential.

        Returns:
            Metadata observed just before deletion.
        """
        entries = metadata(bucket_id=bucket_id, paths=paths, token=token)
        api.objects.pop(path)
        return entries

    with patch.object(api, "get_bucket_paths_info", side_effect=delete_after_metadata):
        with pytest.raises(PublicationError, match="missing after metadata"):
            with bulk_publisher(tmp_path=tmp_path, api=api).session() as session:
                session.publish(
                    records=[_record(score=0.6, timestamp="2026-01-02T00:00:00Z")]
                )
    assert api.upload_calls == 0
    assert api.lock.owner is None


def test_mixed_batches_archive_every_predecessor_before_canonical_upload(
    tmp_path: Path,
) -> None:
    """Multiple archive batches finish verification before any canonical write."""
    lock = CountingLock()
    api = BulkApi(lock=lock)
    originals = bulk_records(count=501)
    for content in originals:
        path = identity_to_path(
            identity_from_eee_record(json.loads(content))
        ).as_posix()
        api.objects[path] = content
    records = bulk_records(count=502, later_evaluation=True)
    publisher = bulk_publisher(tmp_path=tmp_path, api=api)
    with publisher.session() as session:
        assert session.publish(records=records) == 502
    histories = {f"_history/{hashlib.sha256(raw).hexdigest()}.bin" for raw in originals}
    assert histories <= api.objects.keys()
    verified: dict[str, int] = {}
    for operation, paths in api.events:
        if operation == "metadata":
            for path in paths:
                if path in histories:
                    verified[path] = verified.get(path, 0) + 1
        if operation == "upload" and any(
            not path.startswith("_history/") for path in paths
        ):
            assert histories == {path for path, count in verified.items() if count >= 2}
    assert api.upload_calls == 4
    api.events.clear()
    with publisher.session() as session:
        assert session.publish(records=originals) == 0
    assert not any(operation == "upload" for operation, _ in api.events)


def test_non_manual_decision_cannot_authorise_conflict(tmp_path: Path) -> None:
    """Automatic resolver outcomes are not manual authorisation proofs."""
    api = MemoryApi()
    lock = MemoryLock()
    publisher = CanonicalPublisher(
        api=api,
        token="test-token",
        lock=lock,
        checkpoint_path=tmp_path / "publication.sqlite3",
    )
    original = _record(score=0.9)
    candidate = _record(score=0.1)
    identity = identity_from_eee_record(json.loads(original))
    api.objects[identity_to_path(identity).as_posix()] = original
    automatic = resolve_result_conflict(
        current=json.loads(original),
        candidate=json.loads(candidate),
        current_bytes=original,
        candidate_bytes=candidate,
    )
    with pytest.raises(PublicationError, match="not bound"):
        with publisher.session() as session:
            session.publish(
                records=[candidate], sources={identity: original}, decisions=[automatic]
            )
    assert api.objects[identity_to_path(identity).as_posix()] == original


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluation_results", {"score": 0.9}),
        ("raw_results", {"score": 0.9}),
        ("rank", 1),
        ("unapproved_metadata", {"note": "changed"}),
    ],
)
def test_numeric_enrichment_does_not_authorise_other_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    """Numeric enrichment cannot authorise evaluation or arbitrary changes."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, _, _ = _publisher(tmp_path)
    original = _record()
    candidate_record = json.loads(original)
    candidate_record["model_info"]["additional_details"] = {
        "num_model_parameters": 1000
    }
    candidate_record[field] = value
    candidate = json.dumps(candidate_record).encode()
    identity = identity_from_eee_record(json.loads(original))
    with publisher.session() as session:
        session.publish(records=[original])
        with pytest.raises(PublicationError, match="Unresolved"):
            session.publish(records=[candidate], sources={identity: original})


def test_pipeline_numeric_enrichment_is_publishable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publish numeric metadata produced by the model enrichment pipeline."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, _ = _publisher(tmp_path)
    original = json.loads(_record())
    original["model_info"]["name"] = "org/model"
    original["model_info"]["additional_details"] = {
        "open": True,
        "model_url": "https://huggingface.co/org/model",
        "generative_type": "base",
        "merge": False,
        "commercially_licensed": False,
        "trained_from_scratch": False,
        "generative": False,
        "release_date": None,
    }
    original_bytes = json.dumps(original).encode()
    with (
        patch("leaderboards.model_metadata._get_release_date", return_value=None),
        patch(
            "leaderboards.model_metadata.recover_numeric_metadata",
            return_value={
                "num_model_parameters": 123456,
                "vocabulary_size": 32000,
                "max_sequence_length": 8192,
            },
        ),
    ):
        enriched = add_missing_entries(
            record=original,
            trained_from_scratch_patterns=TRAINED_FROM_SCRATCH_PATTERNS,
            cache=Cache(),
        )

    enriched_bytes = json.dumps(enriched).encode()
    identity = identity_from_eee_record(original)
    with publisher.session() as session:
        session.publish(records=[original_bytes])
        assert (
            session.publish(
                records=[enriched_bytes], sources={identity: original_bytes}
            )
            == 1
        )
    canonical = next(name for name in api.objects if name.endswith(".json"))
    assert (
        json.loads(api.objects[canonical])["model_info"]["additional_details"][
            "num_model_parameters"
        ]
        == 123456
    )


def test_prepare_rejects_changed_remote_for_manual_retry(tmp_path: Path) -> None:
    """An approved decision cannot be prepared against an unrelated remote body."""
    api = MemoryApi()
    lock = MemoryLock()
    publisher = CanonicalPublisher(
        api=api,
        token="test-token",
        lock=lock,
        checkpoint_path=tmp_path / "publication.sqlite3",
    )
    original = _record(score=0.9)
    candidate = _record(score=0.1)
    identity = identity_from_eee_record(json.loads(original))
    path = identity_to_path(identity).as_posix()
    api.objects[path] = original
    decision = resolve_result_conflict(
        current=json.loads(original),
        candidate=json.loads(candidate),
        current_bytes=original,
        candidate_bytes=candidate,
        choice=ConflictChoice.INCOMING,
    )
    api.objects[path] = _record(score=0.8)

    with pytest.raises(PublicationError, match="predecessor changed"):
        publisher.prepare(
            records=[candidate], sources={identity: original}, decisions=[decision]
        )


def test_prepare_validates_sources_with_linear_identity_comparisons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Public preparation validates N source identities with O(N) comparisons."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, _, _ = _publisher(tmp_path)
    records: list[bytes] = []
    sources: dict[tuple[str, str, bool | None, bool | None], bytes] = {}
    for index in range(600):
        record = json.loads(_record())
        record["model_info"]["id"] = f"org/comparison-{index}"
        record["eval_library"]["additional_details"]["dataset"] = f"task-{index}"
        content = json.dumps(record).encode()
        records.append(content)
        sources[identity_from_eee_record(record)] = content

    comparisons = 0

    class CountedIdentity(tuple):
        """Tuple identity counting equality checks performed by preparation."""

        def __eq__(self, other: object) -> bool:
            nonlocal comparisons
            comparisons += 1
            return super().__eq__(other)

        __hash__ = tuple.__hash__

    real_identity = identity_from_eee_record

    def counted_identity(
        record: dict[str, object],
    ) -> tuple[str, str, bool | None, bool | None]:
        return CountedIdentity(real_identity(record))

    with patch(
        "leaderboards.canonical_publication.identity_from_eee_record", counted_identity
    ):
        publisher.prepare(records=records, sources=sources)

    assert comparisons < 6 * len(records)


def test_prepared_manual_publication_retry_accepts_exact_candidate(
    tmp_path: Path,
) -> None:
    """A durable manual proof permits retry after the candidate was uploaded."""
    api = MemoryApi()
    lock = MemoryLock()
    checkpoint = tmp_path / "publication.sqlite3"
    publisher = CanonicalPublisher(
        api=api, token="test-token", lock=lock, checkpoint_path=checkpoint
    )
    original = _record(score=0.9)
    candidate = _record(score=0.1)
    identity = identity_from_eee_record(json.loads(original))
    path = identity_to_path(identity).as_posix()
    api.objects[path] = original
    decision = resolve_result_conflict(
        current=json.loads(original),
        candidate=json.loads(candidate),
        current_bytes=original,
        candidate_bytes=candidate,
        choice=ConflictChoice.INCOMING,
    )

    publisher.prepare(
        records=[candidate], sources={identity: original}, decisions=[decision]
    )
    with publisher.session() as session:
        assert session.publish(records=[candidate]) == 1

    history = f"_history/{hashlib.sha256(original).hexdigest()}.bin"
    assert api.objects[history] == original
    uploads_after_publication = api.upload_calls
    retry_publisher = CanonicalPublisher(
        api=api, token="test-token", lock=lock, checkpoint_path=checkpoint
    )
    retry_publisher.prepare(
        records=[candidate], sources={identity: original}, decisions=[decision]
    )
    with retry_publisher.session() as session:
        assert session.publish(records=[candidate]) == 0

    assert api.objects[path] == candidate
    assert api.objects[history] == original
    assert api.upload_calls == uploads_after_publication


def test_publish_archives_predecessor_and_skips_old(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Newer publication archives the prior content and rejects stale overwrite."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, _ = _publisher(tmp_path)
    with publisher.session() as session:
        assert session.publish(records=[_record()]) == 1
        assert session.publish(records=[_record()]) == 0
        assert (
            session.publish(
                records=[_record(score=0.6, timestamp="2026-01-02T00:00:00Z")]
            )
            == 1
        )
    assert len([name for name in api.objects if name.startswith("_history/")]) == 1


def test_publish_batches_remote_comparison_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged records are compared in bounded bulk downloads."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, _ = _publisher(tmp_path)
    records = []
    for index in range(1_001):
        record = json.loads(_record())
        record["model_info"]["id"] = f"org/model-{index}"
        record["eval_library"]["additional_details"]["dataset"] = f"task-{index}"
        records.append(json.dumps(record).encode())
    api.objects.update(
        {
            identity_to_path(
                identity_from_eee_record(json.loads(record))
            ).as_posix(): record
            for record in records
        }
    )

    with publisher.session() as session:
        assert session.publish(records=records) == 0

    assert len(api.download_batches) == 3
    assert [len(batch) for batch in api.download_batches] == [500, 500, 1]
    assert all(len(batch) <= 500 for batch in api.download_batches)


def test_reconstructed_publisher_skips_verified_remote_bodies(tmp_path: Path) -> None:
    """Fresh matching Xet metadata avoids downloading unchanged result bodies."""
    api = FingerprintApi()
    lock = MemoryLock()
    checkpoint = tmp_path / "publication.sqlite3"
    record = _record()
    publisher = CanonicalPublisher(
        api=api, token="test-token", lock=lock, checkpoint_path=checkpoint
    )
    with publisher.session() as session:
        assert session.publish(records=[record]) == 1
    initial_reads = api.download_calls
    assert initial_reads == 0

    publisher = CanonicalPublisher(
        api=api, token="test-token", lock=lock, checkpoint_path=checkpoint
    )
    formatted = json.dumps(json.loads(record), indent=2).encode()
    with publisher.session() as session:
        assert session.publish(records=[formatted]) == 0
    assert api.download_calls == initial_reads
    assert api.metadata_calls == 4


@pytest.mark.parametrize(("fail_batch", "record_count"), [(1, 1), (2, 501)])
def test_remote_comparison_failure_aborts_before_any_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_batch: int, record_count: int
) -> None:
    """A failed batch comparison never treats unknown records as absent."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, _ = _publisher(tmp_path)
    api.fail_download_batch = fail_batch
    records = []
    for index in range(record_count):
        record = json.loads(_record())
        if record_count > 1:
            record["model_info"]["id"] = f"org/model-{index}"
            record["eval_library"]["additional_details"]["dataset"] = f"task-{index}"
        records.append(json.dumps(record).encode())

    with pytest.raises(FileNotFoundError, match="remote comparison failed"):
        with publisher.session() as session:
            session.publish(records=records)

    assert len(api.download_batches) == fail_batch
    assert api.upload_calls == 0
    assert api.objects == {}


@pytest.mark.parametrize("count", [1, 10_000])
def test_replacement_request_budget(tmp_path: Path, count: int) -> None:
    """All-predecessor replacements also stay below the modelled request budget."""
    lock = CountingLock()
    api = BulkApi(lock=lock)
    originals = bulk_records(count=count)
    for raw in originals:
        path = identity_to_path(identity_from_eee_record(json.loads(raw))).as_posix()
        api.objects[path] = raw
    with bulk_publisher(tmp_path=tmp_path, api=api).session() as session:
        assert (
            session.publish(records=bulk_records(count=count, later_evaluation=True))
            == count
        )
    batches = (count + 499) // 500
    assert api.upload_calls == 2 * batches
    assert len(api.events) == 9 * batches
    assert lock.calls == 3 + 4 * batches
    assert max(len(paths) for _, paths in api.events) <= 500
    assert (len(api.events) + lock.calls) * 2 < 900


@pytest.mark.parametrize("failure", ["missing", "corrupt", "partial", "interrupt"])
def test_second_archive_batch_failure_blocks_all_canonical_writes(
    tmp_path: Path, failure: str
) -> None:
    """No replacement starts while any predecessor backup remains unverified."""
    api = BulkApi(lock=CountingLock())
    originals = bulk_records(count=503)
    original_objects = {
        identity_to_path(identity_from_eee_record(json.loads(raw))).as_posix(): raw
        for raw in originals
    }
    api.objects.update(original_objects)
    api.fail_batch = 2
    api.failure = failure
    error = (
        KeyboardInterrupt
        if failure == "interrupt"
        else (OSError if failure == "partial" else PublicationError)
    )
    publisher = bulk_publisher(tmp_path=tmp_path, api=api)
    with pytest.raises(error):
        with publisher.session() as session:
            session.publish(records=bulk_records(count=503, later_evaluation=True))
    assert all(api.objects[path] == raw for path, raw in original_objects.items())
    assert all(
        all(path.startswith("_history/") for path in paths)
        for operation, paths in api.events
        if operation == "upload"
    )
    assert api.lock.owner is not None
    with sqlite3.connect(publisher.checkpoint_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM receipts").fetchone() == (0,)


def test_stale_candidate_cannot_replace_newer_current_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale writer that acquires later leaves newer canonical bytes intact."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, _ = _publisher(tmp_path)
    with publisher.session() as session:
        session.publish(records=[_record(version="3.0.0")])
    path = next(name for name in api.objects if name.endswith(".json"))
    newest = api.objects[path]
    with publisher.session() as session:
        assert session.publish(records=[_record(version="1.0.0")]) == 0
    assert api.objects[path] == newest


@pytest.mark.parametrize(
    "failure",
    ["partial", "lost-response", "interrupt", "corrupt", "missing", "lock-loss"],
)
@pytest.mark.parametrize("replacements", [False, True])
def test_uncertain_batch_preserves_receipts_evidence_and_converges(
    tmp_path: Path, failure: str, replacements: bool
) -> None:
    """Verified earlier chunks survive; uncertain chunks retain lock and evidence."""
    lock = CountingLock()
    api = BulkApi(lock=lock)
    originals = bulk_records(count=503)
    sources = {identity_from_eee_record(json.loads(raw)): raw for raw in originals}
    if replacements:
        api.objects.update(
            {
                identity_to_path(identity).as_posix(): raw
                for identity, raw in sources.items()
            }
        )
    records = []
    for raw in originals:
        record = json.loads(raw)
        record["score"] = 0.6
        record["evaluation_results"] = {"score": 0.6}
        record["eval_timestamp"] = "2026-01-02T00:00:00Z"
        record["retrieved_timestamp"] = "2026-01-02T00:00:00Z"
        records.append(json.dumps(record).encode())
    publisher = bulk_publisher(tmp_path=tmp_path, api=api)
    publisher.prepare(records=records, sources=sources)
    api.fail_batch = 4 if replacements else 2
    api.failure = failure
    exception = (
        KeyboardInterrupt
        if failure == "interrupt"
        else (OSError if failure in {"partial", "lost-response"} else PublicationError)
    )
    with pytest.raises(exception):
        with publisher.session() as session:
            session.publish(records=records)
    assert lock.owner is not None
    with sqlite3.connect(publisher.checkpoint_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM receipts").fetchone() == (500,)
        assert connection.execute("SELECT COUNT(*) FROM sources").fetchone() == (503,)
    # Simulate explicit operator reconciliation/release, never automatic lock stealing.
    lock.owner = None
    api.fail_batch = None
    if failure == "corrupt":
        # Corruption must fail closed, not be silently accepted or overwritten.
        with pytest.raises(PublicationError, match="Invalid JSON"):
            with publisher.session() as session:
                session.publish(records=records)
        corrupt_path = next(
            path
            for path, content in api.objects.items()
            if content == b"corrupt output"
        )
        api.objects.pop(corrupt_path)
    api.events.clear()
    with bulk_publisher(tmp_path=tmp_path, api=api).session() as session:
        session.publish(records=records)
    assert {
        raw for path, raw in api.objects.items() if not path.startswith("_history/")
    } == set(records)
    if replacements:
        assert all(
            api.objects[f"_history/{hashlib.sha256(raw).hexdigest()}.bin"] == raw
            for raw in originals
        )
    # The completed first chunk is receipt-proven, not downloaded or uploaded again.
    assert all(
        len(paths) <= 3 for operation, paths in api.events if operation != "metadata"
    )
    with sqlite3.connect(publisher.checkpoint_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM receipts").fetchone() == (503,)
        assert connection.execute("SELECT COUNT(*) FROM sources").fetchone() == (0,)


@pytest.mark.parametrize(
    "key", ["num_model_parameters", "vocabulary_size", "max_sequence_length"]
)
@pytest.mark.parametrize("value", [None, -1, "-1"])
def test_unknown_numeric_metadata_can_be_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: object
) -> None:
    """Preserve recognised unknown metadata on new and unchanged records.

    Args:
        tmp_path: Isolated publication state directory.
        monkeypatch: Environment isolation fixture.
        key: Numeric metadata field.
        value: Recognised unknown-value placeholder.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, _ = _publisher(tmp_path)
    record = json.loads(_record())
    record["model_info"]["additional_details"] = {key: value}
    content = json.dumps(record).encode()
    identity = identity_from_eee_record(record=record)
    publisher.prepare(records=[content], sources={})
    with publisher.session() as session:
        assert session.publish(records=[content]) == 1
    publisher.prepare(records=[content], sources={identity: content})
    with publisher.session() as session:
        assert session.publish(records=[content], sources={identity: content}) == 0
    assert api.objects[identity_to_path(identity=identity).as_posix()] == content


def test_upload_failure_poison_retains_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ambiguous mutation failures retain the durable lock for recovery."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, lock = _publisher(tmp_path)
    api.fail_upload = True
    with pytest.raises(OSError):
        with publisher.session() as session:
            session.write_verified(path="object", content=b"x", expected=None)
    assert lock.owner is not None
    assert list((tmp_path / "euroeval-publication-locks").glob("*.json"))


@pytest.mark.parametrize(
    ("stored_engine", "incoming_engine", "stored_score", "incoming_score", "replace"),
    [
        ("litellm", "vllm", 0.02032105367, 0.00065181237, True),
        ("vllm", "litellm", 0.00065181237, 0.02032105367, False),
    ],
)
def test_vllm_engine_preference_publication(
    tmp_path: Path,
    stored_engine: str,
    incoming_engine: str,
    stored_score: float,
    incoming_score: float,
    replace: bool,
) -> None:
    """Publish exact preferred-engine source and archive replaced bytes."""
    api = MemoryApi()
    publisher = CanonicalPublisher(
        api=api,
        token="test-token",
        lock=MemoryLock(),
        checkpoint_path=tmp_path / "publication.sqlite3",
    )

    def encode_record(*, engine: str, score: float, timestamp: str) -> bytes:
        record: dict[str, object] = {
            "model_info": {"id": "org/model", "inference_engine": {"name": engine}},
            "eval_library": {
                "name": "euroeval",
                "version": "1.0",
                "additional_details": {
                    "dataset": "task",
                    "few_shot": False,
                    "validation_split": False,
                    "library_version": engine,
                    "raw_results": {"score": score},
                },
            },
            "eval_timestamp": timestamp,
            "evaluation_results": {"score": score},
        }
        if engine == "litellm":
            record["model_type"] = "generative"
        return json.dumps(record).encode()

    stored = encode_record(
        engine=stored_engine, score=stored_score, timestamp="2026-10-22T00:00:00Z"
    )
    incoming = encode_record(
        engine=incoming_engine, score=incoming_score, timestamp="2026-07-18T00:00:00Z"
    )
    identity = identity_from_eee_record(record=json.loads(stored))
    path = identity_to_path(identity).as_posix()
    api.objects[path] = stored
    decision = resolve_result_conflict(
        current=json.loads(stored),
        candidate=json.loads(incoming),
        current_bytes=stored,
        candidate_bytes=incoming,
    )
    assert decision.reason == ConflictReason.ENGINE_PREFERENCE
    assert decision.evidence.current_sha256 == hashlib.sha256(stored).hexdigest()
    assert decision.evidence.candidate_sha256 == hashlib.sha256(incoming).hexdigest()
    assert decision.selected == (
        ConflictChoice.INCOMING if replace else ConflictChoice.STORED
    )
    publisher.prepare(
        records=[incoming], sources={identity: stored}, decisions=[decision]
    )
    with publisher.session() as session:
        published = session.publish(
            records=[incoming], sources={identity: stored}, decisions=[decision]
        )

    assert published == int(replace)
    assert api.objects[path] == (incoming if replace else stored)
    if replace:
        history = f"_history/{hashlib.sha256(stored).hexdigest()}.bin"
        assert api.objects[history] == stored


def test_write_verified_detects_mutated_expected_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lock does not substitute for checking current expected bytes."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, _ = _publisher(tmp_path)
    api.objects["object"] = b"before"
    with publisher.session() as session:
        api.mutate_on_download["object"] = b"changed"
        with pytest.raises(PublicationError, match="Concurrent change"):
            session.write_verified(path="object", content=b"next", expected=b"before")


def test_write_verified_rejects_external_remote_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A remotely deleted predecessor cannot be silently restored by a writer."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, _ = _publisher(tmp_path)
    api.objects["object"] = b"before"
    with publisher.session() as session:
        del api.objects["object"]
        with pytest.raises(PublicationError, match="Concurrent change"):
            session.write_verified(path="object", content=b"next", expected=b"before")
    assert "object" not in api.objects
