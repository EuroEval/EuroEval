"""Tests for the shared canonical publication interface."""

from __future__ import annotations

import hashlib
import json
import typing as t
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from leaderboards.canonical_publication import (
    CanonicalPublisher,
    PublicationError,
    PublicationLock,
)
from leaderboards.result_identity import identity_from_eee_record


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
        self.fail_upload = False
        self.interrupt_after_write = False
        self.fail_name: str | None = None
        self.mutate_on_download: dict[str, bytes] = {}

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
        """Download selected objects into temporary paths."""
        del bucket_id, raise_on_missing_files, token
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


def _publisher(tmp_path: Path) -> tuple[CanonicalPublisher, MemoryApi, MemoryLock]:
    """Create an isolated publisher test fixture.

    Returns:
        Publisher, object adapter, and lock adapter.
    """
    api = MemoryApi()
    lock = MemoryLock()
    return CanonicalPublisher(api=api, token="test-token", lock=lock), api, lock


def _record(*, version: str = "1.0.0", score: float = 0.5) -> bytes:
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
            "retrieved_timestamp": "2026-01-01T00:00:00Z",
            "score": score,
        }
    ).encode()


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
        with pytest.raises(PublicationError, match="Equal-ranked"):
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
    transformed_record["metadata"] = {"open": True}
    transformed_record["model_info"]["additional_details"] = {"open": True}
    transformed = json.dumps(transformed_record).encode()
    identity = identity_from_eee_record(json.loads(original))
    with publisher.session() as session:
        assert session.publish(records=[original]) == 1
        with pytest.raises(PublicationError, match="Equal-ranked"):
            session.publish(records=[transformed])
        assert session.publish(records=[transformed], sources={identity: original}) == 1
    assert any(name.startswith("_history/") for name in api.objects)


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
                session.publish(records=[_record(version="2.0.0")])
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


def test_missing_lock_configuration_fails_closed() -> None:
    """Publisher construction cannot silently fall back to a local lock."""
    with pytest.raises(ValueError, match="required"):
        CanonicalPublisher(api=MemoryApi(), token="token", lock=None)


def test_publish_archives_predecessor_and_skips_old(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Newer publication archives the prior content and rejects stale overwrite."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    publisher, api, _ = _publisher(tmp_path)
    with publisher.session() as session:
        assert session.publish(records=[_record()]) == 1
        assert session.publish(records=[_record()]) == 0
        assert session.publish(records=[_record(version="2.0.0")]) == 1
    assert len([name for name in api.objects if name.startswith("_history/")]) == 1


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
