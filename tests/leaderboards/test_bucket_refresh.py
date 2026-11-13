"""Tests for refreshing canonical result files from the bucket."""

import json
from pathlib import Path

import pytest
from huggingface_hub import BucketFile

from leaderboards import bucket_sync


def test_existing_file_migrates_when_manifest_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An existing pre-manifest file is refreshed from bucket state."""
    path = "org_model/dataset__test__zeroshot.json"
    data = json.dumps(make_record()).encode()
    bucket = FakeBucket({path: ("hash", data)})
    configure(monkeypatch, tmp_path, bucket)
    destination = result_path(tmp_path / "results")
    destination.parent.mkdir(parents=True)
    destination.write_text("stale", encoding="utf-8")

    assert bucket_sync.download_missing_bucket_files() == 1
    assert destination.read_bytes() == data


class FakeBucket:
    """In-memory Hugging Face bucket adapter."""

    def __init__(self, files: dict[str, tuple[str, bytes]]) -> None:
        """Initialise the adapter with bucket object contents.

        Args:
            files:
                Bucket paths mapped to fingerprints and bytes.
        """
        self.files = files
        self.download_calls = 0
        self.omit_download: set[str] = set()

    def download_bucket_files(self, *, files: list, **kwargs: object) -> None:
        """Write configured downloads to requested temporary destinations."""
        self.download_calls += 1
        for entry, destination in files:
            if entry.path not in self.omit_download:
                Path(destination).write_bytes(self.files[entry.path][1])

    def list_bucket_tree(self, *args: object, **kwargs: object) -> list[BucketFile]:
        """List configured bucket objects.

        Returns:
            The configured bucket entries.
        """
        return [
            BucketFile(type="file", path=path, size=len(data), xetHash=fingerprint)
            for path, (fingerprint, data) in self.files.items()
        ]


def configure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bucket: FakeBucket
) -> None:
    """Configure the refresh function to use a local fake bucket."""
    monkeypatch.setattr(bucket_sync, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(bucket_sync, "HfApi", lambda: bucket)
    monkeypatch.setattr(bucket_sync, "resolve_hf_token", lambda: "token")


def make_record(score: int = 1) -> dict[str, object]:
    """Return a valid EEE result record."""
    return {
        "schema_version": "1.0",
        "model_info": {
            "id": "org/model",
            "additional_details": {
                "commercially_licensed": False,
                "open": True,
                "trained_from_scratch": False,
            },
        },
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "validation_split": False,
                "few_shot": False,
            }
        },
        "evaluation_results": [{"score": score}],
    }


def result_path(root: Path) -> Path:
    """Return the canonical result path beneath a result root."""
    return root / "org_model/dataset__test__zeroshot.json"


@pytest.mark.parametrize("failure", ["omit", "invalid", "partial"])
def test_failed_download_does_not_replace_file_or_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    """Incomplete downloads preserve both installed record and generation."""
    path = "org_model/dataset__test__zeroshot.json"
    old = b"previous-good-file"
    data = json.dumps(make_record()).encode()
    if failure == "invalid":
        data = b"not-json"
    elif failure == "partial":
        data = json.dumps({"schema_version": "1"}).encode()
    bucket = FakeBucket({path: ("new-hash", data)})
    if failure == "omit":
        bucket.omit_download.add(path)
    configure(monkeypatch, tmp_path, bucket)
    destination = result_path(tmp_path / "results")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(old)
    manifest = tmp_path / ".results-bucket-manifest.json"
    manifest.write_text('{"old": "generation"}', encoding="utf-8")

    with pytest.raises((ValueError, json.JSONDecodeError)):
        bucket_sync.download_missing_bucket_files()

    assert destination.read_bytes() == old
    assert manifest.read_text(encoding="utf-8") == '{"old": "generation"}'


def test_history_binary_objects_are_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """History binary objects are not treated as result records."""
    bucket = FakeBucket({"_history/object.bin": ("hash", b"history")})
    configure(monkeypatch, tmp_path, bucket)

    assert bucket_sync.download_missing_bucket_files() == 0
    assert bucket.download_calls == 0


def test_preserves_local_only_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Refreshing remote paths does not remove local-only records."""
    path = "org_model/dataset__test__zeroshot.json"
    data = json.dumps(make_record()).encode()
    bucket = FakeBucket({path: ("hash", data)})
    configure(monkeypatch, tmp_path, bucket)
    local_only = tmp_path / "results/local-only.json"
    local_only.parent.mkdir(parents=True)
    local_only.write_text("keep", encoding="utf-8")

    bucket_sync.download_missing_bucket_files()
    assert local_only.read_text(encoding="utf-8") == "keep"


def test_refreshes_same_path_and_size_when_hash_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Changed object fingerprints refresh even when byte lengths match."""
    path = "org_model/dataset__test__zeroshot.json"
    old = json.dumps(make_record(score=1)).encode()
    new = json.dumps(make_record(score=2)).encode()
    assert len(old) == len(new)
    bucket = FakeBucket({path: ("new-hash", new)})
    configure(monkeypatch, tmp_path, bucket)
    destination = result_path(tmp_path / "results")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(old)
    (tmp_path / ".results-bucket-manifest.json").write_text(
        json.dumps({path: "old-hash"}), encoding="utf-8"
    )

    assert bucket_sync.download_missing_bucket_files() == 1
    assert destination.read_bytes() == new


def test_unchanged_hash_skips_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A matching manifest fingerprint avoids network downloads."""
    path = "org_model/dataset__test__zeroshot.json"
    data = json.dumps(make_record()).encode()
    bucket = FakeBucket({path: ("same-hash", data)})
    configure(monkeypatch, tmp_path, bucket)
    destination = result_path(tmp_path / "results")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(data)
    (tmp_path / ".results-bucket-manifest.json").write_text(
        json.dumps({path: "same-hash"}), encoding="utf-8"
    )

    assert bucket_sync.download_missing_bucket_files() == 0
    assert bucket.download_calls == 0
