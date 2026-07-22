"""Tests for the collect_evaluation_results script, focusing on upload_results_to_hf."""

import json
import typing as t
from pathlib import Path

import pytest

from leaderboards import constants
from src.scripts import collect_evaluation_results


class FakeHfApi:
    """Fake HfApi that mocks sync_bucket and batch_bucket_files calls."""

    def sync_bucket(
        self,
        source: str,
        dest: str,
        token: str | None = None,
        ignore_times: bool = False,
        **kwargs: t.Any,
    ) -> None:
        """No-op sync for testing."""
        pass

    def batch_bucket_files(
        self,
        bucket_id: str,
        add: list[tuple[str | Path | bytes, str]] | None = None,
        delete: list[str] | None = None,
        **kwargs: t.Any,
    ) -> None:
        """No-op batch upload for testing."""
        pass


def test_upload_results_to_hf_collision_leaves_results_dir_unmutated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Distinct-identity path collision raises AND leaves RESULTS_DIR unmutated.

    Two records with different identities that sanitise to the same path should
    raise ValueError BEFORE any files are deleted from RESULTS_DIR.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create existing record in RESULTS_DIR
    # Dataset "dataset/one" sanitises to "dataset_one"
    model_dir = results_dir / "foo_bar"
    model_dir.mkdir()
    existing_record_file = model_dir / "dataset_one__test__zeroshot.json"
    existing_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset/one",  # Sanitises to "dataset_one"
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704067200,  # 2024-01-01T00:00:00Z as Unix timestamp
    }
    existing_record_file.write_text(json.dumps(existing_record), encoding="utf-8")

    # Create new results file with a colliding identity
    # Dataset "dataset_one" also sanitises to "dataset_one" - same path!
    new_results_file = tmp_path / "new_results.jsonl"
    colliding_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset_one",  # Different identity, same sanitised path
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704153600,  # 2024-01-02T00:00:00Z as Unix timestamp
    }
    new_results_file.write_text(json.dumps(colliding_record), encoding="utf-8")

    # Monkeypatch RESULTS_DIR and HfApi
    monkeypatch.setattr(
        target=collect_evaluation_results, name="RESULTS_DIR", value=results_dir
    )
    monkeypatch.setattr(
        target=collect_evaluation_results, name="HfApi", value=FakeHfApi
    )

    # Should raise ValueError due to collision
    with pytest.raises(ValueError, match="Identity collision detected"):
        collect_evaluation_results.upload_results_to_hf(
            new_results_path=new_results_file
        )

    # Existing record should STILL exist (no data loss)
    assert existing_record_file.exists(), "Existing record was incorrectly deleted"
    content = json.loads(existing_record_file.read_text(encoding="utf-8"))
    assert content == existing_record


def test_upload_results_to_hf_same_identity_keeps_newer_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same-identity duplicate keeps the NEWER record.

    When two records share the same identity, the newer one wins (by euroeval_version,
    then retrieved_timestamp).
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create existing record with older version
    model_dir = results_dir / "foo_bar"
    model_dir.mkdir()
    existing_record_file = model_dir / "dataset__test__zeroshot.json"
    older_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704067200,  # 2024-01-01T00:00:00Z
    }
    existing_record_file.write_text(json.dumps(older_record), encoding="utf-8")

    # Create new results file with newer version (same identity)
    new_results_file = tmp_path / "new_results.jsonl"
    newer_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "2.0.0",  # Newer version
        },
        "retrieved_timestamp": 1704067200,  # 2024-01-01T00:00:00Z
    }
    new_results_file.write_text(json.dumps(newer_record), encoding="utf-8")

    # Monkeypatch RESULTS_DIR and HfApi
    monkeypatch.setattr(
        target=collect_evaluation_results, name="RESULTS_DIR", value=results_dir
    )
    monkeypatch.setattr(
        target=collect_evaluation_results, name="HfApi", value=FakeHfApi
    )

    # Should succeed
    result = collect_evaluation_results.upload_results_to_hf(
        new_results_path=new_results_file
    )
    assert result is True

    # The newer record should be written
    content = json.loads(existing_record_file.read_text(encoding="utf-8"))
    assert content["eval_library"]["version"] == "2.0.0"


def test_upload_results_to_hf_same_identity_uses_timestamp_tiebreaker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same-identity with same version uses retrieved_timestamp as tiebreaker.

    When versions are equal, the record with newer retrieved_timestamp wins.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create existing record with older timestamp
    model_dir = results_dir / "foo_bar"
    model_dir.mkdir()
    existing_record_file = model_dir / "dataset__test__zeroshot.json"
    older_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704067200,  # 2024-01-01T00:00:00Z
    }
    existing_record_file.write_text(json.dumps(older_record), encoding="utf-8")

    # Create new results file with newer timestamp (same version)
    new_results_file = tmp_path / "new_results.jsonl"
    newer_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",  # Same version
        },
        "retrieved_timestamp": 1704153600,  # 2024-01-02T00:00:00Z (newer)
    }
    new_results_file.write_text(json.dumps(newer_record), encoding="utf-8")

    # Monkeypatch RESULTS_DIR and HfApi
    monkeypatch.setattr(
        target=collect_evaluation_results, name="RESULTS_DIR", value=results_dir
    )
    monkeypatch.setattr(
        target=collect_evaluation_results, name="HfApi", value=FakeHfApi
    )

    # Should succeed
    result = collect_evaluation_results.upload_results_to_hf(
        new_results_path=new_results_file
    )
    assert result is True

    # The newer record (by timestamp) should be written
    content = json.loads(existing_record_file.read_text(encoding="utf-8"))
    assert content["retrieved_timestamp"] == 1704153600


def test_upload_results_to_hf_deletes_stale_jsonl_in_results_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stale root RESULTS_DIR/*.jsonl is deleted on upload."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create stale jsonl file in RESULTS_DIR
    stale_jsonl = results_dir / "stale.jsonl"
    stale_jsonl.write_text('{"old": "data"}', encoding="utf-8")

    model_dir = results_dir / "foo_bar"
    model_dir.mkdir()
    existing_record_file = model_dir / "dataset__test__zeroshot.json"
    existing_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704067200,  # 2024-01-01T00:00:00Z
    }
    existing_record_file.write_text(json.dumps(existing_record), encoding="utf-8")

    # New results file
    new_results_file = tmp_path / "new_results.jsonl"
    new_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704067200,  # 2024-01-01T00:00:00Z
    }
    new_results_file.write_text(json.dumps(new_record), encoding="utf-8")

    # Monkeypatch RESULTS_DIR and HfApi
    monkeypatch.setattr(
        target=collect_evaluation_results, name="RESULTS_DIR", value=results_dir
    )
    monkeypatch.setattr(
        target=collect_evaluation_results, name="HfApi", value=FakeHfApi
    )

    # Should succeed
    result = collect_evaluation_results.upload_results_to_hf(
        new_results_path=new_results_file
    )
    assert result is True

    # Stale jsonl should be deleted
    assert not stale_jsonl.exists(), "Stale jsonl in RESULTS_DIR was not deleted"


def test_upload_results_to_hf_does_not_touch_repo_root_jsonl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Repo-root JSONL files are NOT touched by upload.

    The upload function only touches RESULTS_DIR/*.jsonl files, not repo-root
    JSONL files like new_results.jsonl or euroeval_benchmark_results.jsonl.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    repo_root = tmp_path / "repo_root"
    repo_root.mkdir()

    model_dir = results_dir / "foo_bar"
    model_dir.mkdir()
    existing_record_file = model_dir / "dataset__test__zeroshot.json"
    existing_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704067200,  # 2024-01-01T00:00:00Z
    }
    existing_record_file.write_text(json.dumps(existing_record), encoding="utf-8")

    # Create repo-root JSONL files that should NOT be touched
    new_results_file = repo_root / "new_results.jsonl"
    new_results_file.write_text('{"data": "new"}', encoding="utf-8")
    benchmark_results_file = repo_root / "euroeval_benchmark_results.jsonl"
    benchmark_results_file.write_text('{"data": "benchmark"}', encoding="utf-8")

    # Use new_results_file as input (will be read, not deleted by upload)
    input_file = tmp_path / "input.jsonl"
    input_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704067200,  # 2024-01-01T00:00:00Z
    }
    input_file.write_text(json.dumps(input_record), encoding="utf-8")

    # Monkeypatch RESULTS_DIR, REPO_ROOT, and HfApi
    monkeypatch.setattr(
        target=collect_evaluation_results, name="RESULTS_DIR", value=results_dir
    )
    monkeypatch.setattr(target=constants, name="REPO_ROOT", value=repo_root)
    monkeypatch.setattr(
        target=collect_evaluation_results, name="HfApi", value=FakeHfApi
    )

    # Should succeed
    result = collect_evaluation_results.upload_results_to_hf(
        new_results_path=input_file
    )
    assert result is True

    # Repo-root jsonl files should still exist untouched
    assert new_results_file.exists(), "new_results.jsonl was incorrectly deleted"
    assert benchmark_results_file.exists(), (
        "euroeval_benchmark_results.jsonl was incorrectly deleted"
    )
    assert new_results_file.read_text(encoding="utf-8") == '{"data": "new"}'
    assert benchmark_results_file.read_text(encoding="utf-8") == '{"data": "benchmark"}'
