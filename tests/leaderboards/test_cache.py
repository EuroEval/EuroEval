"""Tests for the cache module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from leaderboards.cache import Cache


def _make_eee_record(
    model_id: str = "test/model",
    model_name: str | None = None,
    dataset: str = "dataset1",
    validation_split: bool | None = False,
    few_shot: bool | None = False,
    commercially_licensed: bool | None = None,
    open_license: bool | None = None,
    trained_from_scratch: bool | None = None,
    model_url: str | None = None,
) -> dict:
    """Helper to create a valid EEE-format record for cache testing.

    Args:
        model_id:
            Model identifier.
        model_name:
            Model name (defaults to model_id).
        dataset:
            Dataset name.
        validation_split:
            Validation split flag.
        few_shot:
            Few-shot flag.
        commercially_licensed:
            Whether commercially licensed.
        open_license:
            Whether open license.
        trained_from_scratch:
            Whether trained from scratch.
        model_url:
            Model URL.

    Returns:
        A valid EEE-format record dict.
    """
    if model_name is None:
        model_name = model_id
    record = {
        "schema_version": "1.0",
        "model_info": {"id": model_id, "name": model_name, "additional_details": {}},
        "eval_library": {
            "additional_details": {
                "dataset": dataset,
                "validation_split": validation_split,
                "few_shot": few_shot,
            }
        },
        "retrieved_timestamp": "100",
        "evaluation_results": [{"label": "pass", "score": 0.5}],
    }

    # Add metadata fields if provided
    has_metadata = any(
        v is not None
        for v in [
            commercially_licensed,
            open_license,
            trained_from_scratch,
            model_url,
        ]
    )
    if has_metadata:
        additional = {}
        if commercially_licensed is not None:
            additional["commercially_licensed"] = commercially_licensed
        if open_license is not None:
            additional["open"] = open_license
        if trained_from_scratch is not None:
            additional["trained_from_scratch"] = trained_from_scratch
        if model_url is not None:
            additional["model_url"] = model_url
        record["model_info"]["additional_details"] = additional

    return record


class TestCacheFromResultsDir:
    """Tests for Cache.from_results_dir with tree structure."""

    def test_loads_from_tree_structure(self, tmp_path: Path) -> None:
        """Should load records from the tree structure."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Record with some metadata to ensure cache populates
        record = _make_eee_record(
            model_id="test/model",
            model_name="test/model",
            open_license=True,
        )
        json_file = model_dir / "dataset1__test__zeroshot.json"
        json_file.write_text(json.dumps(record))

        cache = Cache.from_results_dir(results_dir=tmp_path)

        # Cache should have populated open dict
        assert "test/model" in cache.open
        assert cache.open["test/model"] is True

    def test_preserves_metadata_fields(self, tmp_path: Path) -> None:
        """Should preserve metadata fields in model_info.additional_details."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        record = _make_eee_record(
            model_id="test/model",
            model_name="test/model",
            commercially_licensed=True,
            open_license=True,
            trained_from_scratch=False,
            model_url="https://example.com/model",
        )
        json_file = model_dir / "dataset1__test__zeroshot.json"
        json_file.write_text(json.dumps(record))

        cache = Cache.from_results_dir(results_dir=tmp_path)

        assert cache.commercially_licensed.get("test/model") is True
        assert cache.open.get("test/model") is True
        assert cache.trained_from_scratch.get("test/model") is False
        assert cache.model_url.get("test/model") == "https://example.com/model"

    def test_loads_multiple_models(self, tmp_path: Path) -> None:
        """Should load records from multiple model subdirectories."""
        model_a_dir = tmp_path / "model_a"
        model_b_dir = tmp_path / "model_b"
        model_a_dir.mkdir()
        model_b_dir.mkdir()

        record_a = _make_eee_record(
            model_id="model/a",
            model_name="model/a",
            open_license=True,
        )
        record_b = _make_eee_record(
            model_id="model/b",
            model_name="model/b",
            commercially_licensed=True,
        )

        (model_a_dir / "ds__test__zeroshot.json").write_text(json.dumps(record_a))
        (model_b_dir / "ds__test__zeroshot.json").write_text(json.dumps(record_b))

        cache = Cache.from_results_dir(results_dir=tmp_path)

        assert "model/a" in cache.open
        assert cache.open["model/a"] is True
        assert "model/b" in cache.commercially_licensed
        assert cache.commercially_licensed["model/b"] is True

    def test_raises_on_nonexistent_directory(self) -> None:
        """Should raise FileNotFoundError for nonexistent directory."""
        with pytest.raises(FileNotFoundError, match="Results directory"):
            Cache.from_results_dir(results_dir=Path("/nonexistent/path"))

    def test_empty_directory_returns_empty_cache(self, tmp_path: Path) -> None:
        """Should return empty cache for directory with no JSON files."""
        (tmp_path / "empty_model").mkdir()

        cache = Cache.from_results_dir(results_dir=tmp_path)

        # Cache should be empty
        assert len(cache.commercially_licensed) == 0
        assert len(cache.open) == 0
        assert len(cache.trained_from_scratch) == 0
