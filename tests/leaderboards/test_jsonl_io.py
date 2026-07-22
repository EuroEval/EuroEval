"""Tests for the JSONL I/O utilities, including tree loading."""

from __future__ import annotations

import json
import typing as t
from pathlib import Path

import pytest

from leaderboards.jsonl_io import (
    load_records_from_jsonl_files,
    load_records_from_result_tree,
)


class TestLoadRecordsFromResultTree:
    """Tests for load_records_from_result_tree."""

    def test_loads_single_record_per_file(self, tmp_path: Path) -> None:
        """Should load one record per JSON file in the tree structure."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        record = {"model_info": {"id": "test_model"}, "scores": {"acc": 0.5}}
        json_file = model_dir / "dataset__test__zeroshot.json"
        json_file.write_text(json.dumps(record))

        records = load_records_from_result_tree(results_dir=tmp_path)

        assert len(records) == 1
        # Type ignore needed because ty infers dict[str, object]
        assert records[0]["model_info"]["id"] == "test_model"  # ty: ignore[not-subscriptable]
        assert records[0]["scores"]["acc"] == 0.5  # ty: ignore[not-subscriptable]

    def test_loads_multiple_models(self, tmp_path: Path) -> None:
        """Should load records from multiple model subdirectories."""
        model_a_dir = tmp_path / "model_a"
        model_b_dir = tmp_path / "model_b"
        model_a_dir.mkdir()
        model_b_dir.mkdir()

        record_a = {"model_info": {"id": "model_a"}, "scores": {"acc": 0.6}}
        record_b = {"model_info": {"id": "model_b"}, "scores": {"acc": 0.7}}

        (model_a_dir / "ds__test__zeroshot.json").write_text(json.dumps(record_a))
        (model_b_dir / "ds__test__zeroshot.json").write_text(json.dumps(record_b))

        records = load_records_from_result_tree(results_dir=tmp_path)

        assert len(records) == 2
        model_ids = {t.cast(dict[str, object], r["model_info"])["id"] for r in records}
        assert model_ids == {"model_a", "model_b"}

    def test_loads_multiple_datasets_per_model(self, tmp_path: Path) -> None:
        """Should load multiple dataset files per model."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        record1 = {"model_info": {"id": "test"}, "scores": {"acc": 0.5}}
        record2 = {"model_info": {"id": "test"}, "scores": {"acc": 0.6}}

        (model_dir / "dataset1__test__zeroshot.json").write_text(json.dumps(record1))
        (model_dir / "dataset2__val__fewshot.json").write_text(json.dumps(record2))

        records = load_records_from_result_tree(results_dir=tmp_path)

        assert len(records) == 2

    def test_raises_on_malformed_json(self, tmp_path: Path) -> None:
        """Should raise JSONDecodeError on malformed JSON files."""
        model_dir = tmp_path / "bad_model"
        model_dir.mkdir()

        json_file = model_dir / "dataset__test__zeroshot.json"
        json_file.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            load_records_from_result_tree(results_dir=tmp_path)

    def test_raises_on_non_dict_content(self, tmp_path: Path) -> None:
        """Should raise ValueError if JSON file contains non-dict."""
        model_dir = tmp_path / "bad_model"
        model_dir.mkdir()

        json_file = model_dir / "dataset__test__zeroshot.json"
        json_file.write_text("[1, 2, 3]")

        with pytest.raises(ValueError, match="Expected dict"):
            load_records_from_result_tree(results_dir=tmp_path)

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """Should return empty list for directory with no JSON files."""
        (tmp_path / "empty_model").mkdir()

        records = load_records_from_result_tree(results_dir=tmp_path)

        assert records == []

    def test_preserves_metadata_fields(self, tmp_path: Path) -> None:
        """Should preserve all metadata fields in model_info."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        record = {
            "model_info": {
                "id": "test/model",
                "name": "Test Model",
                "additional_details": {
                    "commercially_licensed": True,
                    "open": True,
                    "trained_from_scratch": False,
                    "model_url": "https://example.com/model",
                },
            },
            "scores": {"acc": 0.5},
        }

        json_file = model_dir / "dataset__test__zeroshot.json"
        json_file.write_text(json.dumps(record))

        records = load_records_from_result_tree(results_dir=tmp_path)

        assert len(records) == 1
        loaded = records[0]
        assert loaded["model_info"]["id"] == "test/model"  # ty: ignore[not-subscriptable]
        assert loaded["model_info"]["name"] == "Test Model"  # ty: ignore[not-subscriptable]
        details = loaded["model_info"]["additional_details"]  # ty: ignore[not-subscriptable]
        assert details["commercially_licensed"] is True
        assert details["open"] is True
        assert details["trained_from_scratch"] is False
        assert details["model_url"] == "https://example.com/model"


class TestLoadRecordsFromJsonlFiles:
    """Tests for load_records_from_jsonl_files (unchanged API)."""

    def test_loads_jsonl_records(self, tmp_path: Path) -> None:
        """Should load records from JSONL files."""
        jsonl_file = tmp_path / "model.jsonl"
        jsonl_file.write_text(
            '{"model_info": {"id": "model_a"}, "scores": {"acc": 0.5}}\n'
            '{"model_info": {"id": "model_b"}, "scores": {"acc": 0.6}}\n'
        )

        records = load_records_from_jsonl_files(paths=[jsonl_file])

        assert len(records) == 2
        assert records[0]["model_info"]["id"] == "model_a"  # ty: ignore[not-subscriptable]
        assert records[1]["model_info"]["id"] == "model_b"  # ty: ignore[not-subscriptable]
