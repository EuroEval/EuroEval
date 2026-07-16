"""Tests for the create_language_spider_plot script."""

from __future__ import annotations

import json
import typing as t

import pytest

from src.scripts.create_language_spider_plot import (
    check_completeness,
    compute_max_score,
    extract_languages_from_record,
    extract_score_from_record,
    filter_by_shots,
    is_few_shot,
)


def make_eee_record(
    model_name: str,
    languages: list[str],
    score: float,
    few_shot: bool,
    metric_name: str = "test_macro_f1",
) -> dict[str, t.Any]:
    """Create a minimal EEE-format record for testing.

    Args:
        model_name:
            Model name/ID.
        languages:
            Language codes.
        score:
            Score value.
        few_shot:
            Whether this is a few-shot record.
        metric_name (optional):
            Metric name. Defaults to "test_macro_f1".

    Returns:
        EEE-format record dictionary.
    """
    return {
        "schema_version": "0.2.1",
        "model_info": {"name": model_name, "id": model_name, "additional_details": {}},
        "eval_library": {
            "name": "euroeval",
            "version": "17.0.0",
            "additional_details": {
                "languages": json.dumps(languages),  # fmt: skip
                "few_shot": "true" if few_shot else "false",
            },
        },
        "evaluation_results": [
            {
                "evaluation_name": metric_name,
                "source_data": {"dataset_name": "test-dataset"},
                "metric_config": {"lower_is_better": False},
                "score_details": {"score": score, "details": {}},
            }
        ],
    }


class TestIsFewShot:
    """Tests for is_few_shot function."""

    def test_few_shot_true(self) -> None:
        """Should detect few-shot records."""
        record = make_eee_record("test-model", ["da"], 80.0, few_shot=True)
        assert is_few_shot(record) is True

    def test_zero_shot_false(self) -> None:
        """Should detect zero-shot records."""
        record = make_eee_record("test-model", ["da"], 80.0, few_shot=False)
        assert is_few_shot(record) is False

    def test_missing_field(self) -> None:
        """Should return None when few_shot field is missing."""
        record = {
            "model_info": {"name": "test"},
            "eval_library": {"additional_details": {}},
        }
        assert is_few_shot(record) is None

    def test_boolean_field(self) -> None:
        """Should handle boolean few_shot field."""
        record = {
            "model_info": {"name": "test"},
            "eval_library": {"additional_details": {"few_shot": True}},
        }
        assert is_few_shot(record) is True


class TestExtractLanguagesFromRecord:
    """Tests for extract_languages_from_record function."""

    def test_eee_format_json_string(self) -> None:
        """Should extract languages from EEE JSON-encoded string."""
        record = make_eee_record("test-model", ["da", "sv"], 80.0, few_shot=False)
        languages = extract_languages_from_record(record)
        assert languages == ["da", "sv"]

    def test_legacy_list_format(self) -> None:
        """Should handle legacy list format."""
        record = {
            "model_info": {"name": "test"},
            "eval_library": {},
            "languages": ["da", "sv"],
        }
        languages = extract_languages_from_record(record)
        assert languages == ["da", "sv"]

    def test_legacy_json_string(self) -> None:
        """Should handle legacy JSON string format."""
        record = {
            "model_info": {"name": "test"},
            "eval_library": {},
            "languages": '["da", "sv"]',
        }
        languages = extract_languages_from_record(record)
        assert languages == ["da", "sv"]

    def test_single_language_string(self) -> None:
        """Should handle single language as string."""
        record = {"model_info": {"name": "test"}, "eval_library": {}, "languages": "da"}
        languages = extract_languages_from_record(record)
        assert languages == ["da"]

    def test_empty_languages(self) -> None:
        """Should return empty list when no languages."""
        record = {"model_info": {"name": "test"}, "eval_library": {}}
        languages = extract_languages_from_record(record)
        assert languages == []


class TestExtractScoreFromRecord:
    """Tests for extract_score_from_record function."""

    def test_extract_macro_f1(self) -> None:
        """Should extract test_macro_f1 score."""
        record = make_eee_record("test-model", ["da"], 85.5, few_shot=False)
        score = extract_score_from_record(record, "test_macro_f1")
        assert score == 85.5

    def test_extract_accuracy(self) -> None:
        """Should extract test_accuracy score."""
        record = make_eee_record(
            "test-model", ["da"], 90.0, few_shot=False, metric_name="test_accuracy"
        )
        score = extract_score_from_record(record, "test_accuracy")
        assert score == 90.0

    def test_missing_metric(self) -> None:
        """Should return None for missing metric."""
        record = make_eee_record("test-model", ["da"], 80.0, few_shot=False)
        score = extract_score_from_record(record, "test_rouge")
        assert score is None

    def test_nan_score(self) -> None:
        """Should handle NaN scores."""
        record = make_eee_record("test-model", ["da"], float("nan"), few_shot=False)
        score = extract_score_from_record(record, "test_macro_f1")
        assert score is not None  # NaN is still a valid float


class TestCheckCompleteness:
    """Tests for check_completeness function."""

    def test_complete_matrix(self) -> None:
        """Should return True when all scores present."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 80.0, "sv": 75.0},
            "model2": {"da": 82.0, "sv": 77.0},
        }
        assert check_completeness(model_scores, shots_filter=None) is True

    def test_incomplete_matrix(self) -> None:
        """Should return False when scores missing."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 80.0, "sv": None},
            "model2": {"da": 82.0, "sv": 77.0},
        }
        assert check_completeness(model_scores, shots_filter=None) is False

    def test_empty_models(self) -> None:
        """Should return True for empty models."""
        model_scores: dict[str, dict[str, float | None]] = {}
        assert check_completeness(model_scores, shots_filter=None) is True

    def test_all_none_scores(self) -> None:
        """Should return False when all scores are None."""
        model_scores = {"model1": {"da": None, "sv": None}}
        assert check_completeness(model_scores, shots_filter=None) is False


class TestComputeMaxScore:
    """Tests for compute_max_score function."""

    def test_auto_compute(self) -> None:
        """Should compute max from scores."""
        model_scores = {
            "model1": {"da": 80.0, "sv": 75.0},
            "model2": {"da": 82.0, "sv": 77.0},
        }
        max_score = compute_max_score(model_scores, max_score_override=None)
        assert max_score == 90.0  # Rounded up to nearest 10

    def test_max_score_at_boundary(self) -> None:
        """Should round up at exact boundary."""
        model_scores = {"model1": {"da": 80.0}}
        max_score = compute_max_score(model_scores, max_score_override=None)
        assert max_score == 80.0

    def test_override_valid(self) -> None:
        """Should use override when valid."""
        model_scores = {"model1": {"da": 80.0}}
        max_score = compute_max_score(model_scores, max_score_override=100.0)
        assert max_score == 100.0

    def test_override_too_small(self) -> None:
        """Should raise error when override is too small."""
        model_scores = {"model1": {"da": 85.0}}
        with pytest.raises(ValueError, match="too small"):
            compute_max_score(model_scores, max_score_override=80.0)

    def test_empty_scores(self) -> None:
        """Should return default for empty scores."""
        model_scores: dict[str, dict[str, float]] = {}
        max_score = compute_max_score(model_scores, max_score_override=None)
        assert max_score == 100.0

    def test_infinite_scores_ignored(self) -> None:
        """Should ignore infinite scores."""
        model_scores = {"model1": {"da": float("inf"), "sv": 80.0}}
        max_score = compute_max_score(model_scores, max_score_override=None)
        assert max_score == 80.0


class TestFilterByShots:
    """Tests for filter_by_shots function."""

    def test_filter_zero_shots(self) -> None:
        """Should filter to zero-shot records only."""
        records = [
            make_eee_record("model1", ["da"], 80.0, few_shot=False),
            make_eee_record("model1", ["da"], 85.0, few_shot=True),
        ]
        filtered = filter_by_shots(records, "zero", ["model1"], ["da"])
        assert filtered is not None
        assert len(filtered) == 1
        assert is_few_shot(filtered[0]) is False

    def test_filter_few_shots(self) -> None:
        """Should filter to few-shot records only."""
        records = [
            make_eee_record("model1", ["da"], 80.0, few_shot=False),
            make_eee_record("model1", ["da"], 85.0, few_shot=True),
        ]
        filtered = filter_by_shots(records, "few", ["model1"], ["da"])
        assert filtered is not None
        assert len(filtered) == 1
        assert is_few_shot(filtered[0]) is True

    def test_auto_selects_complete(self) -> None:
        """Auto should select the complete shot setting."""
        # Zero-shot complete, few-shot incomplete
        records = [
            make_eee_record("model1", ["da"], 80.0, few_shot=False),
            make_eee_record("model1", ["sv"], 75.0, few_shot=False),
            # Few-shot only has da
            make_eee_record("model1", ["da"], 85.0, few_shot=True),
        ]
        filtered = filter_by_shots(records, "auto", ["model1"], ["da", "sv"])
        assert filtered is not None
        assert len(filtered) == 2  # Both zero-shot records

    def test_auto_ambiguous(self, caplog: pytest.LogCaptureFixture) -> None:
        """Auto should fail when both are complete."""
        records = [
            make_eee_record("model1", ["da"], 80.0, few_shot=False),
            make_eee_record("model1", ["sv"], 75.0, few_shot=False),
            make_eee_record("model1", ["da"], 85.0, few_shot=True),
            make_eee_record("model1", ["sv"], 82.0, few_shot=True),
        ]
        filtered = filter_by_shots(records, "auto", ["model1"], ["da", "sv"])
        assert filtered is None
        assert "ambiguous" in caplog.records[-1].getMessage().lower()

    def test_auto_neither_complete(self, caplog: pytest.LogCaptureFixture) -> None:
        """Auto should fail when neither is complete."""
        records = [
            make_eee_record("model1", ["da"], 80.0, few_shot=False),
            # Missing sv for zero-shot
            make_eee_record("model1", ["da"], 85.0, few_shot=True),
            # Missing sv for few-shot
        ]
        filtered = filter_by_shots(records, "auto", ["model1"], ["da", "sv"])
        assert filtered is None
        assert "neither" in caplog.records[-1].getMessage().lower()
