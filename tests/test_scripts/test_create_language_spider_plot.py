"""Tests for the create_language_spider_plot script."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from src.scripts.create_language_spider_plot import (
    JsonDict,
    _build_score_matrix,
    _compute_language_intersection,
    _compute_max_score,
    _create_spider_plot,
    _default_plot_title,
    _extract_languages_from_record,
    _extract_scores_from_record,
    _filter_by_shots,
    _get_language_display_name,
    _hex_to_rgba,
    _normalise_language_input,
    _resolve_languages,
    cli,
)


@pytest.fixture(autouse=True)
def browser_open_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Prevent tests from opening generated PNG files.

    Args:
        monkeypatch:
            Pytest monkeypatch fixture.

    Returns:
        List of file URIs passed to webbrowser.open.
    """
    calls: list[str] = []

    def fake_open(url: str) -> bool:
        calls.append(url)
        return True

    monkeypatch.setattr(
        "src.scripts.create_language_spider_plot.webbrowser.open", fake_open
    )
    return calls


def make_eee_record(
    model_name: str,
    languages: list[str],
    scores: dict[str, float],
    few_shot: bool,
    task: str = "summarization",
    dataset: str = "nordjylland-news",
    raw_scores: list[float] | None = None,
    metric_name: str = "macro_f1",
) -> JsonDict:
    """Create a minimal EEE-format record for testing.

    Uses real dataset names from the official EuroEval configs so that
    dataset-to-task-to-language mapping works correctly.

    Args:
        model_name:
            Model name/ID.
        languages:
            Language codes.
        scores:
            Dict mapping metric names to aggregated scores.
        few_shot:
            Whether this is a few-shot record.
        task (optional):
            Task name. Defaults to "summarization".
        dataset (optional):
            Dataset name. Defaults to "nordjylland-news" (Danish summarisation).
        raw_scores (optional):
            List of raw per-iteration/bootstrap scores. If provided, these
            are used for rank score computation instead of aggregated scores.
        metric_name (optional):
            Metric name for raw scores. Defaults to "macro_f1".

    Returns:
        EEE-format record dictionary.
    """
    eval_results = [
        {
            "evaluation_name": metric_name,
            "source_data": {"dataset_name": dataset},
            "metric_config": {"lower_is_better": False},
            "score_details": {"score": score, "details": {}},
        }
        for metric_name, score in scores.items()
    ]

    additional_details: dict[str, str] = {
        "languages": json.dumps(languages),
        "few_shot": "true" if few_shot else "false",
        "task": task,
        "dataset": dataset,
    }

    # Add raw results if provided
    if raw_scores is not None:
        raw_results = [
            {f"test_{metric_name}": score, metric_name: score} for score in raw_scores
        ]
        additional_details["raw_results"] = json.dumps(raw_results)

    return {
        "schema_version": "0.2.1",
        "model_info": {"name": model_name, "id": model_name, "additional_details": {}},
        "eval_library": {
            "name": "euroeval",
            "version": "17.0.0",
            "additional_details": additional_details,
        },
        "evaluation_results": eval_results,
    }


class TestNormaliseLanguageInput:
    """Tests for _normalise_language_input function."""

    def test_language_name_danish(self) -> None:
        """Should resolve language name 'danish' to code 'da'."""
        codes = _normalise_language_input("danish")
        assert "da" in codes

    def test_language_code_da(self) -> None:
        """Should resolve language code 'da' to itself."""
        codes = _normalise_language_input("da")
        assert codes == {"da"}

    def test_language_name_norwegian(self) -> None:
        """Should resolve 'norwegian' to 'no'."""
        codes = _normalise_language_input("norwegian")
        assert "no" in codes

    def test_invalid_language_raises(self) -> None:
        """Should raise ValueError for invalid language."""
        with pytest.raises(ValueError, match="Cannot resolve"):
            _normalise_language_input("invalid_language_xyz")

    def test_case_insensitive_code(self) -> None:
        """Should handle uppercase language codes."""
        codes = _normalise_language_input("DA")
        assert codes == {"da"}


class TestResolveLanguages:
    """Tests for _resolve_languages function."""

    def test_default_official_languages(self) -> None:
        """Should return official language codes when no input."""
        languages = _resolve_languages(None)
        assert len(languages) >= 24  # EU has 24 official languages

    def test_explicit_language_names(self) -> None:
        """Should resolve explicit language names to codes."""
        languages = _resolve_languages(["danish", "swedish"])
        assert "da" in languages
        assert "sv" in languages

    def test_explicit_language_codes(self) -> None:
        """Should accept explicit language codes."""
        languages = _resolve_languages(["da", "sv"])
        assert languages == ["da", "sv"]

    def test_mixed_names_and_codes(self) -> None:
        """Should handle mixed language names and codes."""
        languages = _resolve_languages(["danish", "sv"])
        assert "da" in languages
        assert "sv" in languages

    def test_invalid_language_raises(self) -> None:
        """Should raise ValueError for invalid language in list."""
        with pytest.raises(ValueError, match="Cannot resolve"):
            _resolve_languages(["da", "invalid_xyz"])


class TestExtractLanguagesFromRecord:
    """Tests for _extract_languages_from_record function."""

    def test_eee_format_json_string(self) -> None:
        """Should extract languages from EEE JSON-encoded string."""
        record: JsonDict = {
            "eval_library": {
                "additional_details": {"languages": json.dumps(["da", "sv"])}
            }
        }
        languages = _extract_languages_from_record(record)
        assert languages == ["da", "sv"]

    def test_legacy_list_format(self) -> None:
        """Should handle legacy list format."""
        record: JsonDict = {"languages": ["da", "sv"]}
        languages = _extract_languages_from_record(record)
        assert languages == ["da", "sv"]

    def test_legacy_json_string(self) -> None:
        """Should handle legacy JSON string format."""
        record: JsonDict = {"languages": json.dumps(["da", "sv"])}
        languages = _extract_languages_from_record(record)
        assert languages == ["da", "sv"]

    def test_single_language_string(self) -> None:
        """Should handle single language as string."""
        record: JsonDict = {"languages": "da"}
        languages = _extract_languages_from_record(record)
        assert languages == ["da"]

    def test_empty_languages(self) -> None:
        """Should return empty list when no languages."""
        record: JsonDict = {}
        languages = _extract_languages_from_record(record)
        assert languages == []


class TestExtractScoresFromRecord:
    """Tests for _extract_scores_from_record function."""

    def test_extract_single_metric(self) -> None:
        """Should extract single metric score."""
        record: JsonDict = {
            "evaluation_results": [
                {"evaluation_name": "test_macro_f1", "score_details": {"score": 80.0}}
            ]
        }
        scores = _extract_scores_from_record(record)
        assert scores == {"test_macro_f1": 80.0}

    def test_extract_multiple_metrics(self) -> None:
        """Should extract multiple metric scores."""
        record: JsonDict = {
            "evaluation_results": [
                {"evaluation_name": "test_macro_f1", "score_details": {"score": 80.0}},
                {"evaluation_name": "test_accuracy", "score_details": {"score": 85.0}},
            ]
        }
        scores = _extract_scores_from_record(record)
        assert scores == {"test_macro_f1": 80.0, "test_accuracy": 85.0}

    def test_missing_metrics(self) -> None:
        """Should return empty dict for missing metrics."""
        record: JsonDict = {}
        scores = _extract_scores_from_record(record)
        assert scores == {}


class TestFilterByShots:
    """Tests for _filter_by_shots function."""

    def test_filter_zero_shots(self) -> None:
        """Should filter to zero-shot records only."""
        records: list[JsonDict] = [
            make_eee_record("model1", ["da"], {"test_macro_f1": 80.0}, False),
            make_eee_record("model1", ["da"], {"test_macro_f1": 85.0}, True),
        ]
        filtered = _filter_by_shots(records, "zero")
        assert len(filtered) == 1
        few_shot_val = filtered[0].get("eval_library", {})
        assert isinstance(few_shot_val, dict)
        assert few_shot_val.get("additional_details", {}).get("few_shot") == "false"

    def test_filter_few_shots(self) -> None:
        """Should filter to few-shot records only."""
        records: list[JsonDict] = [
            make_eee_record("model1", ["da"], {"test_macro_f1": 80.0}, False),
            make_eee_record("model1", ["da"], {"test_macro_f1": 85.0}, True),
        ]
        filtered = _filter_by_shots(records, "few")
        assert len(filtered) == 1
        few_shot_val = filtered[0].get("eval_library", {})
        assert isinstance(few_shot_val, dict)
        assert few_shot_val.get("additional_details", {}).get("few_shot") == "true"

    def test_auto_selects_zero_when_only_zeros(self) -> None:
        """Auto should select zero-shot when only zeros available."""
        records: list[JsonDict] = [
            make_eee_record("model1", ["da"], {"test_macro_f1": 80.0}, False)
        ]
        filtered = _filter_by_shots(records, "auto")
        assert len(filtered) == 1

    def test_auto_selects_few_when_only_fews(self) -> None:
        """Auto should select few-shot when only fews available."""
        records: list[JsonDict] = [
            make_eee_record("model1", ["da"], {"test_macro_f1": 85.0}, True)
        ]
        filtered = _filter_by_shots(records, "auto")
        assert len(filtered) == 1

    def test_auto_ambiguous_raises(self) -> None:
        """Auto should raise when both zero and few are present."""
        records: list[JsonDict] = [
            make_eee_record("model1", ["da"], {"test_macro_f1": 80.0}, False),
            make_eee_record("model1", ["da"], {"test_macro_f1": 85.0}, True),
        ]
        with pytest.raises(ValueError, match="ambiguous"):
            _filter_by_shots(records, "auto")

    def test_auto_no_shot_metadata_treats_as_few(self) -> None:
        """Auto treats records without few_shot metadata as few-shot (default)."""
        record: JsonDict = {
            "schema_version": "0.2.1",
            "model_info": {"name": "model1", "id": "model1"},
            "eval_library": {
                "name": "euroeval",
                "additional_details": {"languages": json.dumps(["da"])},
            },
        }
        filtered = _filter_by_shots([record], "auto")
        assert len(filtered) == 1


class TestBuildScoreMatrix:
    """Tests for _build_score_matrix function with rank scores.

    Tests use raw per-iteration/bootstrap scores to verify the exact
    EuroEval rank score methodology.
    """

    def test_single_model_rank_is_one(self) -> None:
        """Single model should have rank score of 1.0 (best by definition)."""
        # Single model with raw bootstrap scores for mcc metric
        # mean = 0.8, pooled_std = std([0.8, 0.82, 0.78]) ≈ 0.016
        # best_mean = 0.8, rank = 1 + (0.8 - 0.8) / 0.016 = 1.0
        records = [
            make_eee_record(
                "model1",
                ["da"],
                {"test_mcc": 0.8},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.8, 0.82, 0.78],
                metric_name="mcc",
            )
        ]
        matrix = _build_score_matrix(records, ["model1"], ["da"], None)
        assert matrix["model1"]["da"] == 1.0

    def test_two_models_with_raw_scores(self) -> None:
        """Two models: better mean raw score gets rank closer to 1.0."""
        # Model1: raw scores [0.9, 0.92, 0.88] -> mean = 0.9
        # Model2: raw scores [0.7, 0.72, 0.68] -> mean = 0.7
        # All scores: [0.9, 0.92, 0.88, 0.7, 0.72, 0.68]
        # pooled_std = std(all) ≈ 0.098
        # best_mean = 0.9
        # rank1 = 1 + (0.9 - 0.9) / 0.098 = 1.0
        # rank2 = 1 + (0.9 - 0.7) / 0.098 ≈ 3.04
        records = [
            make_eee_record(
                "model1",
                ["da"],
                {"test_mcc": 0.9},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.9, 0.92, 0.88],
                metric_name="mcc",
            ),
            make_eee_record(
                "model2",
                ["da"],
                {"test_mcc": 0.7},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.7, 0.72, 0.68],
                metric_name="mcc",
            ),
        ]
        matrix = _build_score_matrix(records, ["model1", "model2"], ["da"], None)
        assert matrix["model1"]["da"] == 1.0
        assert matrix["model2"]["da"] > 1.0
        assert matrix["model2"]["da"] > matrix["model1"]["da"]

    def test_equal_raw_scores_same_rank(self) -> None:
        """Models with equal mean raw scores should have same rank (1.0)."""
        # Both models have identical raw scores -> same mean -> same rank
        records = [
            make_eee_record(
                "model1",
                ["da"],
                {"test_mcc": 0.8},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.8, 0.82, 0.78],
                metric_name="mcc",
            ),
            make_eee_record(
                "model2",
                ["da"],
                {"test_mcc": 0.8},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.8, 0.82, 0.78],
                metric_name="mcc",
            ),
        ]
        matrix = _build_score_matrix(records, ["model1", "model2"], ["da"], None)
        assert matrix["model1"]["da"] == 1.0
        assert matrix["model2"]["da"] == 1.0

    def test_multiple_datasets_aggregate_to_language(self) -> None:
        """Multiple datasets should aggregate to single language rank score."""
        records = [
            make_eee_record(
                "model1",
                ["da"],
                {"test_mcc": 0.9},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.9, 0.91, 0.89],
                metric_name="mcc",
            ),
            make_eee_record(
                "model1",
                ["da"],
                {"test_mcc": 0.7},
                False,
                task="linguistic-acceptability",
                dataset="da-language-modeling",
                raw_scores=[0.7, 0.71, 0.69],
                metric_name="mcc",
            ),
        ]
        matrix = _build_score_matrix(records, ["model1"], ["da"], None)
        # Single model on both datasets, so rank = 1.0 for each
        # Mean across datasets = 1.0
        assert matrix["model1"]["da"] == 1.0

    def test_two_models_two_datasets_raw_scores(self) -> None:
        """Two models on two datasets with raw scores."""
        records = [
            make_eee_record(
                "model1",
                ["da"],
                {"test_mcc": 0.9},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.9, 0.91, 0.89],
                metric_name="mcc",
            ),
            make_eee_record(
                "model1",
                ["da"],
                {"test_mcc": 0.85},
                False,
                task="linguistic-acceptability",
                dataset="da-language-modeling",
                raw_scores=[0.85, 0.86, 0.84],
                metric_name="mcc",
            ),
            make_eee_record(
                "model2",
                ["da"],
                {"test_mcc": 0.7},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.7, 0.71, 0.69],
                metric_name="mcc",
            ),
            make_eee_record(
                "model2",
                ["da"],
                {"test_mcc": 0.75},
                False,
                task="linguistic-acceptability",
                dataset="da-language-modeling",
                raw_scores=[0.75, 0.76, 0.74],
                metric_name="mcc",
            ),
        ]
        matrix = _build_score_matrix(records, ["model1", "model2"], ["da"], None)
        assert matrix["model1"]["da"] < matrix["model2"]["da"]
        assert matrix["model1"]["da"] >= 1.0

    def test_rank_scores_not_raw_values(self) -> None:
        """Verify plotted values are rank scores, not raw metric values."""
        # Raw scores are 0.85-0.95 range, but rank scores should be 1.x
        records = [
            make_eee_record(
                "model1",
                ["da"],
                {"test_mcc": 0.95},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.95, 0.96, 0.94],
                metric_name="mcc",
            ),
            make_eee_record(
                "model2",
                ["da"],
                {"test_mcc": 0.85},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.85, 0.86, 0.84],
                metric_name="mcc",
            ),
        ]
        matrix = _build_score_matrix(records, ["model1", "model2"], ["da"], None)
        # Rank scores should be around 1-3, not 0.85-0.95
        assert 1.0 <= matrix["model1"]["da"] < 5.0
        assert 1.0 <= matrix["model2"]["da"] < 5.0
        # Best model should have rank 1.0
        assert matrix["model1"]["da"] == 1.0

    def test_score_order_independent(self) -> None:
        """Result should be same regardless of record order."""
        records_ordered = [
            make_eee_record(
                "model1",
                ["da"],
                {"test_mcc": 0.9},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.9, 0.91, 0.89],
                metric_name="mcc",
            ),
            make_eee_record(
                "model2",
                ["da"],
                {"test_mcc": 0.7},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.7, 0.71, 0.69],
                metric_name="mcc",
            ),
        ]
        records_shuffled = [
            make_eee_record(
                "model2",
                ["da"],
                {"test_mcc": 0.7},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.7, 0.71, 0.69],
                metric_name="mcc",
            ),
            make_eee_record(
                "model1",
                ["da"],
                {"test_mcc": 0.9},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.9, 0.91, 0.89],
                metric_name="mcc",
            ),
        ]
        matrix_ordered = _build_score_matrix(
            records_ordered, ["model1", "model2"], ["da"], None
        )
        matrix_shuffled = _build_score_matrix(
            records_shuffled, ["model1", "model2"], ["da"], None
        )
        assert matrix_ordered["model1"]["da"] == matrix_shuffled["model1"]["da"]
        assert matrix_ordered["model2"]["da"] == matrix_shuffled["model2"]["da"]

    def test_no_raw_scores_returns_none(self) -> None:
        """Records without raw_scores should result in None (no data)."""
        # Without raw_scores, the new methodology can't compute rank scores
        records = [
            make_eee_record(
                "model1",
                ["da"],
                {"test_mcc": 0.8},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
            )
        ]
        matrix = _build_score_matrix(records, ["model1"], ["da"], None)
        assert matrix["model1"]["da"] is None

    def test_missing_dataset_skips_record(self) -> None:
        """Records without dataset field should be skipped."""
        record = make_eee_record(
            "model1",
            ["da"],
            {"test_mcc": 0.8},
            False,
            task="sentiment-classification",
            dataset="angry-tweets",
            raw_scores=[0.8, 0.81, 0.79],
            metric_name="mcc",
        )
        # Remove dataset from all possible locations
        eval_lib = record.get("eval_library", {})
        if isinstance(eval_lib, dict):
            additional = eval_lib.get("additional_details", {})
            if isinstance(additional, dict) and "dataset" in additional:
                del additional["dataset"]
        eval_results = record.get("evaluation_results", [])
        if isinstance(eval_results, list):
            for er in eval_results:
                if isinstance(er, dict):
                    source_data = er.get("source_data", {})
                    if isinstance(source_data, dict) and "dataset_name" in source_data:
                        del source_data["dataset_name"]
        matrix = _build_score_matrix([record], ["model1"], ["da"], None)
        assert matrix["model1"]["da"] is None

    def test_rank_scores_use_all_records_as_reference(self) -> None:
        """Rank scores should use all available records, not just selected models.

        This tests the key requirement: when plotting only model1 and model2,
        the rank scores should be computed relative to ALL models in the
        reference population (including model3), not just model1 and model2.
        """
        # model3 best (0.95), model1 middle (0.8), model2 worst (0.7)
        all_records = [
            make_eee_record(
                "model1",
                ["da"],
                {"test_mcc": 0.8},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.8, 0.82, 0.78],
                metric_name="mcc",
            ),
            make_eee_record(
                "model2",
                ["da"],
                {"test_mcc": 0.7},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.7, 0.72, 0.68],
                metric_name="mcc",
            ),
            make_eee_record(
                "model3",  # Best model, not in selected models
                ["da"],
                {"test_mcc": 0.95},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.95, 0.96, 0.94],
                metric_name="mcc",
            ),
        ]

        # Plot only model1 and model2, but use all three as reference
        matrix = _build_score_matrix(
            all_records=all_records,
            models=["model1", "model2"],
            languages=["da"],
            shot_value=None,
        )

        # Neither model1 nor model2 should have rank 1.0 because model3 is best
        # model1 (0.8) should be better than model2 (0.7), so model1 has lower rank
        assert matrix["model1"]["da"] > 1.0, (
            "model1 should not have rank 1.0 when model3 (0.95) exists"
        )
        assert matrix["model2"]["da"] > 1.0, (
            "model2 should not have rank 1.0 when model3 (0.95) exists"
        )
        assert matrix["model1"]["da"] < matrix["model2"]["da"], (
            "model1 (0.8) should have better rank than model2 (0.7)"
        )

    def test_rank_scores_with_single_selected_but_all_reference(self) -> None:
        """Single selected model should not have rank 1.0 if better models exist.

        When only model1 is selected for plotting but model3 exists in the
        reference population with better scores, model1 should not have rank 1.0.
        """
        all_records = [
            make_eee_record(
                "model1",
                ["da"],
                {"test_mcc": 0.8},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.8, 0.82, 0.78],
                metric_name="mcc",
            ),
            make_eee_record(
                "model3",  # Better model, not selected for plotting
                ["da"],
                {"test_mcc": 0.9},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.9, 0.92, 0.88],
                metric_name="mcc",
            ),
        ]

        # Plot only model1, but use all records as reference
        matrix = _build_score_matrix(
            all_records=all_records,
            models=["model1"],
            languages=["da"],
            shot_value=None,
        )

        # model1 should NOT have rank 1.0 because model3 is better
        assert matrix["model1"]["da"] > 1.0, (
            "Single selected model should not have rank 1.0 "
            "when better models exist in reference population"
        )


class TestComputeLanguageIntersection:
    """Tests for _compute_language_intersection function."""

    def test_complete_intersection(self) -> None:
        """Should return all languages when all models have all scores."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 80.0, "sv": 75.0},
            "model2": {"da": 82.0, "sv": 77.0},
        }
        filtered, languages = _compute_language_intersection(model_scores, ["da", "sv"])
        assert languages == ["da", "sv"]
        assert filtered == model_scores

    def test_partial_intersection(self) -> None:
        """Should return only languages with scores for all models."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 80.0, "sv": 75.0, "no": 70.0},
            "model2": {"da": 82.0, "sv": 77.0, "no": None},
        }
        filtered, languages = _compute_language_intersection(
            model_scores, ["da", "sv", "no"]
        )
        assert languages == ["da", "sv"]
        assert "no" not in filtered["model1"]
        assert "no" not in filtered["model2"]

    def test_empty_intersection(self) -> None:
        """Should return empty list when no common languages."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 80.0, "sv": None},
            "model2": {"da": None, "sv": 77.0},
        }
        filtered, languages = _compute_language_intersection(model_scores, ["da", "sv"])
        assert languages == []

    def test_single_model(self) -> None:
        """Should return all languages with scores for single model."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 80.0, "sv": None, "no": 70.0}
        }
        filtered, languages = _compute_language_intersection(
            model_scores, ["da", "sv", "no"]
        )
        assert languages == ["da", "no"]


class TestComputeMaxScore:
    """Tests for _compute_max_score function."""

    def test_auto_compute_rounds_up(self) -> None:
        """Should compute max from rank scores and round up to nearest 0.5."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 3.2, "sv": 2.8},
            "model2": {"da": 3.6, "sv": 3.0},
        }
        max_score = _compute_max_score(model_scores, max_score_override=None)
        assert max_score == 4.0

    def test_max_score_at_boundary(self) -> None:
        """Should not round up when at exact 0.5 boundary."""
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": 3.5}}
        max_score = _compute_max_score(model_scores, max_score_override=None)
        assert max_score == 3.5

    def test_override_valid(self) -> None:
        """Should use override when valid."""
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": 3.2}}
        max_score = _compute_max_score(model_scores, max_score_override=5.0)
        assert max_score == 5.0

    def test_override_too_small(self) -> None:
        """Should raise error when override is too small."""
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": 3.5}}
        with pytest.raises(ValueError, match="too small"):
            _compute_max_score(model_scores, max_score_override=3.0)

    def test_override_nan_raises(self) -> None:
        """Should reject NaN override."""
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": 80.0}}
        with pytest.raises(ValueError, match="invalid.*finite"):
            _compute_max_score(model_scores, max_score_override=float("nan"))

    def test_override_inf_raises(self) -> None:
        """Should reject infinite override."""
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": 80.0}}
        with pytest.raises(ValueError, match="invalid.*finite"):
            _compute_max_score(model_scores, max_score_override=float("inf"))

    def test_override_negative_raises(self) -> None:
        """Should reject negative override."""
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": 3.0}}
        with pytest.raises(ValueError, match="> 1"):
            _compute_max_score(model_scores, max_score_override=-1.0)

    def test_override_zero_raises(self) -> None:
        """Should reject zero override."""
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": 3.0}}
        with pytest.raises(ValueError, match="> 1"):
            _compute_max_score(model_scores, max_score_override=0.0)

    def test_override_one_raises(self) -> None:
        """Should reject override of exactly 1 (perfect score)."""
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": 3.0}}
        with pytest.raises(ValueError, match="> 1"):
            _compute_max_score(model_scores, max_score_override=1.0)

    def test_min_2_5_applied(self) -> None:
        """Should apply minimum of 2.5 even for very small scores."""
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": 1.5}}
        max_score = _compute_max_score(model_scores, max_score_override=None)
        assert max_score == 2.5

    def test_empty_scores_default(self) -> None:
        """Should return default for empty scores."""
        model_scores: dict[str, dict[str, float | None]] = {}
        max_score = _compute_max_score(model_scores, max_score_override=None)
        assert max_score == 2.5

    def test_none_scores_ignored(self) -> None:
        """Should ignore None scores."""
        model_scores = {"model1": {"da": None, "sv": 3.2}}
        max_score = _compute_max_score(model_scores, max_score_override=None)
        assert max_score == 3.5


class TestHexToRgba:
    """Tests for _hex_to_rgba helper function."""

    def test_hex_to_rgba_default_alpha(self) -> None:
        """Should convert hex to rgba with default alpha 0.2."""
        result = _hex_to_rgba("#1f77b4")
        assert result == "rgba(31, 119, 180, 0.2)"

    def test_hex_to_rgba_custom_alpha(self) -> None:
        """Should convert hex to rgba with custom alpha."""
        result = _hex_to_rgba("#ff7f0e", alpha=0.5)
        assert result == "rgba(255, 127, 14, 0.5)"

    def test_hex_to_rgba_transparency(self) -> None:
        """Should produce transparent fill colours for overlapping traces."""
        result = _hex_to_rgba("#2ca02c", alpha=0.1)
        assert "0.1" in result


class TestDefaultPlotTitle:
    """Tests for _default_plot_title function."""

    def test_zero_shot_title(self) -> None:
        """Should create default zero-shot title."""
        assert _default_plot_title(shot_value=False) == "Zero-shot EuroEval Results"

    def test_few_shot_title(self) -> None:
        """Should create default few-shot title."""
        assert _default_plot_title(shot_value=True) == "Few-shot EuroEval Results"


class TestCreateSpiderPlot:
    """Tests for _create_spider_plot function."""

    def test_basic_plot_creation(self) -> None:
        """Should create a basic spider plot."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 1.5, "sv": 2.0}
        }
        fig = _create_spider_plot(model_scores, ["da", "sv"], 3.0)
        assert len(fig.data) == 1
        assert fig.layout.title.text == "Language Performance Comparison"

    def test_multiple_models(self) -> None:
        """Should handle multiple models."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 1.5, "sv": 2.0},
            "model2": {"da": 1.8, "sv": 2.3},
        }
        fig = _create_spider_plot(model_scores, ["da", "sv"], 3.0)
        assert len(fig.data) == 2

    def test_axis_reversed_with_minimum_one(self) -> None:
        """Should reverse radial axis with minimum 1 (not 0)."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 1.5, "sv": 2.0}
        }
        fig = _create_spider_plot(model_scores, ["da", "sv"], 3.0)
        radial_range = fig.layout.polar.radialaxis.range
        # Range should be [max, 1], not [max, 0]
        assert tuple(radial_range) == (3.0, 1)

    def test_radial_tick_format_one_decimal(self) -> None:
        """Radial axis ticks should show one decimal place."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 1.5, "sv": 2.0}
        }
        fig = _create_spider_plot(model_scores, ["da", "sv"], 3.0)
        tickformat = fig.layout.polar.radialaxis.tickformat
        assert tickformat == ".1f"

    def test_radial_axis_dtick(self) -> None:
        """Radial axis should have 0.5 tick interval."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 1.5, "sv": 2.0}
        }
        fig = _create_spider_plot(model_scores, ["da", "sv"], 3.0)
        dtick = fig.layout.polar.radialaxis.dtick
        assert dtick == 0.5

    def test_transparent_fill_for_readability(self) -> None:
        """Should use transparent fills so overlapping traces remain readable."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 1.5, "sv": 2.0},
            "model2": {"da": 1.8, "sv": 2.3},
        }
        fig = _create_spider_plot(model_scores, ["da", "sv"], 3.0)
        # Check that fill colours use rgba with low alpha
        for trace in fig.data:
            fillcolor = trace.fillcolor
            assert fillcolor.startswith("rgba")
            # Alpha should be 0.2 (transparent)
            assert "0.2" in fillcolor

    def test_logo_added_bottom_right(self) -> None:
        """Should add the EuroEval logo in the bottom-right corner."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 1.5, "sv": 2.0}
        }
        fig = _create_spider_plot(model_scores, ["da", "sv"], 3.0)
        assert len(fig.layout.images) == 1
        logo = fig.layout.images[0]
        assert logo.source.startswith("data:image/png;base64,")
        assert logo.x == 1.22
        assert logo.y == -0.14
        assert logo.xanchor == "right"
        assert logo.yanchor == "bottom"
        assert logo.sizex == 0.36
        assert logo.sizey == 0.36

    def test_none_scores_treated_as_zero(self) -> None:
        """Should treat None scores as zero in plot."""
        model_scores = {"model1": {"da": 1.5, "sv": None}}
        fig = _create_spider_plot(model_scores, ["da", "sv"], 3.0)
        assert len(fig.data) == 1


class TestGetLanguageDisplayName:
    """Tests for _get_language_display_name function."""

    def test_known_code_returns_name(self) -> None:
        """Should return language name for known code."""
        name = _get_language_display_name("da")
        assert name == "Danish"

    def test_unknown_code_returns_code(self) -> None:
        """Should return code itself for unknown code."""
        name = _get_language_display_name("xyz")
        assert name == "xyz"


class TestClickCLI:
    """Integration tests for the Click CLI."""

    def test_cli_requires_model_option(self) -> None:
        """CLI should require --model option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--language", "da"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_cli_invalid_language(self) -> None:
        """CLI should fail gracefully for invalid language."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--model", "test-model", "--language", "invalid_xyz"]
        )
        assert result.exit_code != 0
        assert "Cannot resolve" in result.output

    def test_cli_no_metric_option(self) -> None:
        """CLI should not have --metric option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "--metric" not in result.output

    def test_cli_no_lower_is_better_option(self) -> None:
        """CLI should not have --lower-is-better option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "--lower-is-better" not in result.output

    def test_cli_max_score_help_documents_auto(self) -> None:
        """CLI help for --max-score should document automatic computation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "--max-score" in result.output
        assert (
            "automatically" in result.output.lower() or "auto" in result.output.lower()
        )

    def test_cli_max_score_help_documents_optional(self) -> None:
        """CLI help for --max-score should document it is optional."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "--max-score" in result.output
        assert "optional" in result.output.lower() or "omitted" in result.output.lower()

    def test_cli_success_single_line_output(self) -> None:
        """Successful CLI invocation should produce exactly one stdout line."""
        record = make_eee_record(
            "test/model",
            ["da"],
            {"test_mcc": 0.8},
            False,
            task="sentiment-classification",
            dataset="angry-tweets",
            raw_scores=[0.8, 0.82, 0.78],
            metric_name="mcc",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model.jsonl"
            jsonl_path.write_text(json.dumps(record) + "\n")

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                with tempfile.TemporaryDirectory() as workdir:
                    original_cwd = Path.cwd()
                    try:
                        os.chdir(workdir)
                        result = runner.invoke(
                            cli,
                            [
                                "--model",
                                "test/model",
                                "--language",
                                "da",
                                "--shots",
                                "zero",
                            ],
                        )
                        assert result.exit_code == 0, f"CLI failed: {result.output}"
                        lines = [
                            line for line in result.output.strip().split("\n") if line
                        ]
                        assert len(lines) == 1, (
                            f"Expected 1 line, got {len(lines)}: {lines}"
                        )
                    finally:
                        os.chdir(original_cwd)

    def test_cli_success_output_contains_file_uri(self) -> None:
        """Successful output should contain PNG filename or file:// URI."""
        record = make_eee_record(
            "test/model",
            ["da"],
            {"test_mcc": 0.8},
            False,
            task="sentiment-classification",
            dataset="angry-tweets",
            raw_scores=[0.8, 0.82, 0.78],
            metric_name="mcc",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model.jsonl"
            jsonl_path.write_text(json.dumps(record) + "\n")

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                with tempfile.TemporaryDirectory() as workdir:
                    original_cwd = Path.cwd()
                    try:
                        os.chdir(workdir)
                        result = runner.invoke(
                            cli,
                            [
                                "--model",
                                "test/model",
                                "--language",
                                "da",
                                "--shots",
                                "zero",
                            ],
                        )
                        assert result.exit_code == 0, f"CLI failed: {result.output}"
                        assert (
                            "language-spider-plot.png" in result.output
                            or "file://" in result.output
                        ), f"Output should contain filename or URI: {result.output}"
                    finally:
                        os.chdir(original_cwd)


class TestIntegrationWithTempFiles:
    """Integration tests using temporary JSONL files."""

    def test_full_pipeline_with_temp_files(self) -> None:
        """Test full pipeline using temp JSONL files."""
        records = [
            make_eee_record(
                "test/model1",
                ["da", "sv"],
                {"test_mcc": 0.8},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.8, 0.82, 0.78],
                metric_name="mcc",
            ),
            make_eee_record(
                "test/model1",
                ["no"],
                {"test_mcc": 0.75},
                False,
                task="sentiment-classification",
                dataset="norec",
                raw_scores=[0.75, 0.76, 0.74],
                metric_name="mcc",
            ),
            make_eee_record(
                "test/model2",
                ["da", "sv"],
                {"test_mcc": 0.85},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.85, 0.86, 0.84],
                metric_name="mcc",
            ),
            make_eee_record(
                "test/model2",
                ["no"],
                {"test_mcc": 0.78},
                False,
                task="sentiment-classification",
                dataset="norec",
                raw_scores=[0.78, 0.79, 0.77],
                metric_name="mcc",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model1.jsonl"
            jsonl_path.write_text("\n".join(json.dumps(r) for r in records[:2]) + "\n")

            jsonl_path2 = Path(tmpdir) / "test_model2.jsonl"
            jsonl_path2.write_text("\n".join(json.dumps(r) for r in records[2:]) + "\n")

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                with tempfile.TemporaryDirectory() as workdir:
                    original_cwd = Path.cwd()
                    try:
                        os.chdir(workdir)
                        result = runner.invoke(
                            cli,
                            [
                                "--model",
                                "test/model1",
                                "--model",
                                "test/model2",
                                "--language",
                                "da",
                                "--language",
                                "sv",
                                "--shots",
                                "zero",
                            ],
                        )
                        assert result.exit_code == 0, f"CLI failed: {result.output}"
                        output_path = Path(workdir) / "language-spider-plot.png"
                        assert output_path.exists()
                    finally:
                        os.chdir(original_cwd)

    def test_intersection_used_for_missing_languages(self) -> None:
        """Test that language intersection is used when some models miss languages."""
        records = [
            make_eee_record(
                "test/model1",
                ["da", "sv"],
                {"test_mcc": 0.8},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.8, 0.82, 0.78],
                metric_name="mcc",
            ),
            make_eee_record(
                "test/model2",
                ["da"],
                {"test_mcc": 0.85},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.85, 0.86, 0.84],
                metric_name="mcc",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model1.jsonl"
            jsonl_path.write_text(json.dumps(records[0]) + "\n")

            jsonl_path2 = Path(tmpdir) / "test_model2.jsonl"
            jsonl_path2.write_text(json.dumps(records[1]) + "\n")

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                with tempfile.TemporaryDirectory() as workdir:
                    original_cwd = Path.cwd()
                    try:
                        os.chdir(workdir)
                        result = runner.invoke(
                            cli,
                            [
                                "--model",
                                "test/model1",
                                "--model",
                                "test/model2",
                                "--language",
                                "da",
                                "--language",
                                "sv",
                                "--shots",
                                "zero",
                            ],
                        )
                        assert result.exit_code == 0, f"CLI failed: {result.output}"
                        output_path = Path(workdir) / "language-spider-plot.png"
                        assert output_path.exists()
                    finally:
                        os.chdir(original_cwd)

    def test_empty_intersection_fails(self) -> None:
        """Test that empty language intersection fails with clear message."""
        records = [
            make_eee_record("test/model1", ["da"], {"test_macro_f1": 80.0}, False),
            make_eee_record("test/model2", ["sv"], {"test_macro_f1": 85.0}, False),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model1.jsonl"
            jsonl_path.write_text(json.dumps(records[0]) + "\n")

            jsonl_path2 = Path(tmpdir) / "test_model2.jsonl"
            jsonl_path2.write_text(json.dumps(records[1]) + "\n")

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                result = runner.invoke(
                    cli,
                    [
                        "--model",
                        "test/model1",
                        "--model",
                        "test/model2",
                        "--language",
                        "da",
                        "--language",
                        "sv",
                        "--shots",
                        "zero",
                    ],
                )
                assert result.exit_code != 0
                assert "No languages have scores for all" in result.output

    def test_shots_auto_detection(self) -> None:
        """Test auto shot detection with temp files."""
        zero_records = [
            make_eee_record(
                "test/model1",
                ["da"],
                {"test_mcc": 0.8},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.8, 0.82, 0.78],
                metric_name="mcc",
            )
        ]
        few_records = [
            make_eee_record(
                "test/model1",
                ["da"],
                {"test_mcc": 0.85},
                True,
                task="sentiment-classification",
                dataset="angry-tweets",
                raw_scores=[0.85, 0.86, 0.84],
                metric_name="mcc",
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model1.jsonl"
            jsonl_path.write_text(
                "\n".join(json.dumps(r) for r in zero_records + few_records) + "\n"
            )

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()

                with tempfile.TemporaryDirectory() as workdir:
                    original_cwd = Path.cwd()
                    try:
                        os.chdir(workdir)
                        result = runner.invoke(
                            cli,
                            [
                                "--model",
                                "test/model1",
                                "--language",
                                "da",
                                "--shots",
                                "auto",
                            ],
                        )
                        assert result.exit_code != 0
                        assert "ambiguous" in result.output.lower()
                    finally:
                        os.chdir(original_cwd)

                with tempfile.TemporaryDirectory() as workdir:
                    original_cwd = Path.cwd()
                    try:
                        os.chdir(workdir)
                        result = runner.invoke(
                            cli,
                            [
                                "--model",
                                "test/model1",
                                "--language",
                                "da",
                                "--shots",
                                "zero",
                            ],
                        )
                        assert result.exit_code == 0, f"CLI failed: {result.output}"
                    finally:
                        os.chdir(original_cwd)

    def test_cli_no_output_option(self) -> None:
        """CLI should not have --output option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "--output" not in result.output
        assert "-o" not in result.output

    def test_result_files_not_mutated(self) -> None:
        """Test that result JSONL files are not mutated."""
        original_content = (
            json.dumps(
                make_eee_record(
                    "test/model1",
                    ["da"],
                    {"test_mcc": 0.8},
                    False,
                    task="sentiment-classification",
                    dataset="angry-tweets",
                    raw_scores=[0.8, 0.82, 0.78],
                    metric_name="mcc",
                )
            )
            + "\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model1.jsonl"
            jsonl_path.write_text(original_content)

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                with tempfile.TemporaryDirectory() as workdir:
                    original_cwd = Path.cwd()
                    try:
                        os.chdir(workdir)
                        result = runner.invoke(
                            cli,
                            [
                                "--model",
                                "test/model1",
                                "--language",
                                "da",
                                "--shots",
                                "zero",
                            ],
                        )
                        assert result.exit_code == 0
                    finally:
                        os.chdir(original_cwd)

            content_after = jsonl_path.read_text()
            assert content_after == original_content

    def test_concatenated_jsonl_objects(self) -> None:
        """Test handling of concatenated }{ JSON objects.

        This tests the parse_jsonl_lines ability to handle malformed JSONL
        where multiple JSON objects appear on a single line without newlines.
        """
        record1 = make_eee_record(
            "test/model1",
            ["da"],
            {"test_mcc": 0.8},
            False,
            task="sentiment-classification",
            dataset="angry-tweets",
            raw_scores=[0.8, 0.82, 0.78],
            metric_name="mcc",
        )
        record2 = make_eee_record(
            "test/model1",
            ["sv"],
            {"test_mcc": 0.75},
            False,
            task="sentiment-classification",
            dataset="swerec",
            raw_scores=[0.75, 0.76, 0.74],
            metric_name="mcc",
        )

        concatenated = json.dumps(record1) + json.dumps(record2)

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model1.jsonl"
            jsonl_path.write_text(concatenated)

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                with tempfile.TemporaryDirectory() as workdir:
                    original_cwd = Path.cwd()
                    try:
                        os.chdir(workdir)
                        result = runner.invoke(
                            cli,
                            [
                                "--model",
                                "test/model1",
                                "--language",
                                "da",
                                "--language",
                                "sv",
                                "--shots",
                                "zero",
                            ],
                        )
                        assert result.exit_code == 0, f"CLI failed: {result.output}"
                    finally:
                        os.chdir(original_cwd)


class TestTitleAndFilenameOptions:
    """Tests for --title and --filename CLI options."""

    def test_cli_title_option_sets_plot_title(self) -> None:
        """CLI --title option should set the plot title."""
        record = make_eee_record(
            "test/model",
            ["da"],
            {"test_mcc": 0.8},
            False,
            task="sentiment-classification",
            dataset="angry-tweets",
            raw_scores=[0.8, 0.82, 0.78],
            metric_name="mcc",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model.jsonl"
            jsonl_path.write_text(json.dumps(record) + "\n")

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                with tempfile.TemporaryDirectory() as workdir:
                    original_cwd = Path.cwd()
                    try:
                        os.chdir(workdir)
                        result = runner.invoke(
                            cli,
                            [
                                "--model",
                                "test/model",
                                "--language",
                                "da",
                                "--shots",
                                "zero",
                                "--title",
                                "My Custom Plot Title",
                            ],
                        )
                        assert result.exit_code == 0, f"CLI failed: {result.output}"
                    finally:
                        os.chdir(original_cwd)

    def test_cli_filename_option_sets_output_file(self) -> None:
        """CLI --filename option should set the output filename."""
        record = make_eee_record(
            "test/model",
            ["da"],
            {"test_mcc": 0.8},
            False,
            task="sentiment-classification",
            dataset="angry-tweets",
            raw_scores=[0.8, 0.82, 0.78],
            metric_name="mcc",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model.jsonl"
            jsonl_path.write_text(json.dumps(record) + "\n")

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                with tempfile.TemporaryDirectory() as workdir:
                    original_cwd = Path.cwd()
                    try:
                        os.chdir(workdir)
                        result = runner.invoke(
                            cli,
                            [
                                "--model",
                                "test/model",
                                "--language",
                                "da",
                                "--shots",
                                "zero",
                                "--filename",
                                "custom-plot.png",
                            ],
                        )
                        assert result.exit_code == 0, f"CLI failed: {result.output}"
                        output_path = Path(workdir) / "custom-plot.png"
                        assert output_path.exists(), "Custom filename should be created"
                    finally:
                        os.chdir(original_cwd)

    def test_cli_filename_without_png_extension(self) -> None:
        """CLI --filename without .png should have it appended."""
        record = make_eee_record(
            "test/model",
            ["da"],
            {"test_mcc": 0.8},
            False,
            task="sentiment-classification",
            dataset="angry-tweets",
            raw_scores=[0.8, 0.82, 0.78],
            metric_name="mcc",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model.jsonl"
            jsonl_path.write_text(json.dumps(record) + "\n")

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                with tempfile.TemporaryDirectory() as workdir:
                    original_cwd = Path.cwd()
                    try:
                        os.chdir(workdir)
                        result = runner.invoke(
                            cli,
                            [
                                "--model",
                                "test/model",
                                "--language",
                                "da",
                                "--shots",
                                "zero",
                                "--filename",
                                "myplot",
                            ],
                        )
                        assert result.exit_code == 0, f"CLI failed: {result.output}"
                        output_path = Path(workdir) / "myplot.png"
                        assert output_path.exists(), ".png should be appended"
                    finally:
                        os.chdir(original_cwd)

    def test_cli_title_infers_filename_snake_case(self) -> None:
        """CLI --title without --filename should infer filename using snake_case."""
        record = make_eee_record(
            "test/model",
            ["da"],
            {"test_mcc": 0.8},
            False,
            task="sentiment-classification",
            dataset="angry-tweets",
            raw_scores=[0.8, 0.82, 0.78],
            metric_name="mcc",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model.jsonl"
            jsonl_path.write_text(json.dumps(record) + "\n")

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                with tempfile.TemporaryDirectory() as workdir:
                    original_cwd = Path.cwd()
                    try:
                        os.chdir(workdir)
                        result = runner.invoke(
                            cli,
                            [
                                "--model",
                                "test/model",
                                "--language",
                                "da",
                                "--shots",
                                "zero",
                                "--title",
                                "My Plot Title",
                            ],
                        )
                        assert result.exit_code == 0, f"CLI failed: {result.output}"
                        # Title "My Plot Title" -> "my_plot_title.png"
                        output_path = Path(workdir) / "my_plot_title.png"
                        assert output_path.exists(), (
                            "Filename should be inferred from title as snake_case"
                        )
                    finally:
                        os.chdir(original_cwd)

    def test_cli_title_special_chars_infers_filename(self) -> None:
        """CLI --title with special chars should infer clean snake_case filename."""
        record = make_eee_record(
            "test/model",
            ["da"],
            {"test_mcc": 0.8},
            False,
            task="sentiment-classification",
            dataset="angry-tweets",
            raw_scores=[0.8, 0.82, 0.78],
            metric_name="mcc",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model.jsonl"
            jsonl_path.write_text(json.dumps(record) + "\n")

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                with tempfile.TemporaryDirectory() as workdir:
                    original_cwd = Path.cwd()
                    try:
                        os.chdir(workdir)
                        result = runner.invoke(
                            cli,
                            [
                                "--model",
                                "test/model",
                                "--language",
                                "da",
                                "--shots",
                                "zero",
                                "--title",
                                "My Plot! Title: 2024 (Test)",
                            ],
                        )
                        assert result.exit_code == 0, f"CLI failed: {result.output}"
                        # Title with special chars -> "my_plot_title_2024_test.png"
                        output_path = Path(workdir) / "my_plot_title_2024_test.png"
                        assert output_path.exists(), (
                            "Special chars should be removed from inferred filename"
                        )
                    finally:
                        os.chdir(original_cwd)

    def test_cli_default_filename_without_options(self) -> None:
        """CLI without --title or --filename should use default filename."""
        record = make_eee_record(
            "test/model",
            ["da"],
            {"test_mcc": 0.8},
            False,
            task="sentiment-classification",
            dataset="angry-tweets",
            raw_scores=[0.8, 0.82, 0.78],
            metric_name="mcc",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model.jsonl"
            jsonl_path.write_text(json.dumps(record) + "\n")

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                with tempfile.TemporaryDirectory() as workdir:
                    original_cwd = Path.cwd()
                    try:
                        os.chdir(workdir)
                        result = runner.invoke(
                            cli,
                            [
                                "--model",
                                "test/model",
                                "--language",
                                "da",
                                "--shots",
                                "zero",
                            ],
                        )
                        assert result.exit_code == 0, f"CLI failed: {result.output}"
                        output_path = Path(workdir) / "language-spider-plot.png"
                        assert output_path.exists(), "Default filename should be used"
                    finally:
                        os.chdir(original_cwd)

    def test_cli_success_output_is_single_line_with_file_uri(
        self, browser_open_calls: list[str]
    ) -> None:
        """Successful output should be exactly one line with file:// URI."""
        record = make_eee_record(
            "test/model",
            ["da"],
            {"test_mcc": 0.8},
            False,
            task="sentiment-classification",
            dataset="angry-tweets",
            raw_scores=[0.8, 0.82, 0.78],
            metric_name="mcc",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model.jsonl"
            jsonl_path.write_text(json.dumps(record) + "\n")

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                with tempfile.TemporaryDirectory() as workdir:
                    original_cwd = Path.cwd()
                    try:
                        os.chdir(workdir)
                        result = runner.invoke(
                            cli,
                            [
                                "--model",
                                "test/model",
                                "--language",
                                "da",
                                "--shots",
                                "zero",
                                "--title",
                                "Test Title",
                            ],
                        )
                        assert result.exit_code == 0, f"CLI failed: {result.output}"
                        lines = [
                            line for line in result.output.strip().split("\n") if line
                        ]
                        assert len(lines) == 1, (
                            f"Expected exactly 1 line, got {len(lines)}: {lines}"
                        )
                        assert lines[0].startswith("Finished. "), (
                            f"Output should announce completion: {lines[0]}"
                        )
                        assert "output plot can now be found at" in lines[0]
                        assert "file://" in lines[0], (
                            f"Output should include file:// URI: {lines[0]}"
                        )
                        assert lines[0].endswith(".png"), (
                            f"URI should end with .png: {lines[0]}"
                        )
                        assert len(browser_open_calls) == 1
                        assert browser_open_calls[0] in lines[0]
                    finally:
                        os.chdir(original_cwd)

    def test_cli_shots_auto_no_extra_output(self) -> None:
        """CLI --shots auto should not emit extra logs on success."""
        record = make_eee_record(
            "test/model",
            ["da"],
            {"test_mcc": 0.8},
            False,
            task="sentiment-classification",
            dataset="angry-tweets",
            raw_scores=[0.8, 0.82, 0.78],
            metric_name="mcc",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model.jsonl"
            jsonl_path.write_text(json.dumps(record) + "\n")

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                with tempfile.TemporaryDirectory() as workdir:
                    original_cwd = Path.cwd()
                    try:
                        os.chdir(workdir)
                        result = runner.invoke(
                            cli,
                            [
                                "--model",
                                "test/model",
                                "--language",
                                "da",
                                "--shots",
                                "auto",
                            ],
                        )
                        assert result.exit_code == 0, f"CLI failed: {result.output}"
                        lines = [
                            line for line in result.output.strip().split("\n") if line
                        ]
                        assert len(lines) == 1, (
                            f"--shots auto should emit exactly 1 line, got {len(lines)}"
                        )
                    finally:
                        os.chdir(original_cwd)
