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
    _extract_languages_from_record,
    _extract_scores_from_record,
    _filter_by_shots,
    _get_language_display_name,
    _normalise_language_input,
    _resolve_languages,
    cli,
)


def make_eee_record(
    model_name: str,
    languages: list[str],
    scores: dict[str, float],
    few_shot: bool,
    task: str = "summarization",
    dataset: str = "nordjylland-news",
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
            Dict mapping metric names to scores.
        few_shot:
            Whether this is a few-shot record.
        task (optional):
            Task name. Defaults to "summarization".
        dataset (optional):
            Dataset name. Defaults to "nordjylland-news" (Danish summarisation).

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

    return {
        "schema_version": "0.2.1",
        "model_info": {"name": model_name, "id": model_name, "additional_details": {}},
        "eval_library": {
            "name": "euroeval",
            "version": "17.0.0",
            "additional_details": {
                "languages": json.dumps(languages),
                "few_shot": "true" if few_shot else "false",
                "task": task,
                "dataset": dataset,
            },
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
    """Tests for _build_score_matrix function with rank scores."""

    def test_single_model_rank_is_one(self) -> None:
        """Single model should have rank score of 1.0 (best by definition)."""
        records = [make_eee_record("model1", ["da"], {"test_macro_f1": 80.0}, False)]
        matrix = _build_score_matrix(records, ["model1"], ["da"], None)
        # Single model is always best, so rank score = 1.0
        assert matrix["model1"]["da"] == 1.0

    def test_two_models_different_scores(self) -> None:
        """Two models: better score gets rank closer to 1.0."""
        records = [
            make_eee_record("model1", ["da"], {"test_macro_f1": 90.0}, False),
            make_eee_record("model2", ["da"], {"test_macro_f1": 70.0}, False),
        ]
        matrix = _build_score_matrix(records, ["model1", "model2"], ["da"], None)
        # model1 is best (90), so rank = 1.0
        # model2: rank = 1 + (90 - 70) / pooled_std
        # pooled_std = std([90, 70]) = 10*sqrt(2) ≈ 14.14
        # rank2 = 1 + 20/14.14 ≈ 2.41
        assert matrix["model1"]["da"] == 1.0
        assert matrix["model2"]["da"] > 1.0
        assert matrix["model2"]["da"] > matrix["model1"]["da"]

    def test_equal_scores_same_rank(self) -> None:
        """Models with equal scores should have same rank (1.0)."""
        records = [
            make_eee_record("model1", ["da"], {"test_macro_f1": 80.0}, False),
            make_eee_record("model2", ["da"], {"test_macro_f1": 80.0}, False),
        ]
        matrix = _build_score_matrix(records, ["model1", "model2"], ["da"], None)
        # Both tied for best, both get rank = 1.0
        assert matrix["model1"]["da"] == 1.0
        assert matrix["model2"]["da"] == 1.0

    def test_multiple_datasets_aggregate_to_language(self) -> None:
        """Multiple datasets should aggregate to single language rank score."""
        records = [
            make_eee_record(
                "model1",
                ["da"],
                {"test_macro_f1": 90.0},
                False,
                task="summarization",
                dataset="nordjylland-news",
            ),
            make_eee_record(
                "model1",
                ["da"],
                {"test_macro_f1": 70.0},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
            ),
        ]
        matrix = _build_score_matrix(records, ["model1"], ["da"], None)
        # Single model on both datasets, so rank = 1.0 for each
        # Mean across datasets = 1.0
        assert matrix["model1"]["da"] == 1.0

    def test_two_models_two_datasets(self) -> None:
        """Two models on two datasets: aggregation to language level."""
        # Model1: 90 on nordjylland-news, 85 on angry-tweets
        # Model2: 70 on nordjylland-news, 75 on angry-tweets
        records = [
            make_eee_record(
                "model1",
                ["da"],
                {"test_macro_f1": 90.0},
                False,
                task="summarization",
                dataset="nordjylland-news",
            ),
            make_eee_record(
                "model1",
                ["da"],
                {"test_macro_f1": 85.0},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
            ),
            make_eee_record(
                "model2",
                ["da"],
                {"test_macro_f1": 70.0},
                False,
                task="summarization",
                dataset="nordjylland-news",
            ),
            make_eee_record(
                "model2",
                ["da"],
                {"test_macro_f1": 75.0},
                False,
                task="sentiment-classification",
                dataset="angry-tweets",
            ),
        ]
        matrix = _build_score_matrix(records, ["model1", "model2"], ["da"], None)
        # Model1 should have better (lower) rank score than model2
        assert matrix["model1"]["da"] < matrix["model2"]["da"]
        assert matrix["model1"]["da"] >= 1.0

    def test_score_order_independent(self) -> None:
        """Result should be same regardless of record order."""
        records_ordered = [
            make_eee_record("model1", ["da"], {"test_macro_f1": 90.0}, False),
            make_eee_record("model2", ["da"], {"test_macro_f1": 70.0}, False),
        ]
        records_shuffled = [
            make_eee_record("model2", ["da"], {"test_macro_f1": 70.0}, False),
            make_eee_record("model1", ["da"], {"test_macro_f1": 90.0}, False),
        ]
        matrix_ordered = _build_score_matrix(
            records_ordered, ["model1", "model2"], ["da"], None
        )
        matrix_shuffled = _build_score_matrix(
            records_shuffled, ["model1", "model2"], ["da"], None
        )
        assert matrix_ordered["model1"]["da"] == matrix_shuffled["model1"]["da"]
        assert matrix_ordered["model2"]["da"] == matrix_shuffled["model2"]["da"]

    def test_non_finite_scores_ignored(self) -> None:
        """Should ignore NaN and infinite scores."""
        records = [
            make_eee_record("model1", ["da"], {"test_macro_f1": float("nan")}, False),
            make_eee_record("model1", ["sv"], {"test_macro_f1": 75.0}, False),
        ]
        matrix = _build_score_matrix(records, ["model1"], ["da", "sv"], None)
        assert matrix["model1"]["da"] is None
        assert matrix["model1"]["sv"] == 1.0  # Single valid score, rank = 1.0

    def test_missing_dataset_skips_record(self) -> None:
        """Records without dataset field should be skipped."""
        record = make_eee_record("model1", ["da"], {"test_macro_f1": 80.0}, False)
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


class TestCreateSpiderPlot:
    """Tests for _create_spider_plot function."""

    def test_basic_plot_creation(self) -> None:
        """Should create a basic spider plot."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 80.0, "sv": 75.0}
        }
        fig = _create_spider_plot(model_scores, ["da", "sv"], 100.0)
        assert len(fig.data) == 1
        assert fig.layout.title.text == "Language Performance Comparison"

    def test_multiple_models(self) -> None:
        """Should handle multiple models."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 80.0, "sv": 75.0},
            "model2": {"da": 82.0, "sv": 77.0},
        }
        fig = _create_spider_plot(model_scores, ["da", "sv"], 100.0)
        assert len(fig.data) == 2

    def test_axis_always_reversed(self) -> None:
        """Should always reverse radial axis (rank score is lower-is-better)."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 80.0, "sv": 75.0}
        }
        fig = _create_spider_plot(model_scores, ["da", "sv"], 100.0)
        radial_range = fig.layout.polar.radialaxis.range
        assert tuple(radial_range) == (100.0, 0)

    def test_none_scores_treated_as_zero(self) -> None:
        """Should treat None scores as zero in plot."""
        model_scores = {"model1": {"da": 80.0, "sv": None}}
        fig = _create_spider_plot(model_scores, ["da", "sv"], 100.0)
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


class TestIntegrationWithTempFiles:
    """Integration tests using temporary JSONL files."""

    def test_full_pipeline_with_temp_files(self) -> None:
        """Test full pipeline using temp JSONL files."""
        records = [
            make_eee_record(
                "test/model1", ["da", "sv"], {"test_macro_f1": 80.0}, False
            ),
            make_eee_record("test/model1", ["no"], {"test_macro_f1": 75.0}, False),
            make_eee_record(
                "test/model2", ["da", "sv"], {"test_macro_f1": 85.0}, False
            ),
            make_eee_record("test/model2", ["no"], {"test_macro_f1": 78.0}, False),
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
                "test/model1", ["da", "sv"], {"test_macro_f1": 80.0}, False
            ),
            make_eee_record("test/model2", ["da"], {"test_macro_f1": 85.0}, False),
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
            make_eee_record("test/model1", ["da"], {"test_macro_f1": 80.0}, False)
        ]
        few_records = [
            make_eee_record("test/model1", ["da"], {"test_macro_f1": 85.0}, True)
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
                make_eee_record("test/model1", ["da"], {"test_macro_f1": 80.0}, False)
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
        record1 = make_eee_record("test/model1", ["da"], {"test_macro_f1": 80.0}, False)
        record2 = make_eee_record("test/model1", ["sv"], {"test_macro_f1": 75.0}, False)

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
