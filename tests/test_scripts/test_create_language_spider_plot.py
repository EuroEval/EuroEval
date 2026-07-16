"""Tests for the create_language_spider_plot script."""

from __future__ import annotations

import json
import tempfile
import typing as t
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from src.scripts.create_language_spider_plot import (
    _build_score_matrix,
    _check_completeness,
    _compute_max_score,
    _create_spider_plot,
    _extract_languages_from_record,
    _extract_scores_from_record,
    _filter_by_shots,
    _get_language_display_name,
    _normalise_language_input,
    _resolve_languages,
    main,
)


def make_eee_record(
    model_name: str,
    languages: list[str],
    scores: dict[str, float],
    few_shot: bool,
    task: str = "summarization",
) -> dict[str, t.Any]:
    """Create a minimal EEE-format record for testing.

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

    Returns:
        EEE-format record dictionary.
    """
    eval_results = [
        {
            "evaluation_name": metric_name,
            "source_data": {"dataset_name": "test-dataset"},
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
        assert "da" in codes


class TestResolveLanguages:
    """Tests for _resolve_languages function."""

    def test_default_official_languages(self) -> None:
        """Should return official language codes when no input."""
        languages = _resolve_languages(None)
        assert len(languages) > 0
        assert all(isinstance(lang, str) for lang in languages)
        assert languages == sorted(languages)

    def test_explicit_language_names(self) -> None:
        """Should resolve explicit language names to codes."""
        languages = _resolve_languages(["danish", "swedish"])
        assert "da" in languages
        assert "sv" in languages

    def test_explicit_language_codes(self) -> None:
        """Should accept explicit language codes."""
        languages = _resolve_languages(["da", "sv"])
        assert "da" in languages
        assert "sv" in languages

    def test_mixed_names_and_codes(self) -> None:
        """Should handle mixed language names and codes."""
        languages = _resolve_languages(["danish", "sv"])
        assert "da" in languages
        assert "sv" in languages

    def test_invalid_language_raises(self) -> None:
        """Should raise ValueError for invalid language in list."""
        with pytest.raises(ValueError, match="Cannot resolve"):
            _resolve_languages(["danish", "invalid_xyz"])


class TestExtractLanguagesFromRecord:
    """Tests for _extract_languages_from_record function."""

    def test_eee_format_json_string(self) -> None:
        """Should extract languages from EEE JSON-encoded string."""
        record = make_eee_record(
            "test-model", ["da", "sv"], {"test_macro_f1": 80.0}, False
        )
        languages = _extract_languages_from_record(record)
        assert languages == ["da", "sv"]

    def test_legacy_list_format(self) -> None:
        """Should handle legacy list format."""
        record = {
            "model_info": {"name": "test"},
            "eval_library": {},
            "languages": ["da", "sv"],
        }
        languages = _extract_languages_from_record(record)
        assert languages == ["da", "sv"]

    def test_legacy_json_string(self) -> None:
        """Should handle legacy JSON string format."""
        record = {
            "model_info": {"name": "test"},
            "eval_library": {},
            "languages": '["da", "sv"]',
        }
        languages = _extract_languages_from_record(record)
        assert languages == ["da", "sv"]

    def test_single_language_string(self) -> None:
        """Should handle single language as string."""
        record = {"model_info": {"name": "test"}, "eval_library": {}, "languages": "da"}
        languages = _extract_languages_from_record(record)
        assert languages == ["da"]

    def test_empty_languages(self) -> None:
        """Should return empty list when no languages."""
        record = {"model_info": {"name": "test"}, "eval_library": {}}
        languages = _extract_languages_from_record(record)
        assert languages == []


class TestExtractScoresFromRecord:
    """Tests for _extract_scores_from_record function."""

    def test_extract_single_metric(self) -> None:
        """Should extract single metric score."""
        record = make_eee_record("test-model", ["da"], {"test_macro_f1": 85.5}, False)
        scores = _extract_scores_from_record(record)
        assert scores == {"test_macro_f1": 85.5}

    def test_extract_multiple_metrics(self) -> None:
        """Should extract multiple metric scores."""
        record = make_eee_record(
            "test-model", ["da"], {"test_macro_f1": 85.5, "test_accuracy": 90.0}, False
        )
        scores = _extract_scores_from_record(record)
        assert scores == {"test_macro_f1": 85.5, "test_accuracy": 90.0}

    def test_missing_metrics(self) -> None:
        """Should return empty dict for missing metrics."""
        record = {"model_info": {"name": "test"}, "eval_library": {}}
        scores = _extract_scores_from_record(record)
        assert scores == {}


class TestFilterByShots:
    """Tests for _filter_by_shots function."""

    def test_filter_zero_shots(self) -> None:
        """Should filter to zero-shot records only."""
        records = [
            make_eee_record("model1", ["da"], {"test_macro_f1": 80.0}, few_shot=False),
            make_eee_record("model1", ["da"], {"test_macro_f1": 85.0}, few_shot=True),
        ]
        filtered = _filter_by_shots(records, "zero")
        assert len(filtered) == 1
        assert filtered[0]["eval_library"]["additional_details"]["few_shot"] == "false"

    def test_filter_few_shots(self) -> None:
        """Should filter to few-shot records only."""
        records = [
            make_eee_record("model1", ["da"], {"test_macro_f1": 80.0}, few_shot=False),
            make_eee_record("model1", ["da"], {"test_macro_f1": 85.0}, few_shot=True),
        ]
        filtered = _filter_by_shots(records, "few")
        assert len(filtered) == 1
        assert filtered[0]["eval_library"]["additional_details"]["few_shot"] == "true"

    def test_auto_selects_zero_when_only_zeros(self) -> None:
        """Auto should select zero-shot when only zeros available."""
        records = [
            make_eee_record("model1", ["da"], {"test_macro_f1": 80.0}, few_shot=False),
            make_eee_record("model1", ["sv"], {"test_macro_f1": 75.0}, few_shot=False),
        ]
        filtered = _filter_by_shots(records, "auto")
        assert len(filtered) == 2

    def test_auto_selects_few_when_only_fews(self) -> None:
        """Auto should select few-shot when only fews available."""
        records = [
            make_eee_record("model1", ["da"], {"test_macro_f1": 85.0}, few_shot=True),
            make_eee_record("model1", ["sv"], {"test_macro_f1": 82.0}, few_shot=True),
        ]
        filtered = _filter_by_shots(records, "auto")
        assert len(filtered) == 2

    def test_auto_ambiguous_raises(self) -> None:
        """Auto should raise when both zero and few are present."""
        records = [
            make_eee_record("model1", ["da"], {"test_macro_f1": 80.0}, few_shot=False),
            make_eee_record("model1", ["da"], {"test_macro_f1": 85.0}, few_shot=True),
        ]
        with pytest.raises(ValueError, match="ambiguous"):
            _filter_by_shots(records, "auto")

    def test_auto_no_shot_metadata_treats_as_few(self) -> None:
        """Auto treats records without few_shot metadata as few-shot (default).

        The get_few_shot helper defaults to True when metadata is missing.
        """
        records = [
            {"model_info": {"name": "test"}, "eval_library": {"additional_details": {}}}
        ]
        filtered = _filter_by_shots(records, "auto")
        assert len(filtered) == 1


class TestBuildScoreMatrix:
    """Tests for _build_score_matrix function."""

    def test_build_complete_matrix(self) -> None:
        """Should build complete score matrix."""
        records = [
            make_eee_record("model1", ["da"], {"test_macro_f1": 80.0}, False),
            make_eee_record("model1", ["sv"], {"test_macro_f1": 75.0}, False),
        ]
        matrix = _build_score_matrix(
            records, ["model1"], ["da", "sv"], "test_macro_f1", False
        )
        assert matrix["model1"]["da"] == 80.0
        assert matrix["model1"]["sv"] == 75.0

    def test_first_score_wins_no_replacement(self) -> None:
        """Should use first valid score, not replace with None from later records."""
        records = [
            make_eee_record("model1", ["da"], {"test_macro_f1": 80.0}, False),
            make_eee_record("model1", ["da"], {}, False),
        ]
        matrix = _build_score_matrix(
            records, ["model1"], ["da"], "test_macro_f1", False
        )
        assert matrix["model1"]["da"] == 80.0

    def test_primary_metric_fallback(self) -> None:
        """Should fallback to standard metrics when using 'primary'."""
        records = [
            make_eee_record(
                "model1", ["da"], {"test_accuracy": 90.0}, False, task="summarization"
            )
        ]
        matrix = _build_score_matrix(records, ["model1"], ["da"], "primary", None)
        assert matrix["model1"]["da"] == 90.0

    def test_non_finite_scores_ignored(self) -> None:
        """Should ignore NaN and infinite scores."""
        records = [
            make_eee_record("model1", ["da"], {"test_macro_f1": float("nan")}, False),
            make_eee_record("model1", ["sv"], {"test_macro_f1": 75.0}, False),
        ]
        matrix = _build_score_matrix(
            records, ["model1"], ["da", "sv"], "test_macro_f1", None
        )
        assert matrix["model1"]["da"] is None
        assert matrix["model1"]["sv"] == 75.0


class TestCheckCompleteness:
    """Tests for _check_completeness function."""

    def test_complete_matrix(self) -> None:
        """Should return True when all scores present."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 80.0, "sv": 75.0},
            "model2": {"da": 82.0, "sv": 77.0},
        }
        assert _check_completeness(model_scores) is True

    def test_incomplete_matrix(self) -> None:
        """Should return False when scores missing."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 80.0, "sv": None},
            "model2": {"da": 82.0, "sv": 77.0},
        }
        assert _check_completeness(model_scores) is False

    def test_empty_models(self) -> None:
        """Should return True for empty models."""
        model_scores: dict[str, dict[str, float | None]] = {}
        assert _check_completeness(model_scores) is True

    def test_all_none_scores(self) -> None:
        """Should return False when all scores are None."""
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": None, "sv": None}}
        assert _check_completeness(model_scores) is False


class TestComputeMaxScore:
    """Tests for _compute_max_score function."""

    def test_auto_compute_rounds_up(self) -> None:
        """Should compute max from scores and round up to nearest 10."""
        model_scores: dict[str, dict[str, float | None]] = {
            "model1": {"da": 80.0, "sv": 75.0},
            "model2": {"da": 82.0, "sv": 77.0},
        }
        max_score = _compute_max_score(model_scores, max_score_override=None)
        assert max_score == 90.0

    def test_max_score_at_boundary(self) -> None:
        """Should not round up when at exact boundary."""
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": 80.0}}
        max_score = _compute_max_score(model_scores, max_score_override=None)
        assert max_score == 80.0

    def test_override_valid(self) -> None:
        """Should use override when valid."""
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": 80.0}}
        max_score = _compute_max_score(model_scores, max_score_override=100.0)
        assert max_score == 100.0

    def test_override_too_small(self) -> None:
        """Should raise error when override is too small."""
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": 85.0}}
        with pytest.raises(ValueError, match="too small"):
            _compute_max_score(model_scores, max_score_override=80.0)

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
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": 80.0}}
        with pytest.raises(ValueError, match="positive"):
            _compute_max_score(model_scores, max_score_override=-10.0)

    def test_override_zero_raises(self) -> None:
        """Should reject zero override."""
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": 80.0}}
        with pytest.raises(ValueError, match="positive"):
            _compute_max_score(model_scores, max_score_override=0.0)

    def test_empty_scores_default(self) -> None:
        """Should return default for empty scores."""
        model_scores: dict[str, dict[str, float | None]] = {}
        max_score = _compute_max_score(model_scores, max_score_override=None)
        assert max_score == 100.0

    def test_none_scores_ignored(self) -> None:
        """Should ignore None scores."""
        model_scores = {"model1": {"da": None, "sv": 80.0}}
        max_score = _compute_max_score(model_scores, max_score_override=None)
        assert max_score == 80.0


class TestCreateSpiderPlot:
    """Tests for _create_spider_plot function."""

    def test_basic_plot_creation(self) -> None:
        """Should create a basic spider plot."""
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": 80.0, "sv": 75.0}}
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

    def test_lower_is_better_reverses_axis(self) -> None:
        """Should reverse radial axis when lower_is_better is True."""
        model_scores: dict[str, dict[str, float | None]] = {"model1": {"da": 80.0, "sv": 75.0}}
        fig_normal = _create_spider_plot(
            model_scores, ["da", "sv"], 100.0, lower_is_better=False
        )
        fig_reversed = _create_spider_plot(
            model_scores, ["da", "sv"], 100.0, lower_is_better=True
        )

        normal_range = fig_normal.layout.polar.radialaxis.range
        reversed_range = fig_reversed.layout.polar.radialaxis.range

        assert tuple(normal_range) == (0, 100.0)
        assert tuple(reversed_range) == (100.0, 0)

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
        result = runner.invoke(main, ["--language", "da"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_cli_invalid_language(self) -> None:
        """CLI should fail gracefully for invalid language."""
        runner = CliRunner()
        result = runner.invoke(
            main, ["--model", "test-model", "--language", "invalid_xyz"]
        )
        assert result.exit_code != 0
        assert "Cannot resolve" in result.output


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
                with tempfile.TemporaryDirectory() as outdir:
                    output_path = Path(outdir) / "plot.html"
                    result = runner.invoke(
                        main,
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
                            "--output",
                            str(output_path),
                            "--allow-incomplete",
                        ],
                    )
                    assert result.exit_code == 0, f"CLI failed: {result.output}"
                    assert output_path.exists()

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

                with tempfile.TemporaryDirectory() as outdir:
                    output_path = Path(outdir) / "plot.html"
                    result = runner.invoke(
                        main,
                        [
                            "--model",
                            "test/model1",
                            "--language",
                            "da",
                            "--shots",
                            "auto",
                            "--output",
                            str(output_path),
                            "--allow-incomplete",
                        ],
                    )
                    assert result.exit_code != 0
                    assert "ambiguous" in result.output.lower()

                with tempfile.TemporaryDirectory() as outdir:
                    output_path = Path(outdir) / "plot.html"
                    result = runner.invoke(
                        main,
                        [
                            "--model",
                            "test/model1",
                            "--language",
                            "da",
                            "--shots",
                            "zero",
                            "--output",
                            str(output_path),
                            "--allow-incomplete",
                        ],
                    )
                    assert result.exit_code == 0, f"CLI failed: {result.output}"

    def test_output_path_parent_created(self) -> None:
        """Test that output path parent directories are created."""
        records = [
            make_eee_record("test/model1", ["da"], {"test_macro_f1": 80.0}, False)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model1.jsonl"
            jsonl_path.write_text(json.dumps(records[0]) + "\n")

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                with tempfile.TemporaryDirectory() as outdir:
                    nested_output = Path(outdir) / "subdir" / "nested" / "plot.html"
                    result = runner.invoke(
                        main,
                        [
                            "--model",
                            "test/model1",
                            "--language",
                            "da",
                            "--shots",
                            "zero",
                            "--output",
                            str(nested_output),
                            "--allow-incomplete",
                        ],
                    )
                    assert result.exit_code == 0, f"CLI failed: {result.output}"
                    assert nested_output.exists()

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
                with tempfile.TemporaryDirectory() as outdir:
                    output_path = Path(outdir) / "plot.html"
                    result = runner.invoke(
                        main,
                        [
                            "--model",
                            "test/model1",
                            "--language",
                            "da",
                            "--shots",
                            "zero",
                            "--output",
                            str(output_path),
                            "--allow-incomplete",
                        ],
                    )
                    assert result.exit_code == 0

            content_after = jsonl_path.read_text()
            assert content_after == original_content

    def test_concatenated_jsonl_objects(self) -> None:
        """Test handling of concatenated }{ JSON objects.

        This tests the parse_jsonl_lines ability to handle malformed JSONL
        where multiple JSON objects appear on a single line without newlines.
        """
        record1 = make_eee_record("test/model1", ["da"], {"test_macro_f1": 80.0}, False)
        record2 = make_eee_record("test/model1", ["sv"], {"test_macro_f1": 75.0}, False)

        # Concatenated JSON objects produce }{ pattern naturally
        # (first object ends with }, second starts with {)
        concatenated = json.dumps(record1) + json.dumps(record2)

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "test_model1.jsonl"
            jsonl_path.write_text(concatenated)

            with patch(
                "src.scripts.create_language_spider_plot.RESULTS_DIR", Path(tmpdir)
            ):
                runner = CliRunner()
                with tempfile.TemporaryDirectory() as outdir:
                    output_path = Path(outdir) / "plot.html"
                    result = runner.invoke(
                        main,
                        [
                            "--model",
                            "test/model1",
                            "--language",
                            "da",
                            "--language",
                            "sv",
                            "--shots",
                            "zero",
                            "--output",
                            str(output_path),
                            "--allow-incomplete",
                        ],
                    )
                    assert result.exit_code == 0, f"CLI failed: {result.output}"
