"""Tests for the `cli` module."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from click.types import ParamType

from euroeval.cli import benchmark


def test_cli_param_names(cli_params: dict[str | None, ParamType]) -> None:
    """Test that the CLI parameters have the correct names."""
    assert set(cli_params.keys()) == {
        "model",
        "task",
        "language",
        "dataset",
        "finetuning_batch_size",
        "progress_bar",
        "raise_errors",
        "verbose",
        "save_results",
        "cache_dir",
        "api_key",
        "force",
        "device",
        "trust_remote_code",
        "clear_model_cache",
        "evaluate_test_split",
        "few_shot",
        "num_iterations",
        "api_base",
        "api_version",
        "gpu_memory_utilization",
        "attention_backend",
        "requires_safetensors",
        "generative_type",
        "custom_datasets_file",
        "use_bits_per_character",
        "download_only",
        "debug",
        "max_context_length",
        "vocabulary_size",
        "num_parameters",
        "help",
    }


def test_contamination_detection_is_selected_as_a_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The canary is selected through the ordinary task option."""
    mock_benchmarker_cls = MagicMock()
    monkeypatch.setattr("euroeval.cli.Benchmarker", mock_benchmarker_cls)

    result = CliRunner().invoke(
        benchmark, ["--model", "dummy", "--task", "contamination-detection"]
    )

    assert result.exit_code == 0
    assert mock_benchmarker_cls.call_args.kwargs["task"] == ["contamination-detection"]


@pytest.mark.parametrize(
    argnames=["options", "conflicting_options"],
    argvalues=[
        (
            ["--dataset", "dansk", "--task", "classification"],
            ["`--task`", "`--dataset`"],
        ),
        (["--dataset", "dansk", "--language", "da"], ["`--language`", "`--dataset`"]),
    ],
)
def test_dataset_and_task_conflict(
    options: list[str], conflicting_options: list[str]
) -> None:
    """Test that `--dataset` cannot be combined with `--task` or `--language`."""
    result = CliRunner().invoke(benchmark, ["--model", "dummy"] + options)
    assert result.exit_code == 2
    assert all(option in result.output for option in conflicting_options)
    assert "Traceback" not in result.output


def test_dataset_selects_the_languages_it_contains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that `--dataset` alone still reaches the benchmarker with every language.

    A dataset ID selects the configurations and splits of a dataset, so which languages
    are benchmarked follows from the dataset itself; the default `--language all` is not
    a request and so is not a conflict.
    """
    mock_benchmarker_cls = MagicMock()
    monkeypatch.setattr("euroeval.cli.Benchmarker", mock_benchmarker_cls)
    result = CliRunner().invoke(
        benchmark, ["--model", "dummy", "--dataset", "dansk", "--language", "all"]
    )
    assert result.exit_code == 0
    assert mock_benchmarker_cls.call_args.kwargs["dataset"] == ["dansk"]
    assert mock_benchmarker_cls.call_args.kwargs["language"] == ["all"]


def test_missing_gliner_extra_fails_without_cached_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing GLiNER extra is a CLI error, not an empty cached run."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["SpanExtractor"],
                "model_type": "extractor",
                "architecture": "span",
                "config_version": 3,
                "architecture_version": 1,
                "span_head": {"span_mode": "marker"},
            }
        )
    )
    monkeypatch.setitem(sys.modules, "gliner2", None)

    result = CliRunner().invoke(
        benchmark, ["-m", str(tmp_path), "-l", "da", "--no-save-results"]
    )

    assert result.exit_code != 0
    assert "pip install euroeval[gliner]" in result.output
    assert "No benchmarks to run" not in result.output


def test_num_parameters_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that `--num-parameters` is forwarded to the benchmarker."""
    mock_benchmarker_cls = MagicMock()
    monkeypatch.setattr("euroeval.cli.Benchmarker", mock_benchmarker_cls)

    result = CliRunner().invoke(
        benchmark, ["--model", "dummy", "--num-parameters", "123456"]
    )

    assert result.exit_code == 0
    assert mock_benchmarker_cls.call_args.kwargs["num_parameters"] == 123_456
