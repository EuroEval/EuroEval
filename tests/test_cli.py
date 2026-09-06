"""Tests for the `cli` module."""

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
        "help",
    }


@pytest.mark.parametrize(argnames=["language"], argvalues=[("all",), ("da",)])
def test_dataset_and_language_does_not_conflict(
    monkeypatch: pytest.MonkeyPatch, language: str
) -> None:
    """Test that `--language` can be combined with `--dataset`, narrowing its choice."""
    mock_benchmarker_cls = MagicMock()
    monkeypatch.setattr("euroeval.cli.Benchmarker", mock_benchmarker_cls)
    result = CliRunner().invoke(
        benchmark, ["--model", "dummy", "--dataset", "dansk", "--language", language]
    )
    assert result.exit_code == 0
    assert mock_benchmarker_cls.call_args.kwargs["dataset"] == ["dansk"]
    assert mock_benchmarker_cls.call_args.kwargs["language"] == [language]


@pytest.mark.parametrize(
    argnames=["options", "conflicting_options"],
    argvalues=[
        (
            ["--dataset", "dansk", "--task", "classification"],
            ["`--task`", "`--dataset`"],
        )
    ],
)
def test_dataset_and_task_conflict(
    options: list[str], conflicting_options: list[str]
) -> None:
    """Test that `--dataset` cannot be combined with `--task`."""
    result = CliRunner().invoke(benchmark, ["--model", "dummy"] + options)
    assert result.exit_code == 2
    assert all(option in result.output for option in conflicting_options)
    assert "Traceback" not in result.output
