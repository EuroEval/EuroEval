"""Tests for the `cli` module."""

from click.types import BOOL, FLOAT, INT, STRING, Choice, ParamType


def test_cli_param_names(cli_params: dict[str, ParamType]) -> None:
    """Test that the CLI parameters have the correct names."""
    assert set(cli_params.keys()) == {
        "model",
        "task",
        "language",
        "finetuning_batch_size",
        "dataset",
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
        "debug",
        "help",
        "requires_safetensors",
        "generative_type",
        "download_only",
        "model_language",
        "dataset_language",
        "batch_size",
    }


def test_cli_param_types(cli_params: dict[str, ParamType]) -> None:
    """Test that the CLI parameters have the correct types."""
    assert cli_params["model"] == STRING
    assert cli_params["dataset"] == STRING
    assert isinstance(cli_params["language"], Choice)
    assert cli_params["task"] == STRING
    assert isinstance(cli_params["finetuning_batch_size"], Choice)
    assert cli_params["progress_bar"] == BOOL
    assert cli_params["raise_errors"] == BOOL
    assert cli_params["verbose"] == BOOL
    assert cli_params["save_results"] == BOOL
    assert cli_params["cache_dir"] == STRING
    assert cli_params["api_key"] == STRING
    assert cli_params["force"] == BOOL
    assert isinstance(cli_params["device"], Choice)
    assert cli_params["trust_remote_code"] == BOOL
    assert cli_params["clear_model_cache"] == BOOL
    assert cli_params["evaluate_test_split"] == BOOL
    assert cli_params["few_shot"] == BOOL
    assert cli_params["num_iterations"] == INT
    assert cli_params["api_base"] == STRING
    assert cli_params["api_version"] == STRING
    assert cli_params["gpu_memory_utilization"] == FLOAT
    assert cli_params["debug"] == BOOL
    assert cli_params["help"] == BOOL
    assert cli_params["requires_safetensors"] == BOOL
    assert isinstance(cli_params["generative_type"], Choice)
    assert cli_params["download_only"] == BOOL
