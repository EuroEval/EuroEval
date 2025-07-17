"""Unit tests for the `cli` module."""

from typing import Generator

import pytest
from click.types import BOOL, FLOAT, INT, STRING, Choice, ParamType

from euroeval.cli import benchmark


@pytest.fixture(scope="module")
def params() -> Generator[dict[str | None, ParamType], None, None]:
    """Yields a dictionary of the CLI parameters."""
    ctx = benchmark.make_context(info_name="testing", args=["--model", "test-model"])
    yield {p.name: p.type for p in benchmark.get_params(ctx)}


def test_cli_param_names(params: dict[str, ParamType]) -> None:
    """Test that the CLI parameters have the correct names."""
    assert set(params.keys()) == {
        "model",
        "task",
        "language",
        "model_language",
        "dataset_language",
        "dataset",
        "batch_size",
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
        "only_allow_safetensors",
    }


def test_cli_param_types(params: dict[str, ParamType]) -> None:
    """Test that the CLI parameters have the correct types."""
    assert params["model"] == STRING
    assert isinstance(params["dataset"], Choice)
    assert isinstance(params["language"], Choice)
    assert isinstance(params["model_language"], Choice)
    assert isinstance(params["dataset_language"], Choice)
    assert isinstance(params["task"], Choice)
    assert isinstance(params["batch_size"], Choice)
    assert params["progress_bar"] == BOOL
    assert params["raise_errors"] == BOOL
    assert params["verbose"] == BOOL
    assert params["save_results"] == BOOL
    assert params["cache_dir"] == STRING
    assert params["api_key"] == STRING
    assert params["force"] == BOOL
    assert isinstance(params["device"], Choice)
    assert params["trust_remote_code"] == BOOL
    assert params["clear_model_cache"] == BOOL
    assert params["evaluate_test_split"] == BOOL
    assert params["few_shot"] == BOOL
    assert params["num_iterations"] == INT
    assert params["api_base"] == STRING
    assert params["api_version"] == STRING
    assert params["gpu_memory_utilization"] == FLOAT
    assert params["debug"] == BOOL
    assert params["help"] == BOOL
    assert params["only_allow_safetensors"] == BOOL
