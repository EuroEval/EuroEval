"""Tests for the `cli` module."""

from click.types import ParamType


def test_cli_param_names(cli_params: dict[str, ParamType]) -> None:
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
        "requires_safetensors",
        "generative_type",
        "custom_datasets_file",
        "download_only",
        "debug",
        "model_language",
        "dataset_language",
        "batch_size",
        "help",
    }
