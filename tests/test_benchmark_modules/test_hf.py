"""Unit tests for the `hf` module."""

from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
import torch
from huggingface_hub.hf_api import HfApi

from euroeval.benchmark_modules.hf import get_model_repo_info, get_torch_dtype
from euroeval.data_models import BenchmarkConfig


@pytest.mark.parametrize(
    argnames=["test_device", "torch_dtype_is_set", "bf16_available", "expected"],
    argvalues=[
        ("cpu", True, True, torch.float32),
        ("cpu", True, False, torch.float32),
        ("cpu", False, True, torch.float32),
        ("cpu", False, False, torch.float32),
        ("mps", True, True, torch.float32),
        ("mps", True, False, torch.float32),
        ("mps", False, True, torch.float32),
        ("mps", False, False, torch.float32),
        ("cuda", True, True, "auto"),
        ("cuda", True, False, "auto"),
        ("cuda", False, True, torch.bfloat16),
        ("cuda", False, False, torch.float16),
    ],
)
def test_get_torch_dtype(
    test_device: str,
    torch_dtype_is_set: bool,
    bf16_available: bool,
    expected: torch.dtype,
) -> None:
    """Test that the torch dtype is set correctly."""
    assert (
        get_torch_dtype(
            device=torch.device(test_device),
            torch_dtype_is_set=torch_dtype_is_set,
            bf16_available=bf16_available,
        )
        == expected
    )


@pytest.mark.parametrize(
    argnames=["repo_files", "only_allow_safetensors", "model_exists"],
    argvalues=[
        (["model.safetensors", "config.json"], True, True),
        (["pytorch_model.bin", "config.json"], True, False),
        (["pytorch_model.bin", "config.json"], False, True),
        ([], True, False),
    ],
    ids=[
        "Model with safetensors",
        "Model without safetensors",
        "Safetensors check disabled",
        "Empty repo files",
    ],
)
def test_safetensors_check(
    repo_files: list[str],
    only_allow_safetensors: bool,
    model_exists: bool,
    benchmark_config: BenchmarkConfig,
) -> None:
    """Test the safetensors availability check functionality."""
    cloned_benchmark_config = deepcopy(benchmark_config)
    cloned_benchmark_config.only_allow_safetensors = only_allow_safetensors
    with (
        patch.object(HfApi, "list_repo_files") as mock_list_files,
        patch.object(HfApi, "model_info") as mock_model_info,
    ):
        mock_list_files.return_value = repo_files
        mock_model_info.return_value = MagicMock(
            id="test-model", tags=["test"], pipeline_tag="fill-mask"
        )
        result = get_model_repo_info(
            model_id="test-model",
            revision="main",
            benchmark_config=cloned_benchmark_config,
        )
        assert (result is not None) == model_exists
