"""Unit tests for the `hf` module."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from huggingface_hub.hf_api import HfApi

from euroeval.benchmark_modules.hf import get_model_repo_info, get_torch_dtype
from euroeval.data_models import BenchmarkConfig
from euroeval.exceptions import InvalidModel


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


def test_safetensors_check(benchmark_config: BenchmarkConfig) -> None:
    """Test the safetensors availability check functionality."""
    benchmark_config.only_allow_safetensors = True

    # Mock HfApi and its list_files method
    with (
        patch.object(HfApi, "list_repo_files") as mock_list_files,
        patch.object(HfApi, "model_info") as mock_model_info,
    ):
        # Test case 1: Model with safetensors
        mock_list_files.return_value = ["model.safetensors", "config.json"]
        mock_model_info.return_value = MagicMock(
            id="test-model", tags=["test"], pipeline_tag="fill-mask"
        )

        # Should not raise an exception
        result = get_model_repo_info("test-model", "main", benchmark_config)
        assert result is not None

        # Test case 2: Model without safetensors
        mock_list_files.return_value = ["pytorch_model.bin", "config.json"]

        # Should raise InvalidModel
        with pytest.raises(InvalidModel) as exc_info:
            get_model_repo_info("test-model", "main", benchmark_config)
        assert "does not have safetensors weights available" in str(exc_info.value)

        # Test case 3: Safetensors check disabled
        benchmark_config.only_allow_safetensors = False
        mock_list_files.return_value = ["pytorch_model.bin", "config.json"]

        # Should not raise an exception when check is disabled
        result = get_model_repo_info("test-model", "main", benchmark_config)
        assert result is not None
