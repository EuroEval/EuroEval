"""Unit tests for the `hf` module."""

import hashlib
from unittest.mock import MagicMock, patch

import pytest
import torch
from huggingface_hub.hf_api import HfApi

from euroeval.benchmark_modules.hf import get_dtype, get_model_repo_info
from euroeval.data_models import BenchmarkConfig


@pytest.mark.parametrize(
    argnames=["test_device", "dtype_is_set", "bf16_available", "expected"],
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
def test_get_dtype(
    test_device: str, dtype_is_set: bool, bf16_available: bool, expected: torch.dtype
) -> None:
    """Test that the dtype is set correctly."""
    assert (
        get_dtype(
            device=torch.device(test_device),
            dtype_is_set=dtype_is_set,
            bf16_available=bf16_available,
        )
        == expected
    )


@pytest.mark.parametrize(
    argnames=["repo_files", "requires_safetensors", "model_exists"],
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
    requires_safetensors: bool,
    model_exists: bool,
    benchmark_config: BenchmarkConfig,
) -> None:
    """Test the safetensors availability check functionality."""
    with (
        patch.object(HfApi, "list_repo_files") as mock_list_files,
        patch.object(HfApi, "model_info") as mock_model_info,
    ):
        mock_list_files.return_value = repo_files
        mock_model_info.return_value = MagicMock(
            id="test-model", tags=["test"], pipeline_tag="fill-mask"
        )
        hash_model_id = hashlib.md5(
            ",".join(repo_files).encode("utf-8")
            + str(requires_safetensors).encode("utf-8")
        ).hexdigest()
        result = get_model_repo_info(
            model_id=f"model-{hash_model_id}",
            revision="main",
            api_key=benchmark_config.api_key,
            cache_dir=benchmark_config.cache_dir,
            trust_remote_code=benchmark_config.trust_remote_code,
            requires_safetensors=requires_safetensors,
            run_with_cli=benchmark_config.run_with_cli,
        )
        assert (result is not None) == model_exists
