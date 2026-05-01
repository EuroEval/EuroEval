"""Unit tests for the `hf` module."""

import dataclasses
import hashlib
from unittest.mock import MagicMock, patch

import pytest
import torch
from huggingface_hub.hf_api import HfApi

from euroeval.benchmark_modules.hf import (
    HuggingFaceEncoderModel,
    get_dtype,
    get_model_repo_info,
)
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.enums import EvaluationType
from euroeval.exceptions import InvalidBenchmark


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


class TestCFGating:
    """Tests that the HF encoder backend rejects Cloze Formulation evaluation.

    CF requires per-token logprobs of forced continuations, which the encoder
    backend does not produce. The guard sits in `__init__` so the user gets a
    clear error before any model loading happens.
    """

    def test_init_raises_invalid_benchmark_on_cf(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Constructing the HF encoder with `evaluation_type=CF` raises early."""
        cf_config = dataclasses.replace(
            benchmark_config, evaluation_type=EvaluationType.CF
        )
        # Patch out the param-allow-list check so the CF guard is what fires.
        with patch("euroeval.benchmark_modules.hf.raise_if_wrong_params"):
            with pytest.raises(InvalidBenchmark, match="Cloze Formulation"):
                HuggingFaceEncoderModel(
                    model_config=model_config,
                    dataset_config=dataset_config,
                    benchmark_config=cf_config,
                    log_metadata=False,
                )

    def test_error_message_points_to_vllm_backend(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """The CF rejection message names vLLM as the supported alternative."""
        cf_config = dataclasses.replace(
            benchmark_config, evaluation_type=EvaluationType.CF
        )
        with patch("euroeval.benchmark_modules.hf.raise_if_wrong_params"):
            with pytest.raises(InvalidBenchmark, match="vLLM backend"):
                HuggingFaceEncoderModel(
                    model_config=model_config,
                    dataset_config=dataset_config,
                    benchmark_config=cf_config,
                    log_metadata=False,
                )
