"""Unit tests for the `vllm` module."""

import importlib.util
import shutil
from unittest.mock import MagicMock, patch

import pytest
import torch

from euroeval.benchmark_modules.vllm import VLLMModel
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.exceptions import NeedsSystemDependency


class TestNvccCheck:
    """Tests for the nvcc system-dependency check in VLLMModel.__init__."""

    @pytest.mark.parametrize(
        argnames=["cuda_available", "hip_version", "nvcc_path", "needs_nvcc"],
        argvalues=[
            (True, None, None, True),
            (True, "5.7.0", None, False),
            (False, None, None, False),
            (True, None, "/usr/bin/nvcc", False),
        ],
        ids=[
            "NVIDIA CUDA without nvcc needs nvcc",
            "ROCm without nvcc does not need nvcc",
            "No CUDA without nvcc does not need nvcc",
            "NVIDIA CUDA with nvcc does not need nvcc",
        ],
    )
    def test_nvcc_check_condition(
        self,
        cuda_available: bool,
        hip_version: str | None,
        nvcc_path: str | None,
        needs_nvcc: bool,
    ) -> None:
        """Test the condition used in the nvcc system-dependency check.

        The check should require nvcc only when CUDA is available on a non-ROCm
        (i.e. NVIDIA) build and nvcc is missing from PATH.
        """
        with (
            patch.object(torch.cuda, "is_available", return_value=cuda_available),
            patch.object(torch.version, "hip", new=hip_version),
            patch.object(shutil, "which", return_value=nvcc_path),
        ):
            condition = (
                torch.cuda.is_available()
                and torch.version.hip is None
                and shutil.which("nvcc") is None
            )

        assert condition == needs_nvcc

    def test_nvcc_check_raises_on_nvidia_without_nvcc(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Test that NeedsSystemDependency is raised on NVIDIA CUDA without nvcc."""
        original_find_spec = importlib.util.find_spec

        def mock_find_spec(name: str) -> object:
            if name == "vllm":
                return MagicMock()
            return original_find_spec(name)

        with (
            patch("importlib.util.find_spec", side_effect=mock_find_spec),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.version, "hip", new=None),
            patch.object(shutil, "which", return_value=None),
        ):
            with pytest.raises(NeedsSystemDependency):
                VLLMModel(
                    model_config=model_config,
                    dataset_config=dataset_config,
                    benchmark_config=benchmark_config,
                    log_metadata=False,
                )

    def test_nvcc_check_skipped_on_rocm(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Test that NeedsSystemDependency is not raised on ROCm without nvcc."""
        original_find_spec = importlib.util.find_spec

        def mock_find_spec(name: str) -> object:
            if name == "vllm":
                return MagicMock()
            return original_find_spec(name)

        with (
            patch("importlib.util.find_spec", side_effect=mock_find_spec),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.version, "hip", new="5.7.0"),
            patch.object(shutil, "which", return_value=None),
        ):
            try:
                VLLMModel(
                    model_config=model_config,
                    dataset_config=dataset_config,
                    benchmark_config=benchmark_config,
                    log_metadata=False,
                )
            except NeedsSystemDependency:
                pytest.fail(
                    "NeedsSystemDependency was raised unexpectedly on ROCm hardware "
                    "without nvcc"
                )
            except Exception:
                pass  # Other exceptions are acceptable in this test
