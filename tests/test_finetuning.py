"""Tests for the `finetuning` module."""

import pytest
import torch
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments

from euroeval.data_models import BenchmarkConfig, ModelConfig
from euroeval.enums import DataType
from euroeval.finetuning import get_training_args


# TODO
def test_finetune() -> None:
    """Test that the `finetune` function works as expected."""
    pass


# TODO
def test_finetune_single_iteration() -> None:
    """Test that the `finetune_single_iteration` function works as expected."""
    pass


class TestGetTrainingArgs:
    """Test that the `get_training_args` function works as expected."""

    def test_return_type(
        self, benchmark_config: BenchmarkConfig, model_config: ModelConfig
    ) -> None:
        """Test that the return type is correct."""
        args = get_training_args(
            benchmark_config=benchmark_config,
            model_config=model_config,
            iteration_idx=0,
            dtype=DataType.FP32,
            batch_size=None,
        )
        assert isinstance(args, TrainingArguments)

    @pytest.mark.parametrize(argnames=["batch_size"], argvalues=[(8,), (16,), (32,)])
    def test_both_batch_sizes_are_set(
        self,
        benchmark_config: BenchmarkConfig,
        model_config: ModelConfig,
        batch_size: int,
    ) -> None:
        """Test that both training and eval batch sizes are set."""
        args = get_training_args(
            benchmark_config=benchmark_config,
            model_config=model_config,
            iteration_idx=0,
            dtype=DataType.FP32,
            batch_size=batch_size,
        )
        assert args.per_device_train_batch_size == batch_size
        assert args.per_device_eval_batch_size == batch_size

    @pytest.mark.parametrize(
        argnames=["batch_size", "expected_gradient_accumulation_steps"],
        argvalues=[(1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1)],
    )
    def test_gradient_accumulation(
        self,
        benchmark_config: BenchmarkConfig,
        model_config: ModelConfig,
        batch_size: int,
        expected_gradient_accumulation_steps: int,
    ) -> None:
        """Test that the gradient accumulation is correct."""
        args = get_training_args(
            benchmark_config=benchmark_config,
            model_config=model_config,
            iteration_idx=0,
            dtype=DataType.FP32,
            batch_size=batch_size,
        )
        assert args.gradient_accumulation_steps == expected_gradient_accumulation_steps

    def test_batch_size_default_value(
        self, benchmark_config: BenchmarkConfig, model_config: ModelConfig
    ) -> None:
        """Test that the default value for the batch size is correct."""
        args = get_training_args(
            benchmark_config=benchmark_config,
            model_config=model_config,
            iteration_idx=0,
            dtype=DataType.FP32,
            batch_size=None,
        )
        assert args.per_device_train_batch_size == benchmark_config.batch_size

    @pytest.mark.parametrize(
        argnames=["verbose", "expected_logging_strategy"],
        argvalues=[(True, IntervalStrategy.STEPS), (False, IntervalStrategy.NO)],
    )
    def test_logging_strategy(
        self,
        benchmark_config: BenchmarkConfig,
        model_config: ModelConfig,
        verbose: bool,
        expected_logging_strategy: IntervalStrategy,
    ) -> None:
        """Test that the logging strategy is correct."""
        old_verbose = benchmark_config.verbose
        benchmark_config.verbose = verbose
        args = get_training_args(
            benchmark_config=benchmark_config,
            model_config=model_config,
            iteration_idx=0,
            dtype=DataType.FP32,
            batch_size=None,
        )
        assert args.logging_strategy == expected_logging_strategy
        benchmark_config.verbose = old_verbose

    @pytest.mark.parametrize(
        argnames=["progress_bar", "expected_disable_tqdm"],
        argvalues=[(True, False), (False, True)],
    )
    def test_disable_tqdm(
        self,
        benchmark_config: BenchmarkConfig,
        model_config: ModelConfig,
        progress_bar: bool,
        expected_disable_tqdm: bool,
    ) -> None:
        """Test that the disable tqdm option is correct."""
        old_progress_bar = benchmark_config.progress_bar
        benchmark_config.progress_bar = progress_bar
        args = get_training_args(
            benchmark_config=benchmark_config,
            model_config=model_config,
            iteration_idx=0,
            dtype=DataType.FP32,
            batch_size=None,
        )
        assert args.disable_tqdm == expected_disable_tqdm
        benchmark_config.progress_bar = old_progress_bar

    @pytest.mark.parametrize(
        argnames=["datatype", "expected_fp16", "expected_bf16"],
        argvalues=[
            (DataType.FP32, False, False),
            (DataType.FP16, True, False),
            (DataType.BF16, False, True),
        ],
    )
    def test_dtype(
        self,
        benchmark_config: BenchmarkConfig,
        model_config: ModelConfig,
        datatype: DataType,
        expected_fp16: bool,
        expected_bf16: bool,
    ) -> None:
        """Test that the fp16 and bf16 arguments have been correctly set."""
        args = get_training_args(
            benchmark_config=benchmark_config,
            model_config=model_config,
            iteration_idx=0,
            dtype=datatype,
            batch_size=None,
        )
        assert args.fp16 == expected_fp16
        assert args.bf16 == expected_bf16

    @pytest.mark.parametrize(
        argnames=["device_name", "expected_use_cpu"],
        argvalues=[("cuda", False), ("mps", False), ("cpu", True)],
    )
    def test_use_cpu(
        self,
        benchmark_config: BenchmarkConfig,
        model_config: ModelConfig,
        device_name: str,
        expected_use_cpu: bool,
    ) -> None:
        """Test that the use_cpu argument is correct."""
        old_device = benchmark_config.device
        benchmark_config.device = torch.device(device_name)
        args = get_training_args(
            benchmark_config=benchmark_config,
            model_config=model_config,
            iteration_idx=0,
            dtype=DataType.FP32,
            batch_size=None,
        )
        assert args.use_cpu == expected_use_cpu
        benchmark_config.device = old_device
