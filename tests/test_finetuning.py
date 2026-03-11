"""Tests for the `finetuning` module."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments

from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.enums import DataType
from euroeval.exceptions import InvalidBenchmark, NaNValueInModelOutput
from euroeval.finetuning import (
    finetune,
    get_training_args,
    remove_extra_tensors_from_logits,
)
from euroeval.metrics import HuggingFaceMetric


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
        assert (
            args.per_device_train_batch_size == benchmark_config.finetuning_batch_size
        )

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
        benchmark_config.device = torch.device(device_name)  # type: ignore[read-only]
        args = get_training_args(
            benchmark_config=benchmark_config,
            model_config=model_config,
            iteration_idx=0,
            dtype=DataType.FP32,
            batch_size=None,
        )
        assert args.use_cpu == expected_use_cpu
        benchmark_config.device = old_device  # type: ignore[read-only]


class TestFinetune:
    """Test that the `finetune` function works as expected."""

    @patch("euroeval.finetuning.load_model")
    @patch("euroeval.finetuning.get_training_args")
    @patch("euroeval.finetuning.get_pbar")
    def test_finetune_single_iteration_nan_error(
        self,
        mock_get_pbar: MagicMock,
        mock_get_training_args: MagicMock,
        mock_load_model: MagicMock,
        benchmark_config: BenchmarkConfig,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        metric: HuggingFaceMetric,
    ) -> None:
        """Test NaNValueInModelOutput exception handling."""
        mock_dataset = {
            "train": MagicMock(),
            "val": MagicMock(),
            "test": MagicMock(),
            "original_test": MagicMock(),
        }
        mock_datasets = [mock_dataset]
        mock_get_pbar.return_value = range(1)
        mock_get_training_args.return_value = TrainingArguments(output_dir="test")

        mock_model = MagicMock()
        mock_model.trainer_class.return_value.evaluate.side_effect = (
            NaNValueInModelOutput("NaN detected")
        )
        mock_load_model.return_value = mock_model

        with pytest.raises(
            InvalidBenchmark,
            match="NaN value detected in model outputs, even with mixed precision",
        ):
            finetune(
                model=mock_model,
                datasets=mock_datasets,
                model_config=model_config,
                dataset_config=dataset_config,
                benchmark_config=benchmark_config,
            )

    @patch("euroeval.finetuning.load_model")
    @patch("euroeval.finetuning.get_training_args")
    @patch("euroeval.finetuning.get_pbar")
    @patch("euroeval.finetuning.clear_memory")
    def test_finetune_batch_size_reduction_on_cuda_oom(
        self,
        mock_clear_memory: MagicMock,
        mock_get_pbar: MagicMock,
        mock_get_training_args: MagicMock,
        mock_load_model: MagicMock,
        benchmark_config: BenchmarkConfig,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
    ) -> None:
        """Test batch size reduction when CUDA OOM occurs."""
        mock_dataset = {
            "train": MagicMock(),
            "val": MagicMock(),
            "test": MagicMock(),
            "original_test": MagicMock(),
        }
        mock_datasets = [mock_dataset]
        mock_get_pbar.return_value = range(1)

        mock_training_args = TrainingArguments(output_dir="test")
        mock_get_training_args.side_effect = [mock_training_args, mock_training_args]

        mock_model = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.evaluate.side_effect = [
            RuntimeError("CUDA out of memory"),
            {"test/accuracy": 0.9},
        ]
        mock_model.trainer_class.return_value = mock_trainer
        mock_load_model.return_value = mock_model

        benchmark_config.finetuning_batch_size = 2

        scores = finetune(
            model=None,
            datasets=mock_datasets,
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )

        assert isinstance(scores, list)
        assert len(scores) == 1
        assert mock_clear_memory.call_count > 0

    @patch("euroeval.finetuning.load_model")
    @patch("euroeval.finetuning.get_training_args")
    @patch("euroeval.finetuning.get_pbar")
    @patch("euroeval.finetuning.clear_memory")
    def test_finetune_batch_size_reduction_exhausted(
        self,
        mock_clear_memory: MagicMock,
        mock_get_pbar: MagicMock,
        mock_get_training_args: MagicMock,
        mock_load_model: MagicMock,
        benchmark_config: BenchmarkConfig,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        metric: HuggingFaceMetric,
    ) -> None:
        """Test InvalidBenchmark when batch size reduction fails."""
        mock_dataset = {
            "train": MagicMock(),
            "val": MagicMock(),
            "test": MagicMock(),
            "original_test": MagicMock(),
        }
        mock_datasets = [mock_dataset]
        mock_get_pbar.return_value = range(1)
        mock_get_training_args.return_value = TrainingArguments(output_dir="test")

        mock_model = MagicMock()
        mock_model.trainer_class.return_value.evaluate.side_effect = RuntimeError(
            "CUDA out of memory"
        )
        mock_load_model.return_value = mock_model

        benchmark_config.finetuning_batch_size = 1

        with pytest.raises(
            InvalidBenchmark,
            match="Could not benchmark the model, even with a batch size of 1!",
        ):
            finetune(
                model=mock_model,
                datasets=mock_datasets,
                model_config=model_config,
                dataset_config=dataset_config,
                benchmark_config=benchmark_config,
            )


class TestRemoveExtraTensorsFromLogits:
    """Test that the `remove_extra_tensors_from_logits` function works as expected."""

    def test_remove_extra_tensors_from_logits_tuple(self) -> None:
        """Test removal of extra tensors from tuple logits."""
        logits = (torch.randn(2, 3), (torch.randn(2, 3), torch.randn(2, 3)))
        labels = torch.randint(0, 3, (2,))

        result = remove_extra_tensors_from_logits(logits=logits, labels=labels)

        assert torch.equal(result, logits[0])

    def test_remove_extra_tensors_from_logits_single_tensor(self) -> None:
        """Test that single tensor logits are returned unchanged."""
        logits = torch.randn(2, 3)
        labels = torch.randint(0, 3, (2,))

        result = remove_extra_tensors_from_logits(logits=logits, labels=labels)

        assert torch.equal(result, logits)
