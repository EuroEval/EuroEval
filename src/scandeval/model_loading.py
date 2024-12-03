"""Functions related to the loading of models."""

import typing as t

from scandeval.benchmark_modules.fresh import FreshEncoderModel
from scandeval.benchmark_modules.vllm import VLLMModel

from .benchmark_modules import HuggingFaceEncoderModel, LiteLLMModel
from .constants import GENERATIVE_DATASET_SUPERTASKS, GENERATIVE_DATASET_TASKS
from .enums import ModelType
from .exceptions import InvalidBenchmark

if t.TYPE_CHECKING:
    from .benchmark_modules import BenchmarkModule
    from .data_models import BenchmarkConfig, DatasetConfig, ModelConfig


def load_model(
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
) -> "BenchmarkModule":
    """Load a model.

    Args:
        model_config:
            The model configuration.
        dataset_config:
            The dataset configuration.
        benchmark_config:
            The benchmark configuration.

    Returns:
        The model.
    """
    model_type_to_module_mapping: dict[ModelType, t.Type[BenchmarkModule]] = {
        ModelType.FRESH: FreshEncoderModel,
        ModelType.HF_HUB_ENCODER: HuggingFaceEncoderModel,
        ModelType.HF_HUB_GENERATIVE: VLLMModel,
        ModelType.API: LiteLLMModel,
        ModelType.LOCAL: VLLMModel,  # TODO: Add LocalModel
    }
    model_class = model_type_to_module_mapping[model_config.model_type]

    model = model_class(
        model_config=model_config,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
    )

    # Refuse to benchmark non-generative models on generative tasks
    if (
        (
            dataset_config.task.supertask in GENERATIVE_DATASET_SUPERTASKS
            or dataset_config.task.name in GENERATIVE_DATASET_TASKS
        )
        and model is not None
        and not model.is_generative
    ):
        raise InvalidBenchmark(
            f"Cannot benchmark non-generative model {model_config.model_id!r} on "
            f"generative task {dataset_config.task.name!r}."
        )

    return model
