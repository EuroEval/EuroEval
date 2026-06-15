"""Functions related to the loading of models."""

import logging
import typing as t

from .benchmark_modules import (
    FreshEncoderModel,
    HuggingFaceEncoderModel,
    LiteLLMModel,
    VLLMModel,
)
from .enums import GenerativeType, InferenceBackend, ModelType
from .exceptions import InvalidModel
from .logging_utils import log_once


def validate_bits_per_character(
    model_config: "ModelConfig", benchmark_config: "BenchmarkConfig"
) -> None:
    """Validate that BPC scoring is supported for the given model.

    BPC is only supported with vLLM backend and base decoder models.

    Args:
        model_config:
            The model configuration.
        benchmark_config:
            The benchmark configuration.

    Raises:
        InvalidModel:
            If BPC scoring is requested but not supported.
    """
    if not benchmark_config.use_bits_per_character:
        return

    # Only vLLM + base decoders support BPC
    if model_config.inference_backend != InferenceBackend.VLLM:
        raise InvalidModel(
            f"Bits-per-character (BPC) scoring requires the vLLM backend, but "
            f"{model_config.inference_backend.value} was specified."
        )

    if model_config.generative_type != GenerativeType.BASE:
        raise InvalidModel(
            f"Bits-per-character (BPC) scoring requires a base decoder model, but "
            f"{model_config.generative_type.value} was specified."
        )


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

    Raises:
        InvalidModel:
            If the model is not a supported model type.
    """
    log_once(f"\nLoading the model {model_config.model_id}...", level=logging.INFO)

    # The order matters; the first model type that matches will be used. For this
    # reason, they have been ordered in terms of the most common model types.
    model_class: t.Type["BenchmarkModule"]
    match (model_config.model_type, model_config.inference_backend, model_config.fresh):
        case (ModelType.GENERATIVE, InferenceBackend.VLLM, False):
            model_class = VLLMModel
        case (ModelType.ENCODER, InferenceBackend.TRANSFORMERS, False):
            model_class = HuggingFaceEncoderModel
        case (ModelType.GENERATIVE, InferenceBackend.LITELLM, False):
            model_class = LiteLLMModel
        case (ModelType.ENCODER, InferenceBackend.TRANSFORMERS, True):
            model_class = FreshEncoderModel
        case (_, _, True):
            raise InvalidModel(
                "Cannot load a freshly initialised model with the model type "
                f"{model_config.model_type!r} and inference backend "
                f"{model_config.inference_backend!r}."
            )
        case _:
            raise InvalidModel(
                f"Cannot load model with model type {model_config.model_type!r} and "
                f"inference backend {model_config.inference_backend!r}."
            )

    # Validate BPC support before loading
    validate_bits_per_character(
        model_config=model_config, benchmark_config=benchmark_config
    )

    model = model_class(
        model_config=model_config,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
    )

    return model
