"""Functions related to the loading of models."""

import logging
import typing as t

from .benchmark_modules import (
    FreshEncoderModel,
    HuggingFaceEncoderModel,
    LiteLLMModel,
    VLLMModel,
)
from .enums import DataType, GenerativeType, InferenceBackend, ModelType
from .exceptions import InvalidModel
from .logging_utils import log_once


def validate_bits_per_character_backend(
    model_config: "ModelConfig", benchmark_config: "BenchmarkConfig"
) -> None:
    """Validate that the model's backend supports BPC scoring.

    BPC is only supported with the vLLM backend. This check uses only the model
    configuration, so it can run before the (potentially expensive) model load.

    Args:
        model_config:
            The model configuration.
        benchmark_config:
            The benchmark configuration.

    Raises:
        InvalidModel:
            If BPC scoring is requested with an unsupported backend.
    """
    if not benchmark_config.use_bits_per_character:
        return

    if model_config.inference_backend != InferenceBackend.VLLM:
        raise InvalidModel(
            f"Bits-per-character (BPC) scoring requires the vLLM backend, but "
            f"{model_config.inference_backend.value} was specified."
        )


def validate_bits_per_character_generative_type(
    model: "BenchmarkModule", benchmark_config: "BenchmarkConfig"
) -> None:
    """Validate that the model's generative type supports BPC scoring.

    BPC is only supported for base decoder models. The generative type is derived
    from the model's tokeniser, so this check can only run after the model has been
    loaded.

    Args:
        model:
            The loaded model.
        benchmark_config:
            The benchmark configuration.

    Raises:
        InvalidModel:
            If BPC scoring is requested for a non-base model.
    """
    if not benchmark_config.use_bits_per_character:
        return

    if model.generative_type != GenerativeType.BASE:
        detected = (
            model.generative_type.value
            if model.generative_type is not None
            else "unknown"
        )
        raise InvalidModel(
            f"Bits-per-character (BPC) scoring requires a base decoder model, but "
            f"the model {model.model_config.model_id!r} was detected as "
            f"{detected!r}."
        )


if t.TYPE_CHECKING:
    from .benchmark_modules import BenchmarkModule
    from .data_models import BenchmarkConfig, DatasetConfig, ModelConfig


def load_model(
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
    dtype_override: "DataType | None" = None,
) -> "BenchmarkModule":
    """Load a model.

    Args:
        model_config:
            The model configuration.
        dataset_config:
            The dataset configuration.
        benchmark_config:
            The benchmark configuration.
        dtype_override:
            An explicit data type to load the model weights in, taking precedence
            over the hardware-derived default. Only applies to encoder models loaded
            for finetuning; used by the NaN-retry to force a full fp32 reload.

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

    # Validate BPC backend support before the (potentially expensive) load
    validate_bits_per_character_backend(
        model_config=model_config, benchmark_config=benchmark_config
    )

    # The dtype override is only plumbed through the plain encoder loading path (the
    # only one that derives its dtype from the hardware via `get_dtype`), so restrict
    # it to exactly that class -- subclasses such as `VLLMModel` and `FreshEncoderModel`
    # neither finetune from a pretrained checkpoint nor accept this argument.
    if dtype_override is not None and model_class is HuggingFaceEncoderModel:
        model = HuggingFaceEncoderModel(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            dtype_override=dtype_override,
        )
    else:
        model = model_class(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )

    # The generative type is only known once the model's tokeniser is loaded, so this
    # check must run after instantiation.
    validate_bits_per_character_generative_type(
        model=model, benchmark_config=benchmark_config
    )

    return model
