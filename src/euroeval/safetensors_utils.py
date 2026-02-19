"""Utility functions related to parsing of safetensors metadata of models."""

import logging
from pathlib import Path

from huggingface_hub import get_safetensors_metadata
from huggingface_hub.errors import NotASafetensorsRepoError

from .logging_utils import log_once
from .utils import get_hf_token, internet_connection_available


def get_num_params_from_safetensors_metadata(
    model_id: str, revision: str, api_key: str | None
) -> int | None:
    """Get the number of parameters from the safetensors metadata.

    Args:
        model_id:
            The model ID.
        revision:
            The revision of the model.
        api_key:
            The API key to use for authentication with the Hugging Face Hub. Can be
            None if no API key is needed.

    Returns:
        The number of parameters, or None if the metadata could not be found.
    """
    # We cannot determine the number of parameters if there is no internet connection
    # or if the model is stored locally
    if not internet_connection_available() or Path(model_id).exists():
        return None

    try:
        metadata = get_safetensors_metadata(
            repo_id=model_id, revision=revision, token=get_hf_token(api_key=api_key)
        )
    except NotASafetensorsRepoError:
        log_once(
            "The number of parameters could not be determined for the model "
            f"{model_id}, since the model is not stored in the safetensors format. "
            "If this is your own model, then you can use this Hugging Face Space to "
            "convert your model to the safetensors format: "
            "https://huggingface.co/spaces/safetensors/convert.",
            level=logging.WARNING,
        )
        return None

    parameter_count_dict = metadata.parameter_count
    match len(parameter_count_dict):
        case 0:
            log_once(
                "Failed to determine the number of parameters for the model "
                f"{model_id}, even though the model is stored in the safetensors "
                "format. Please report this issue at "
                "https://github.com/EuroEval/EuroEval/issues.",
                level=logging.WARNING,
            )
            return None
        case 1:
            return max(parameter_count_dict.values())
        case _:
            log_once(
                f"The model {model_id} has multiple parameter count entries in its "
                f"safetensors metadata: {parameter_count_dict}. Using the largest one.",
                level=logging.DEBUG,
            )
            return max(parameter_count_dict.values())
