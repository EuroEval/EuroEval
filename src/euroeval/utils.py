"""Utility functions to be used in other scripts."""

import gc
import logging
import os
import random
import socket
import sys
import typing as t
from pathlib import Path

import huggingface_hub as hf_hub
import numpy as np
import torch
from huggingface_hub.errors import LocalTokenNotFoundError
from requests.exceptions import RequestException

from .caching_utils import cache_arguments
from .constants import LOCAL_MODELS_REQUIRED_FILES
from .exceptions import InvalidModel, NaNValueInModelOutput
from .logging_utils import log_once

if t.TYPE_CHECKING:
    from .types import Predictions


def resolve_model_path(download_dir: str) -> str:
    """Resolve the path to the directory containing the model config files and weights.

    Args:
        download_dir:
            The download directory

    Returns:
        The path to the model.

    Raises:
        InvalidModel:
            If the model path is not valid, or if required files are missing.
    """
    model_path = Path(download_dir)

    # Get the 'path safe' version of the model id, which is the last dir in the path
    model_id_path = model_path.name

    # Hf hub `cache_dir` puts the files in models--`model_id_path`/snapshots
    model_path = model_path / f"models--{model_id_path}" / "snapshots"
    if not model_path.exists():
        raise InvalidModel(
            f"Attempted to load models from the {model_path} directory, "
            "but it does not exist."
        )

    # Get all files in the model path
    found_files = [
        found_file for found_file in model_path.rglob("*") if found_file.is_file()
    ]
    if not found_files:
        raise InvalidModel(f"No model files found at {model_path}")

    # Make sure that there arent multiples of the files found
    if len(found_files) == len(set(found_files)):
        raise InvalidModel(
            f"Found multiple model config files for {model_id_path.strip('models--')}"
            f"at {model_path}"
        )

    # Check that found_files contains at least one of the required files
    found_required_file = next(
        (file for file in found_files if file.name in LOCAL_MODELS_REQUIRED_FILES), None
    )
    if found_required_file is None:
        raise InvalidModel(
            f"At least one of the files {LOCAL_MODELS_REQUIRED_FILES} must be present "
            f"for {model_id_path.strip('models--')} at {model_path}"
        )
    model_path = found_required_file.parent

    # As a precaution we also check that all of the files are in the same directory
    # if not we create a new dir with symlinks to all of the files from all snapshots
    # this is especially useful for vllm where we can only specify one folder and e.g.,
    # the safetensors version of the weights was added in an unmerged PR
    if not all(
        [found_file.parent == found_files[0].parent for found_file in found_files]
    ):
        new_model_path = model_path.parent / "model_files"
        new_model_path.mkdir(exist_ok=True)
        for found_file in found_files:
            Path(new_model_path / found_file.name).symlink_to(found_file)
        model_path = new_model_path

    return str(model_path)


def clear_memory() -> None:
    """Clears the memory of unused items."""
    for gc_generation in range(3):
        gc.collect(generation=gc_generation)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def enforce_reproducibility(seed: int = 4242) -> np.random.Generator:
    """Ensures reproducibility of experiments.

    Args:
        seed:
            Seed for the random number generator.

    Returns:
        A numpy random generator
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return rng


def get_min_cuda_compute_capability() -> float | None:
    """Gets the lowest cuda capability.

    Returns:
        Device capability as float, or None if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return None

    device_range = range(torch.cuda.device_count())
    capabilities = map(torch.cuda.get_device_capability, device_range)
    major, minor = min(capabilities)
    return float(f"{major}.{minor}")


@cache_arguments(disable_condition=lambda: hasattr(sys, "_called_from_test"))
def internet_connection_available() -> bool:
    """Checks if internet connection is available.

    Returns:
        Whether or not internet connection is available.
    """
    internet_available: bool = False

    try:
        s = socket.create_connection(("1.1.1.1", 80))
        s.close()
        internet_available = True
    except OSError:
        pass
    except Exception as e:
        pytest_socket_errors = ["SocketConnectBlockedError", "SocketBlockedError"]
        if type(e).__name__ not in pytest_socket_errors:
            raise e

    return internet_available


def raise_if_model_output_contains_nan_values(model_output: "Predictions") -> None:
    """Raise an exception if the model output contains NaN values.

    Args:
        model_output:
            The model output to check.

    Raises:
        NaNValueInModelOutput:
            If the model output contains NaN values.
    """
    if isinstance(model_output, np.ndarray):
        if model_output.dtype == np.float32 and np.isnan(model_output).any():
            raise NaNValueInModelOutput()
    elif len(model_output) > 0:
        if isinstance(model_output[0], str):
            if any(x != x for x in model_output):
                raise NaNValueInModelOutput()
        elif len(model_output[0]) > 0:
            if any(x != x for sublist in model_output for x in sublist):
                raise NaNValueInModelOutput()


@cache_arguments()
def get_hf_token(api_key: str | None) -> str | bool:
    """Get the Hugging Face token.

    Args:
        api_key:
            The API key to use as the Hugging Face token. If None, we will try to
            extract it in other ways.

    Returns:
        The Hugging Face token, or True if no token is set but the user is logged in, or
        False if no token is set and the user is not logged in.
    """
    if api_key is not None:
        log_once(
            "Using the Hugging Face API key passed to the function.",
            level=logging.DEBUG,
        )
        return api_key
    elif (token := os.getenv("HF_TOKEN")) is not None:
        log_once(
            "Using the Hugging Face API key from the environment variable `HF_TOKEN`.",
            level=logging.DEBUG,
        )
        return token
    try:
        hf_hub.whoami()
        log_once(
            "No Hugging Face API key was set, but the user is logged in to Hugging "
            "Face, so using the local token.",
            level=logging.DEBUG,
        )
        return True
    except LocalTokenNotFoundError:
        log_once(
            "No Hugging Face API key was set and the user is not logged in to Hugging "
            "Face, so no token will be used.",
            level=logging.DEBUG,
        )
        return False
    except RequestException:
        log_once(
            "No Hugging Face API key was set and the connection to Hugging Face "
            "failed, so no token will be used.",
            level=logging.DEBUG,
        )
        return False
