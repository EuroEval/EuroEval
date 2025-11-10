"""Utility functions to be used in other scripts."""

import asyncio
import collections.abc as c
import gc
import importlib
import importlib.metadata
import importlib.util
import logging
import os
import random
import re
import socket
import sys
import typing as t
from pathlib import Path
from types import ModuleType

import demjson3
import huggingface_hub as hf_hub
import numpy as np
import torch

from .caching_utils import cache_arguments
from .constants import T
from .exceptions import InvalidBenchmark, InvalidModel, NaNValueInModelOutput
from .logging_utils import log, log_once

if t.TYPE_CHECKING:
    from .data_models import ModelIdComponents
    from .types import Predictions


def create_model_cache_dir(cache_dir: str, model_id: str) -> str:
    """Create cache directory for a model.

    Args:
        cache_dir:
            The cache directory.
        model_id:
            The model ID.

    Returns:
        The path to the cache directory.
    """
    # to avoid nesting due to models name containing '/'
    _model_id = model_id.replace("/", "--")
    cache_dir_path = Path(cache_dir) / "model_cache" / _model_id
    return str(cache_dir_path)


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

    # Check that found_files contains at least a 'config.json'
    config_file = next(
        (file for file in found_files if file.name == "config.json"), None
    )
    if config_file is None:
        raise InvalidModel(
            f"Missing required file 'config.json' for {model_id_path.strip('models--')}"
            f"at {model_path}"
        )
    model_path = config_file.parent

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


def get_class_by_name(
    class_name: str | c.Sequence[str], module_name: str
) -> t.Type | None:
    """Get a class by its name.

    Args:
        class_name:
            The name of the class, written in kebab-case. The corresponding class name
            must be the same, but written in PascalCase, and lying in a module with the
            same name, but written in snake_case. If a list of strings is passed, the
            first class that is found is returned.
        module_name:
            The name of the module where the class is located.

    Returns:
        The class. If the class is not found, None is returned.
    """
    if isinstance(class_name, str):
        class_name = [class_name]

    error_messages = list()
    for name in class_name:
        try:
            module = importlib.import_module(name=module_name)
            class_: t.Type = getattr(module, name)
            return class_
        except (ModuleNotFoundError, AttributeError) as e:
            error_messages.append(str(e))

    if error_messages:
        errors = "\n- " + "\n- ".join(error_messages)
        log(
            f"Could not find the class with the name(s) {', '.join(class_name)}. The "
            f"following error messages were raised: {errors}",
            level=logging.DEBUG,
        )

    # If the class could not be found, return None
    return None


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
    """Checks if internet connection is available by pinging google.com.

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


def scramble(text: str) -> str:
    """Scramble a string in a bijective manner.

    Args:
        text:
            The string to scramble.

    Returns:
        The scrambled string.
    """
    rng = np.random.default_rng(seed=4242)
    permutation = rng.permutation(x=len(text))
    scrambled = "".join(text[i] for i in permutation)
    return scrambled


def unscramble(scrambled_text: str) -> str:
    """Unscramble a string in a bijective manner.

    Args:
        scrambled_text:
            The scrambled string to unscramble.

    Returns:
        The unscrambled string.
    """
    rng = np.random.default_rng(seed=4242)
    permutation = rng.permutation(x=len(scrambled_text))
    inverse_permutation = np.argsort(permutation)
    unscrambled = "".join(scrambled_text[i] for i in inverse_permutation)
    return unscrambled


def get_package_version(package_name: str) -> str | None:
    """Get the version of a package.

    Args:
        package_name:
            The name of the package.

    Returns:
        The version of the package, or None if the package is not installed.
    """
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def safe_run(coroutine: t.Coroutine[t.Any, t.Any, T]) -> T:
    """Run a coroutine, ensuring that the event loop is always closed when we're done.

    Args:
        coroutine:
            The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(coroutine)
        return response
    finally:
        loop.close()
        asyncio.set_event_loop(None)


async def add_semaphore_and_catch_exception(
    coroutine: t.Coroutine[t.Any, t.Any, T], semaphore: asyncio.Semaphore
) -> T | Exception:
    """Run a coroutine with a semaphore.

    Args:
        coroutine:
            The coroutine to run.
        semaphore:
            The semaphore to use.

    Returns:
        The result of the coroutine.
    """
    async with semaphore:
        try:
            return await coroutine
        except Exception as exc:
            return exc


def extract_json_dict_from_string(s: str) -> dict | None:
    """Extract a JSON dictionary from a string.

    Args:
        s:
            The string to extract the JSON dictionary from.

    Returns:
        The extracted JSON dictionary, or None if no JSON dictionary could be found.
    """
    json_regex = r"\{[^{}]*?\}"
    if (json_match := re.search(pattern=json_regex, string=s, flags=re.DOTALL)) is None:
        log(
            "The model output does not contain any JSON dictionary, so cannot parse "
            f"it. Skipping. Here is the output: {s!r}",
            level=logging.DEBUG,
        )
        return None
    json_string = json_match.group()
    try:
        json_output = demjson3.decode(txt=json_string)
    except demjson3.JSONDecodeError:
        log(
            "The model output is not valid JSON, so cannot parse it. Skipping. "
            f"Here is the output: {json_string!r}",
            level=logging.DEBUG,
        )
        return None
    if not isinstance(json_output, dict):
        log(
            "The model output is not a JSON dictionary, so cannot parse "
            f"it. Skipping. Here is the output: {json_string!r}",
            level=logging.DEBUG,
        )
        return None
    elif not all(isinstance(key, str) for key in json_output.keys()):
        log(
            "The model output is not a JSON dictionary with string keys, "
            "so cannot parse it. Skipping. Here is the output: "
            f"{json_string!r}",
            level=logging.DEBUG,
        )
        return None
    return json_output


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
    elif (token := os.getenv("HUGGINGFACE_API_KEY")) is not None:
        log_once(
            "Using the Hugging Face API key from the environment variable "
            "`HUGGINGFACE_API_KEY`.",
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
    except hf_hub.errors.LocalTokenNotFoundError:
        log_once(
            "No Hugging Face API key was set and the user is not logged in to Hugging "
            "Face, so no token will be used.",
            level=logging.DEBUG,
        )
        return False


def extract_multiple_choice_labels(
    prompt: str, candidate_labels: c.Sequence[str]
) -> c.Sequence[str]:
    """Extract multiple choice labels from a prompt.

    Args:
        prompt:
            The prompt to extract the labels from.
        candidate_labels:
            The candidate labels to look for in the prompt.

    Returns:
        The extracted labels.
    """
    sample_candidate_labels: list[str] = list()
    for candidate_label in candidate_labels:
        candidate_label_match = re.search(
            pattern=rf"\b{candidate_label}\. ", string=prompt, flags=re.IGNORECASE
        )
        if candidate_label_match is not None:
            sample_candidate_labels.append(candidate_label)
    if not sample_candidate_labels:
        raise InvalidBenchmark(
            "Could not extract any candidate labels from the prompt. Please ensure "
            "that the candidate labels are present in the prompt, each followed by a "
            "dot and a space (e.g., 'a. '). The candidate labels are: "
            f"{', '.join(candidate_labels)}. Here is the prompt: {prompt!r}"
        )
    return sample_candidate_labels


def split_model_id(model_id: str) -> "ModelIdComponents":
    """Split a model ID into its components.

    Args:
        model_id:
            The model ID to split.

    Returns:
        The split model ID.

    Raises:
        If the model ID is not valid.
    """
    # Importing here to avoid circular imports
    from .data_models import ModelIdComponents

    # Attempt to extract the model ID, revision, and param using regex
    model_id_match = re.match(pattern=r"^[^@#]+", string=model_id)
    revision_match = re.search(pattern=r"@([^@#]+)", string=model_id)
    param_match = re.search(pattern=r"#([^@#]+)", string=model_id)

    # If we cannot extract the model ID, raise an error
    if model_id_match is None:
        raise InvalidModel(f"The model ID {model_id!r} is not valid.")
    model_id = model_id_match.group()

    # Extract the revision and param and return the result
    revision = revision_match.group(1) if revision_match is not None else "main"
    param = param_match.group(1) if param_match is not None else None
    return ModelIdComponents(model_id=model_id, revision=revision, param=param)


def load_custom_datasets_module() -> ModuleType | None:
    """Load the custom datasets module if it exists.

    Raises:
        RuntimeError:
            If the custom datasets module cannot be loaded.
    """
    custom_datasets_file = Path("custom_datasets.py")
    if custom_datasets_file.exists():
        spec = importlib.util.spec_from_file_location(
            name="custom_datasets_module", location=str(custom_datasets_file.resolve())
        )
        if spec is None:
            log_once(
                "Could not load the spec for the custom datasets file from "
                f"{custom_datasets_file.resolve()}.",
                level=logging.ERROR,
            )
            return None
        module = importlib.util.module_from_spec(spec=spec)
        if spec.loader is None:
            log_once(
                "Could not load the module for the custom datasets file from "
                f"{custom_datasets_file.resolve()}.",
                level=logging.ERROR,
            )
            return None
        spec.loader.exec_module(module)
        return module
    return None


class flash_attention_backend:
    """Context manager to temporarily set the flash attention backend.

    This sets the `VLLM_ATTENTION_BACKEND` environment variable to `FLASH_ATTN`
    for the duration of the context manager, and restores the previous value afterwards.
    """

    def __init__(self, disabled: bool = False) -> None:
        """Initialise the context manager.

        Args:
            disabled:
                If True, this context manager does nothing.
        """
        self.disabled = disabled
        self.previous_value: str | None = None

    def __enter__(self) -> None:
        """Enter the context manager."""
        if self.disabled:
            return
        self.previous_value = os.getenv("VLLM_ATTENTION_BACKEND")
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

    def __exit__(
        self,
        exc_type: t.Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: type[BaseException] | None,
    ) -> None:
        """Exit the context manager.

        Args:
            exc_type:
                The type of the exception.
            exc_value:
                The value of the exception.
            exc_tb:
                The traceback of the exception.
        """
        if self.disabled:
            return
        if self.previous_value is None:
            os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        else:
            os.environ["VLLM_ATTENTION_BACKEND"] = self.previous_value
