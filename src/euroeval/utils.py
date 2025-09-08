"""Utility functions to be used in other scripts."""

import asyncio
import gc
import importlib
import importlib.metadata
import importlib.util
import logging
import os
import random
import re
import sys
import typing as t
import warnings
from functools import cache
from pathlib import Path

import demjson3
import huggingface_hub as hf_hub
import litellm
import numpy as np
import requests
import torch
from datasets.utils import disable_progress_bar
from requests.exceptions import RequestException
from transformers import logging as tf_logging

from .exceptions import InvalidBenchmark, NaNValueInModelOutput

if t.TYPE_CHECKING:
    from types import TracebackType

    from .types import Predictions


logger = logging.getLogger("euroeval")


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


def block_terminal_output() -> None:
    """Blocks libraries from writing output to the terminal.

    This filters warnings from some libraries, sets the logging level to ERROR for some
    libraries, disabled tokeniser progress bars when using Hugging Face tokenisers, and
    disables most of the logging from the `transformers` library.
    """
    # Ignore miscellaneous warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.getLogger("absl").setLevel(logging.CRITICAL)

    # Disable matplotlib logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.CRITICAL)

    # Disable PyTorch logging
    logging.getLogger("torch.utils.cpp_extension").setLevel(logging.CRITICAL)
    warnings.filterwarnings(action="ignore", module="torch*")
    os.environ["TORCH_LOGS"] = "-all"

    # Disable huggingface_hub logging
    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)

    # Disable LiteLLM logging
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
    logging.getLogger("LiteLLM Router").setLevel(logging.CRITICAL)
    logging.getLogger("LiteLLM Proxy").setLevel(logging.CRITICAL)
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    litellm.suppress_debug_info = True

    # Disable vLLM logging
    logging.getLogger("vllm").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.engine.llm_engine").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.transformers_utils.tokenizer").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.core.scheduler").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.model_executor.weight_utils").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.platforms").setLevel(logging.CRITICAL)
    logging.getLogger("mistral_common.tokens.tokenizers.tekken").setLevel(
        logging.CRITICAL
    )
    os.environ["LOG_LEVEL"] = "CRITICAL"
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

    # Disable datasets logging
    logging.getLogger("datasets").setLevel(logging.CRITICAL)
    logging.getLogger("filelock").setLevel(logging.CRITICAL)
    disable_progress_bar()

    # Disable evaluate logging
    warnings.filterwarnings("ignore", module="seqeval*")

    # Disable most of the `transformers` logging
    tf_logging._default_log_level = logging.CRITICAL
    tf_logging.set_verbosity(logging.CRITICAL)
    logging.getLogger("transformers.trainer").setLevel(logging.CRITICAL)
    logging.getLogger("accelerate").setLevel(logging.CRITICAL)


def get_class_by_name(class_name: str | list[str], module_name: str) -> t.Type | None:
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
        logger.debug(
            f"Could not find the class with the name(s) {', '.join(class_name)}. The "
            f"following error messages were raised: {errors}"
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


def internet_connection_available() -> bool:
    """Checks if internet connection is available by pinging google.com.

    Returns:
        Whether or not internet connection is available.
    """
    try:
        requests.get("https://www.google.com")
        return True
    except RequestException:
        return False


class HiddenPrints:
    """Context manager which removes all terminal output."""

    def __enter__(self) -> None:
        """Enter the context manager."""
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(
        self,
        exc_type: t.Type[BaseException],
        exc_val: BaseException,
        exc_tb: "TracebackType",
    ) -> None:
        """Exit the context manager."""
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


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


@cache
def log_once(message: str, level: int = logging.INFO) -> None:
    """Log a message once.

    This is ensured by caching the input/output pairs of this function, using the
    `functools.cache` decorator.

    Args:
        message:
            The message to log.
        level:
            The logging level. Defaults to logging.INFO.
    """
    match level:
        case logging.DEBUG:
            logger.debug(message)
        case logging.INFO:
            logger.info(message)
        case logging.WARNING:
            logger.warning(message)
        case logging.ERROR:
            logger.error(message)
        case logging.CRITICAL:
            logger.critical(message)
        case _:
            raise ValueError(f"Invalid logging level: {level}")


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


T = t.TypeVar("T", bound=object)


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
        return loop.run_until_complete(coroutine)
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
    json_regex = r"\{[^{}]+?\}"
    if (json_match := re.search(pattern=json_regex, string=s, flags=re.DOTALL)) is None:
        logger.debug(
            "The model output does not contain any JSON dictionary, so cannot parse "
            f"it. Skipping. Here is the output: {s!r}"
        )
        return None
    json_string = json_match.group()
    try:
        json_output = demjson3.decode(txt=json_string)
    except demjson3.JSONDecodeError:
        logger.debug(
            "The model output is not valid JSON, so cannot parse it. Skipping. "
            f"Here is the output: {json_string!r}"
        )
        return None
    if not isinstance(json_output, dict):
        logger.debug(
            "The model output is not a JSON dictionary, so cannot parse "
            f"it. Skipping. Here is the output: {json_string!r}"
        )
        return None
    elif not all(isinstance(key, str) for key in json_output.keys()):
        logger.debug(
            "The model output is not a JSON dictionary with string keys, "
            "so cannot parse it. Skipping. Here is the output: "
            f"{json_string!r}"
        )
        return None
    return json_output


@cache
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
    prompt: str, candidate_labels: list[str]
) -> list[str]:
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
