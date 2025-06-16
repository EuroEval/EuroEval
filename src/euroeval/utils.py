"""Utility functions to be used in other scripts."""

import asyncio
import gc
import importlib
import importlib.metadata
import importlib.util
import logging
import os
import random
import sys
import typing as t
import warnings
from functools import cache
from pathlib import Path

import litellm
import numpy as np
import requests
import torch
from datasets.utils import disable_progress_bar
from requests.exceptions import RequestException
from transformers import logging as tf_logging

from .exceptions import NaNValueInModelOutput

if importlib.util.find_spec("ray") is not None:
    import ray

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
    warnings.filterwarnings(
        "ignore",
        module="torch.nn.parallel*",
        message="Was asked to gather along dimension 0, but all input tensors were "
        "scalars; will instead unsqueeze and return a vector.",
    )
    warnings.filterwarnings("ignore", module="seqeval*")

    # Up the logging level, to disable outputs
    logging.getLogger("filelock").setLevel(logging.CRITICAL)
    logging.getLogger("absl").setLevel(logging.CRITICAL)
    logging.getLogger("datasets").setLevel(logging.CRITICAL)
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.CRITICAL)
    logging.getLogger("torch.nn.parallel.distributed").setLevel(logging.CRITICAL)
    logging.getLogger("vllm").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.engine.llm_engine").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.transformers_utils.tokenizer").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.core.scheduler").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.model_executor.weight_utils").setLevel(logging.CRITICAL)
    logging.getLogger("vllm.platforms").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("ray._private.worker").setLevel(logging.CRITICAL)
    logging.getLogger("ray._private.services").setLevel(logging.CRITICAL)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.CRITICAL)
    logging.getLogger("accelerate").setLevel(logging.CRITICAL)
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
    logging.getLogger("LiteLLM Router").setLevel(logging.CRITICAL)
    logging.getLogger("LiteLLM Proxy").setLevel(logging.CRITICAL)
    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)

    # This suppresses vLLM logging
    os.environ["LOG_LEVEL"] = "CRITICAL"
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

    if importlib.util.find_spec("ray") is not None:
        ray._private.worker._worker_logs_enabled = False

    # Disable the tokeniser progress bars
    disable_progress_bar()

    # Disable most of the `transformers` logging
    tf_logging._default_log_level = logging.CRITICAL
    tf_logging.set_verbosity(logging.CRITICAL)
    logging.getLogger("transformers.trainer").setLevel(logging.CRITICAL)

    # Disable logging from `litellm`
    litellm.suppress_debug_info = True


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
