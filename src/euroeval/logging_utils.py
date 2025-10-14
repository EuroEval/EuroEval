"""Utility functions related to logging."""

import datetime as dt
import logging
import os
import sys
import warnings
from functools import cache

import litellm
from datasets.utils import disable_progress_bars as disable_datasets_progress_bars
from evaluate import disable_progress_bar as disable_evaluate_progress_bar
from huggingface_hub.utils.tqdm import (
    disable_progress_bars as disable_hf_hub_progress_bars,
)
from termcolor import colored
from tqdm.auto import tqdm
from transformers import logging as tf_logging

logger = logging.getLogger("euroeval")


def get_pbar(*tqdm_args, **tqdm_kwargs) -> tqdm:
    """Get a progress bar for vLLM with custom hard-coded arguments.

    Args:
        *tqdm_args:
            Positional arguments to pass to tqdm.
        **tqdm_kwargs:
            Additional keyword arguments to pass to tqdm.

    Returns:
        A tqdm progress bar.
    """
    tqdm_kwargs = dict(colour="yellow", ascii="—▰", leave=False) | tqdm_kwargs
    tqdm_kwargs["desc"] = colored(
        text=tqdm_kwargs.get("desc", "Processing"), color="light_yellow"
    )
    return tqdm(*tqdm_args, **tqdm_kwargs)


def log(message: str, level: int = logging.INFO, colour: str | None = None) -> None:
    """Log a message.

    Args:
        message:
            The message to log.
        level:
            The logging level. Defaults to logging.INFO.

    Raises:
        ValueError:
            If the logging level is invalid.
    """
    match level:
        case logging.DEBUG:
            message = colored(
                text=f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}",
                color=colour or "light_blue",
            )
            logger.debug(message)
        case logging.INFO:
            if colour is not None:
                message = colored(text=message, color=colour)
            logger.info(message)
        case logging.WARNING:
            message = colored(text=message, color=colour or "light_red")
            logger.warning(message)
        case logging.ERROR:
            message = colored(text=message, color=colour or "red")
            logger.error(message)
        case logging.CRITICAL:
            message = colored(text=message, color=colour or "red")
            logger.critical(message)
        case _:
            raise ValueError(f"Invalid logging level: {level}")


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
    log(message=message, level=level)


def block_terminal_output() -> None:
    """Blocks libraries from writing output to the terminal.

    This filters warnings from some libraries, sets the logging level to ERROR for some
    libraries, disabled tokeniser progress bars when using Hugging Face tokenisers, and
    disables most of the logging from the `transformers` library.
    """
    if os.getenv("FULL_LOG") == "1":
        return

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
    disable_hf_hub_progress_bars()

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
    disable_datasets_progress_bars()

    # Disable evaluate logging
    warnings.filterwarnings("ignore", module="seqeval*")
    disable_evaluate_progress_bar()

    # Disable most of the `transformers` logging
    tf_logging._default_log_level = logging.CRITICAL
    tf_logging.set_verbosity(logging.CRITICAL)
    logging.getLogger("transformers.trainer").setLevel(logging.CRITICAL)
    logging.getLogger("accelerate").setLevel(logging.CRITICAL)


class no_terminal_output:
    """Context manager that suppresses all terminal output."""

    def __init__(self, disable: bool = False) -> None:
        """Initialise the context manager.

        Args:
            disable:
                If True, this context manager does nothing.
        """
        self.disable = disable
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

    def __enter__(self) -> None:
        """Suppress all terminal output."""
        if not self.disable:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: type[BaseException] | None,
    ) -> None:
        """Re-enable terminal output."""
        if not self.disable:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr


def adjust_logging_level(verbose: bool, ignore_testing: bool = False) -> int:
    """Adjust the logging level based on verbosity.

    Args:
        verbose:
            Whether to output additional output.
        ignore_testing:
            Whether to ignore the testing flag.

    Returns:
        The logging level that was set.
    """
    if hasattr(sys, "_called_from_test") and not ignore_testing:
        logging_level = logging.CRITICAL
    elif verbose:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    logger.setLevel(logging_level)
    return logging_level
