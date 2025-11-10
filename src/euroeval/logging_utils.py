"""Utility functions related to logging."""

import datetime as dt
import logging
import os
import sys
import warnings
from io import TextIOWrapper

import litellm
from datasets.utils import disable_progress_bars as disable_datasets_progress_bars
from evaluate import disable_progress_bar as disable_evaluate_progress_bar
from huggingface_hub.utils.tqdm import (
    disable_progress_bars as disable_hf_hub_progress_bars,
)
from termcolor import colored
from tqdm.auto import tqdm
from transformers import logging as tf_logging

from .caching_utils import cache_arguments

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


def log(message: str, level: int, colour: str | None = None) -> None:
    """Log a message.

    Args:
        message:
            The message to log.
        level:
            The logging level. Defaults to logging.INFO.
        colour:
            The colour to use for the message. If None, a default colour will be used
            based on the logging level.

    Raises:
        ValueError:
            If the logging level is invalid.
    """
    match level:
        case logging.DEBUG:
            message = colored(
                text=(
                    "[DEBUG] "
                    + dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    + f" · {message}"
                ),
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


@cache_arguments("message")
def log_once(message: str, level: int = logging.INFO, prefix: str = "") -> None:
    """Log a message once.

    This is ensured by caching the "message" argument and only logging it the first time
    this function is called with that message.

    Args:
        message:
            The message to log.
        level:
            The logging level. Defaults to logging.INFO.
        prefix:
            A prefix to add to the message, which is not considered when determining if
            the message has been logged before.
    """
    log(message=prefix + message, level=level)


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

    # Disable flashinfer logging
    os.environ["FLASHINFER_LOGGING_LEVEL"] = "CRITICAL"

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
        self.devnull_file: TextIOWrapper | None = None
        self._original_stdout_fd: int | None = None
        self._original_stderr_fd: int | None = None

    def _log_windows_warning(self) -> None:
        """Log a warning about Windows not supporting blocking terminal output."""
        log_once(
            "Your operating system (probably Windows) does not support blocking "
            "terminal output, so expect more messy output - sorry!",
            level=logging.WARNING,
        )

    def __enter__(self) -> None:
        """Suppress all terminal output."""
        if self.disable:
            return

        try:
            # Save original FDs by duplicating them
            self._original_stdout_fd = os.dup(sys.stdout.fileno())
            self._original_stderr_fd = os.dup(sys.stderr.fileno())

            # Open /dev/null
            self.devnull_file = open(os.devnull, "w")

            # Redirect stdout/stderr to /dev/null
            os.dup2(self.devnull_file.fileno(), sys.stdout.fileno())
            os.dup2(self.devnull_file.fileno(), sys.stderr.fileno())

        except OSError:
            self._log_windows_warning()
            # If setup fails, clean up any resources we might have acquired
            self.__exit__(None, None, None)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: type[BaseException] | None,
    ) -> None:
        """Re-enable terminal output."""
        if self.disable:
            return

        # Restore stdout/stderr from our saved FDs
        try:
            if self._original_stdout_fd is not None:
                os.dup2(self._original_stdout_fd, sys.stdout.fileno())
            if self._original_stderr_fd is not None:
                os.dup2(self._original_stderr_fd, sys.stderr.fileno())
        except OSError:
            self._log_windows_warning()
        finally:
            # Close the duplicated FDs we created
            if self._original_stdout_fd is not None:
                os.close(self._original_stdout_fd)
            if self._original_stderr_fd is not None:
                os.close(self._original_stderr_fd)

            # Close the /dev/null file
            if self.devnull_file is not None:
                self.devnull_file.close()


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
