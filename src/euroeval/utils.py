"""Utility functions to be used in other scripts."""

import gc
import importlib
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
from types import TracebackType

import litellm
import numpy as np
import pkg_resources
import requests
import torch
from datasets.utils import disable_progress_bar
from requests.exceptions import RequestException
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase
from transformers import logging as tf_logging

from .exceptions import InvalidModel, NaNValueInModelOutput

if importlib.util.find_spec("ray") is not None:
    import ray

if t.TYPE_CHECKING:
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


def is_module_installed(module: str) -> bool:
    """Check if a module is installed.

    This is used when dealing with spaCy models, as these are installed as separate
    Python packages.

    Args:
        module:
            The name of the module.

    Returns:
        Whether the module is installed or not.
    """
    # Get list of all modules, including their versions
    installed_modules_with_versions = list(pkg_resources.working_set)

    # Strip the module versions from the list of modules. Also make the modules lower
    # case and replace dashes with underscores
    installed_modules = [
        re.sub("[0-9. ]", "", str(module)).lower().replace("-", "_")
        for module in installed_modules_with_versions
    ]

    # Check if the module is installed by checking if the module name is in the list
    return module.lower() in installed_modules


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
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("ray._private.worker").setLevel(logging.CRITICAL)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.CRITICAL)
    logging.getLogger("accelerate").setLevel(logging.CRITICAL)
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
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


def kebab_to_pascal(kebab_string: str) -> str:
    """Converts a kebab-case string to PascalCase.

    Args:
        kebab_string:
            The kebab-case string.

    Returns:
        The PascalCase string.
    """
    return "".join(word.title() for word in kebab_string.split("-"))


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


def get_special_token_metadata(tokenizer: "PreTrainedTokenizerBase") -> dict:
    """Get the special token metadata for a tokenizer.

    Args:
        tokenizer:
            The tokenizer.

    Returns:
        The special token metadata.
    """
    # Create some test input IDs, to check if the tokenizer is adding special tokens
    test_input_ids = tokenizer("Test").input_ids

    # Extract the CLS token IDs from the tokenizer, if it's using them
    has_cls_token = True
    if tokenizer.cls_token_id in test_input_ids:
        cls_token_id = tokenizer.cls_token_id
        cls_token = tokenizer.cls_token
    elif tokenizer.bos_token_id in test_input_ids:
        cls_token_id = tokenizer.bos_token_id
        cls_token = tokenizer.bos_token
    elif tokenizer.cls_token is not None:
        cls_token_id = tokenizer.cls_token_id
        cls_token = tokenizer.cls_token
        has_cls_token = False
    else:
        cls_token_id = tokenizer.bos_token_id
        cls_token = tokenizer.bos_token
        has_cls_token = False

    # Extract the SEP token IDs from the tokenizer, if it's using them
    has_sep_token = True
    if tokenizer.sep_token_id in test_input_ids:
        sep_token = tokenizer.sep_token
    elif tokenizer.eos_token_id in test_input_ids:
        sep_token = tokenizer.eos_token
    elif tokenizer.sep_token is not None:
        sep_token = tokenizer.sep_token
        has_sep_token = False
    else:
        sep_token = tokenizer.eos_token
        has_sep_token = False

    return dict(
        cls_token_id=cls_token_id,
        cls_token=cls_token,
        sep_token=sep_token,
        has_cls_token=has_cls_token,
        has_sep_token=has_sep_token,
    )


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
        exc_tb: TracebackType,
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


def should_prompts_be_stripped(
    labels_to_be_generated: list[str], tokenizer: "PreTrainedTokenizer"
) -> bool:
    """Determine if we should strip the prompts for few-shot evaluation.

    This is the case if the tokenizer needs to include the space as part of the label
    token. The strategy is thus to tokenize a label with a preceeding colon (as in the
    prompts), i.e., ": positive", and check if the tokenization starts with the tokens
    of ": ". If this is the case, then we should not strip the prompts, since the
    tokenizer produces the whitespace token separately.

    Args:
        labels_to_be_generated:
            The labels that are to be generated.
        tokenizer:
            The tokenizer used to tokenize the labels.

    Returns:
        Whether we should strip the prompts.
    """
    strip_prompts = True
    for label in labels_to_be_generated:
        colon_tokens = tokenizer(": ", add_special_tokens=False).input_ids
        label_tokens = tokenizer(": " + label, add_special_tokens=False).input_ids

        if isinstance(colon_tokens, torch.Tensor):
            colon_tokens = list(colon_tokens.squeeze(0))
        if isinstance(label_tokens, torch.Tensor):
            label_tokens = list(label_tokens.squeeze(0))

        label_tokens_start_with_colon_tokens = (
            label_tokens[: len(colon_tokens)] == colon_tokens
        )
        if label_tokens_start_with_colon_tokens:
            strip_prompts = False

    return strip_prompts


# TODO: This is currently not used - maybe remove.
def should_prefix_space_be_added_to_labels(
    labels_to_be_generated: list[str], tokenizer: "PreTrainedTokenizer"
) -> bool:
    """Determine if we should add a prefix space to the labels.

    This is the case if the prompts are stripped and the tokenizer doesn't
    automatically add prefix whitespaces to the labels.

    Args:
        labels_to_be_generated:
            The labels that are to be generated.
        tokenizer:
            The tokenizer used to tokenize the labels.

    Returns:
        Whether we should add a prefix space to the labels.
    """
    if not should_prompts_be_stripped(
        labels_to_be_generated=labels_to_be_generated, tokenizer=tokenizer
    ):
        return False

    whitespace_token = tokenizer.convert_ids_to_tokens(
        ids=tokenizer(" ", add_special_tokens=False).input_ids[0]
    )[0]

    add_prefix_space = True
    for label in labels_to_be_generated:
        label_tokens = tokenizer(label, add_special_tokens=False).input_ids
        if isinstance(label_tokens, torch.Tensor):
            label_tokens = list(label_tokens.squeeze(0))
        first_label_token: int = int(label_tokens[0])
        first_character_of_label = tokenizer.convert_ids_to_tokens(first_label_token)[0]
        has_prefix_space = first_character_of_label == whitespace_token
        if has_prefix_space:
            add_prefix_space = False
            break

    return add_prefix_space


def get_bos_token(tokenizer: "PreTrainedTokenizer") -> tuple[str, int]:
    """Get the beginning-of-sequence token from a tokenizer.

    Args:
        tokenizer:
            The tokenizer.

    Returns:
        A pair (token, token_id) representing the beginning-of-sequence token and its
        token ID.
    """
    if isinstance(tokenizer.bos_token, str) and isinstance(tokenizer.bos_token_id, int):
        return tokenizer.bos_token, tokenizer.bos_token_id

    vocab: dict[str, int] = tokenizer.get_vocab()

    candidate_bos_tokens = ["<s>", "<|begin_of_text|>", "[CLS]"]
    for candidate_bos_token in candidate_bos_tokens:
        if candidate_bos_token in vocab:
            bos_token = candidate_bos_token
            bos_token_id = vocab[bos_token]
            break
    else:
        raise InvalidModel(
            "The model does not have a beginning-of-sequence token. Please ensure that "
            "this has been set in the tokenizer's configuration."
        )

    return bos_token, bos_token_id


def get_eos_token(tokenizer: "PreTrainedTokenizer") -> tuple[str, int]:
    """Get the end-of-sequence token from a tokenizer.

    Args:
        tokenizer:
            The tokenizer.

    Returns:
        A pair (token, token_id) representing the end-of-sequence token and its token
        ID.
    """
    if isinstance(tokenizer.eos_token, str) and isinstance(tokenizer.eos_token_id, int):
        return tokenizer.eos_token, tokenizer.eos_token_id

    vocab: dict[str, int] = tokenizer.get_vocab()

    candidate_eos_tokens = ["</s>", "<|end_of_text|>", "[SEP]"]
    for candidate_eos_token in candidate_eos_tokens:
        if candidate_eos_token in vocab:
            eos_token = candidate_eos_token
            eos_token_id = vocab[eos_token]
            break
    else:
        raise InvalidModel(
            "The model does not have an end-of-sequence token. Please ensure that this "
            "has been set in the tokenizer's configuration."
        )

    return eos_token, eos_token_id


def get_end_of_chat_token_ids(tokenizer: "PreTrainedTokenizer") -> list[int] | None:
    """Get the end token ID for chat models.

    This is only relevant for tokenizers with a chat template.

    Args:
        tokenizer:
            The tokenizer.

    Returns:
        The token IDs used to end chats, or None if the tokenizer does not have a chat
        template.

    Raises:
        ValueError:
            If the end-of-chat token could not be located.
    """
    if tokenizer.chat_template is None:
        return None

    user_message: dict[t.Literal["role", "content"], str] = dict()
    user_message["role"] = "user"
    user_message["content"] = "X"
    token_ids = tokenizer.apply_chat_template(conversation=[user_message])
    assert isinstance(token_ids, list)

    for idx, token in enumerate(tokenizer.convert_ids_to_tokens(token_ids)):
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert isinstance(token_id, int)
        token = tokenizer.decode([token_id])
        if "X" in token:
            x_token_index = idx
            break
    else:
        raise ValueError("Could not locate the end-of-chat token for the model.")

    end_of_chat_tokens = token_ids[x_token_index + 1 :]
    if len(end_of_chat_tokens) == 0:
        return None
    return end_of_chat_tokens


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
