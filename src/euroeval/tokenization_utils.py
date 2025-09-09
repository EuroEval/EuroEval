"""Utility functions related to tokenization."""

import logging
import re
import typing as t

import torch
from transformers import MistralCommonTokenizer

from .enums import GenerativeType
from .exceptions import InvalidModel
from .utils import log_once

if t.TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from .data_models import DatasetConfig, ModelConfig


logger = logging.getLogger("euroeval")


def get_special_token_metadata(tokeniser: "PreTrainedTokenizerBase") -> dict:
    """Get the special token metadata for a tokeniser.

    Args:
        tokeniser:
            The tokeniser.

    Returns:
        The special token metadata.
    """
    # Create some test input IDs, to check if the tokeniser is adding special tokens
    test_input_ids = tokeniser("Test").input_ids

    # Extract the CLS token IDs from the tokeniser, if it's using them
    has_cls_token = True
    if tokeniser.cls_token_id in test_input_ids:
        cls_token_id = tokeniser.cls_token_id
        cls_token = tokeniser.cls_token
    elif tokeniser.bos_token_id in test_input_ids:
        cls_token_id = tokeniser.bos_token_id
        cls_token = tokeniser.bos_token
    elif tokeniser.cls_token is not None:
        cls_token_id = tokeniser.cls_token_id
        cls_token = tokeniser.cls_token
        has_cls_token = False
    else:
        cls_token_id = tokeniser.bos_token_id
        cls_token = tokeniser.bos_token
        has_cls_token = False

    # Extract the SEP token IDs from the tokeniser, if it's using them
    has_sep_token = True
    if tokeniser.sep_token_id in test_input_ids:
        sep_token = tokeniser.sep_token
    elif tokeniser.eos_token_id in test_input_ids:
        sep_token = tokeniser.eos_token
    elif tokeniser.sep_token is not None:
        sep_token = tokeniser.sep_token
        has_sep_token = False
    else:
        sep_token = tokeniser.eos_token
        has_sep_token = False

    return dict(
        cls_token_id=cls_token_id,
        cls_token=cls_token,
        sep_token=sep_token,
        has_cls_token=has_cls_token,
        has_sep_token=has_sep_token,
    )


def should_prompts_be_stripped(
    labels_to_be_generated: list[str], tokeniser: "PreTrainedTokenizer"
) -> bool:
    """Determine if we should strip the prompts for few-shot evaluation.

    This is the case if the tokeniser needs to include the space as part of the label
    token. The strategy is thus to tokenize a label with a preceeding colon (as in the
    prompts), i.e., ": positive", and check if the tokenization starts with the tokens
    of ": ". If this is the case, then we should not strip the prompts, since the
    tokeniser produces the whitespace token separately.

    Args:
        labels_to_be_generated:
            The labels that are to be generated.
        tokeniser:
            The tokeniser used to tokenize the labels.

    Returns:
        Whether we should strip the prompts.
    """
    strip_prompts = True
    for label in labels_to_be_generated:
        colon_tokens = tokeniser(": ", add_special_tokens=False).input_ids
        label_tokens = tokeniser(": " + label, add_special_tokens=False).input_ids

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


def should_prefix_space_be_added_to_labels(
    labels_to_be_generated: list[str], tokeniser: "PreTrainedTokenizer"
) -> bool:
    """Determine if we should add a prefix space to the labels.

    This is the case if the prompts are stripped and the tokeniser doesn't
    automatically add prefix whitespaces to the labels.

    Args:
        labels_to_be_generated:
            The labels that are to be generated.
        tokeniser:
            The tokeniser used to tokenize the labels.

    Returns:
        Whether we should add a prefix space to the labels.
    """
    if not should_prompts_be_stripped(
        labels_to_be_generated=labels_to_be_generated, tokeniser=tokeniser
    ):
        return False

    whitespace_token = tokeniser.convert_ids_to_tokens(
        ids=tokeniser(" ", add_special_tokens=False).input_ids[0]
    )[0]

    add_prefix_space = True
    for label in labels_to_be_generated:
        label_tokens = tokeniser(label, add_special_tokens=False).input_ids
        if isinstance(label_tokens, torch.Tensor):
            label_tokens = list(label_tokens.squeeze(0))
        first_label_token: int = int(label_tokens[0])
        first_character_of_label = tokeniser.convert_ids_to_tokens(first_label_token)[0]
        has_prefix_space = first_character_of_label == whitespace_token
        if has_prefix_space:
            add_prefix_space = False
            break

    return add_prefix_space


def get_bos_token(
    tokeniser: "PreTrainedTokenizer",
) -> tuple[str, int] | tuple[None, None]:
    """Get the beginning-of-sequence token from a tokeniser.

    Args:
        tokeniser:
            The tokeniser.

    Returns:
        A pair (token, token_id) representing the beginning-of-sequence token and its
        token ID, or (None, None) if no BOS token is found.
    """
    if isinstance(tokeniser.bos_token, str) and isinstance(tokeniser.bos_token_id, int):
        return tokeniser.bos_token, tokeniser.bos_token_id

    vocab: dict[str, int] = tokeniser.get_vocab()

    candidate_bos_tokens = ["<s>", "<|begin_of_text|>", "<|startoftext|>", "[CLS]"]
    for candidate_bos_token in candidate_bos_tokens:
        if candidate_bos_token in vocab:
            bos_token = candidate_bos_token
            bos_token_id = vocab[bos_token]
            break
    else:
        log_once(
            "The model does not have a beginning-of-sequence token. Please ensure that "
            "this has been set in the tokeniser's configuration. Using no BOS token."
            " This may lead to unexpected behavior in the model.",
            level=logging.INFO,
        )
        return None, None

    log_once(
        f"Beginning-of-sequence token was not set, but detected it as {bos_token!r} "
        f"with ID {bos_token_id}.",
        level=logging.DEBUG,
    )
    return bos_token, bos_token_id


def get_eos_token(
    tokeniser: "PreTrainedTokenizer",
) -> tuple[str, int] | tuple[None, None]:
    """Get the end-of-sequence token from a tokeniser.

    Args:
        tokeniser:
            The tokeniser.

    Returns:
        A pair (token, token_id) representing the end-of-sequence token and its token
        ID, or (None, None) if no EOS token is found.
    """
    if isinstance(tokeniser.eos_token, str) and isinstance(tokeniser.eos_token_id, int):
        return tokeniser.eos_token, tokeniser.eos_token_id

    vocab: dict[str, int] = tokeniser.get_vocab()

    candidate_eos_tokens = ["</s>", "<|end_of_text|>", "<|endoftext|>", "[SEP]"]
    for candidate_eos_token in candidate_eos_tokens:
        if candidate_eos_token in vocab:
            eos_token = candidate_eos_token
            eos_token_id = vocab[eos_token]
            break
    else:
        log_once(
            "The model does not have an end-of-sequence token. Please ensure that this "
            "has been set in the tokeniser's configuration. Using no EOS token. This "
            "may lead to unexpected behavior in the model.",
            level=logging.INFO,
        )
        return None, None

    log_once(
        f"End-of-sequence token was not set, but detected it as {eos_token!r} with "
        f"ID {eos_token_id}.",
        level=logging.DEBUG,
    )
    return eos_token, eos_token_id


def get_pad_token(
    tokeniser: "PreTrainedTokenizer",
) -> tuple[str, int] | tuple[None, None]:
    """Get the padding token from a tokeniser.

    Args:
        tokeniser:
            The tokeniser.

    Returns:
        A pair (token, token_id) representing the padding token and its token ID, or
        (None, None) if no padding token is found.
    """
    # If the tokeniser already has a padding token, return it
    if tokeniser.pad_token is not None and tokeniser.pad_token_id is not None:
        assert isinstance(tokeniser.pad_token, str), (
            "Expected tokeniser.pad_token to be a string, but got "
            f"{type(tokeniser.pad_token)}."
        )
        assert isinstance(tokeniser.pad_token_id, int), (
            "Expected tokeniser.pad_token_id to be an integer, but got "
            f"{type(tokeniser.pad_token_id)}."
        )
        return (tokeniser.pad_token, tokeniser.pad_token_id)

    # If the tokeniser has a BOS token, use it as the padding token
    if tokeniser.bos_token is not None and tokeniser.bos_token_id is not None:
        assert isinstance(tokeniser.bos_token, str), (
            "Expected tokeniser.bos_token to be a string, but got "
            f"{type(tokeniser.bos_token)}."
        )
        assert isinstance(tokeniser.bos_token_id, int), (
            "Expected tokeniser.bos_token_id to be an integer, but got "
            f"{type(tokeniser.bos_token_id)}."
        )
        pad_token = tokeniser.bos_token
        pad_token_id = tokeniser.bos_token_id

    # If the tokeniser has an EOS token, use it as the padding token
    elif tokeniser.eos_token is not None and tokeniser.eos_token_id is not None:
        assert isinstance(tokeniser.eos_token, str), (
            "Expected tokeniser.eos_token to be a string, but got "
            f"{type(tokeniser.eos_token)}."
        )
        assert isinstance(tokeniser.eos_token_id, int), (
            "Expected tokeniser.eos_token_id to be an integer, but got "
            f"{type(tokeniser.eos_token_id)}."
        )
        pad_token = tokeniser.eos_token
        pad_token_id = tokeniser.eos_token_id

    # Otherwise, try to find a candidate padding token in the vocabulary
    else:
        pad_token_candidates = [
            "<pad>",
            "[pad]",
            "<|endoftext|>",
            "<｜end▁of▁sentence｜>",
            "<|im_end|>",
        ]
        pad_token_candidates.extend([c.upper() for c in pad_token_candidates])
        for candidate in pad_token_candidates:
            if candidate in tokeniser.get_vocab():
                pad_token = candidate
                pad_token_id = tokeniser.get_vocab()[candidate]
                break
        else:
            log_once(
                "Could not identify a padding token for the model. Please ensure that "
                "this has been set in the tokeniser's configuration. Using no padding "
                "token. This may lead to unexpected behavior in the model.",
                level=logging.INFO,
            )
            return None, None

    log_once(
        f"Padding token was not set, but detected it as {pad_token!r} with ID "
        f"{pad_token_id}.",
        level=logging.DEBUG,
    )
    return pad_token, pad_token_id


def get_end_of_chat_token_ids(tokeniser: "PreTrainedTokenizer") -> list[int] | None:
    """Get the end token ID for chat models.

    This is only relevant for tokenisers with a chat template.

    Args:
        tokeniser:
            The tokeniser.

    Returns:
        The token IDs used to end chats, or None if the tokeniser does not have a chat
        template or if no end-of-chat token could be found.
    """
    if not has_chat_template(tokeniser=tokeniser):
        return None

    user_message: dict[str, str] = dict(role="user", content="X")
    token_ids = apply_chat_template(
        conversation=[user_message],
        tokeniser=tokeniser,
        tokenize=True,
        add_generation_prompt=False,
    )
    assert isinstance(token_ids, list)

    for idx, token in enumerate(tokeniser.convert_ids_to_tokens(token_ids)):
        if "X" in token:
            x_token_index = idx
            break
    else:
        logger.debug("Could not locate the end-of-chat token for the model.")
        return None

    end_of_chat_tokens = token_ids[x_token_index + 1 :]
    if len(end_of_chat_tokens) == 0:
        logger.debug("Could not locate the end-of-chat token for the model.")
        return None

    log_once(
        f"Detected end-of-chat token IDs as {end_of_chat_tokens}, corresponding to "
        f"tokens {tokeniser.convert_ids_to_tokens(end_of_chat_tokens)}.",
        level=logging.DEBUG,
    )
    return end_of_chat_tokens


def get_first_label_token_mapping(
    dataset_config: "DatasetConfig",
    model_config: "ModelConfig",
    tokeniser: "PreTrainedTokenizer | None",
    generative_type: "GenerativeType | None",
    log_metadata: bool,
) -> dict[str, str] | bool:
    """Check if the model should output scores.

    Args:
        dataset_config:
            The dataset configuration.
        model_config:
            The model configuration.
        tokeniser:
            The tokeniser, or None if not available.
        generative_type:
            The generative type, or None if not available.
        log_metadata:
            Whether to log metadata.

    Returns:
        A mapping from labels to the first token in each label, or alternatively a
        Boolean value indicating whether the model should output scores (if the mapping
        is outputted then the model will always output scores).
    """
    if not (dataset_config.task.uses_logprobs and dataset_config.labels):
        if log_metadata:
            log_once(
                "We will not use logprobs with the model, since the dataset does not "
                "have labels.",
                level=logging.DEBUG,
            )
        return False
    elif generative_type == GenerativeType.REASONING:
        if log_metadata:
            log_once(
                f"The model {model_config.model_id!r} is a reasoning model and "
                "thus does not support logprobs, so we do not enable it.",
                level=logging.DEBUG,
            )
        return False
    elif tokeniser is None:
        if log_metadata:
            log_once(
                f"We will use logprobs with the model {model_config.model_id!r} "
                "since the dataset supports it and no tokeniser is available.",
                level=logging.DEBUG,
            )
        return True

    local_labels = [
        dataset_config.prompt_label_mapping[label].strip()
        for label in dataset_config.labels
    ]

    # Tokenize some text containing each label, which we will use to extract the
    # first token of each label
    all_tokens: list[list[str]]
    if not has_chat_template(tokeniser=tokeniser):
        add_prefix_space = should_prefix_space_be_added_to_labels(
            labels_to_be_generated=local_labels, tokeniser=tokeniser
        )
        all_tokens = [
            tokeniser.tokenize(text=f" {label}" if add_prefix_space else label)
            for label in local_labels
        ]
    else:
        all_tokens = [
            tokeniser.convert_ids_to_tokens(
                ids=apply_chat_template(
                    conversation=[
                        dict(role="user", content=""),
                        dict(role="assistant", content=label),
                        # Adding extra user message as Mistral tokenisers require
                        # conversamtions to end with a user message
                        dict(role="user", content=""),
                    ],
                    tokeniser=tokeniser,
                    tokenize=True,
                )
            )
            for label in local_labels
        ]

    # Remove any non-alphabetic characters from the tokens
    all_tokens = [
        [
            re.sub(
                pattern=r"^[^a-zæøåüöä0-9]+|[^a-zæøåüöä0-9]+$",
                repl="",
                string=token.lower(),
            )
            for token in token_list
        ]
        for token_list in all_tokens
    ]

    # Extract the first token of each label
    first_tokens: list[str] = list()
    for token_list, label in zip(all_tokens, local_labels):
        matching_tokens = [tok for tok in token_list if tok and label.startswith(tok)]
        if not matching_tokens:
            if log_metadata:
                log_once(
                    f"No matching token found in token_list for label '{label}', so "
                    "we will not use logprobs with the model.",
                    level=logging.DEBUG,
                )
            return False
        first_tokens.append(matching_tokens[0])

    # Build a mapping from labels to the first token in each label if the first
    # tokens are distinct
    if len(first_tokens) == len(set(first_tokens)):
        mapping = {
            label: first_token for label, first_token in zip(local_labels, first_tokens)
        }
        if log_metadata:
            log_once(
                "Using logprobs as evaluation strategy for the model, with the "
                f"following mapping from labels to their first token: {mapping}.",
                level=logging.DEBUG,
            )
        return mapping
    else:
        if log_metadata:
            log_once(
                "We will not use logprobs with the model since the first tokens of the "
                "labels are not distinct. The first tokens for the labels "
                f"{local_labels} are {first_tokens}"
            )
        return False


def has_chat_template(tokeniser: "PreTrainedTokenizer") -> bool:
    """Check if a tokeniser has a chat template.

    Args:
        tokeniser:
            The tokeniser.

    Returns:
        Whether the tokeniser has a chat template.
    """
    if hasattr(tokeniser, "chat_template"):
        has_template = tokeniser.chat_template is not None
        if has_template:
            log_once(
                "The tokeniser has a chat template, so assuming that the model is "
                "instruction tuned.",
                level=logging.DEBUG,
            )
        return has_template
    elif isinstance(tokeniser, MistralCommonTokenizer):
        log_once(
            "The tokeniser is a Mistral tokeniser, so assuming that the model is "
            "instruction tuned.",
            level=logging.DEBUG,
        )
        return True
    else:
        log_once(
            "We cannot find a chat template for the tokeniser, so assuming that the "
            "model isn't instruction tuned.",
            level=logging.DEBUG,
        )
        return False


def apply_chat_template(
    conversation: list[dict[str, str]],
    tokeniser: "PreTrainedTokenizer",
    tokenize: bool = False,
    add_generation_prompt: bool = True,
    **transformers_tokeniser_kwargs,
) -> str | list[int]:
    """Apply the chat template to a prompt.

    Args:
        conversation:
            The conversation to apply the chat template to.
        tokeniser:
            The tokeniser.
        tokenize:
            Whether to tokenize the resulting prompt, returning a list of token IDs
            instead of a string.
        add_generation_prompt:
            Whether to add a generation prompt at the end of the conversation. This is
            only relevant for regular Hugging Face tokenisers, as Mistral tokenisers
            always add a generation prompt.
        **transformers_tokeniser_kwargs:
            Additional keyword arguments to pass to the tokeniser, in case the tokeniser
            is a regular Hugging Face tokeniser.

    Returns:
        The prompt with the chat template applied, either as a string or a list of
        token IDs, depending on the value of `tokenize`.

    Raises:
        InvalidModel:
            If the tokeniser does not have a chat template.
    """
    if not has_chat_template(tokeniser=tokeniser):
        raise InvalidModel(
            "The tokeniser does not have a chat template, so cannot apply it."
        )
    elif isinstance(tokeniser, MistralCommonTokenizer):
        templated_prompt = tokeniser.apply_chat_template(
            conversation=conversation, tokenize=tokenize
        )
    else:
        templated_prompt = tokeniser.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **transformers_tokeniser_kwargs,
        )
    return templated_prompt
