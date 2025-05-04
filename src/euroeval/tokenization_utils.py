"""Utility functions related to tokenization."""

import logging
import re
import typing as t

import torch

from .constants import TASK_GROUPS_USING_LOGPROBS
from .enums import GenerativeType
from .exceptions import InvalidModel
from .utils import log_once

if t.TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from .data_models import DatasetConfig, ModelConfig


logger = logging.getLogger("euroeval")


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

    candidate_bos_tokens = ["<s>", "<|begin_of_text|>", "<|startoftext|>", "[CLS]"]
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

    candidate_eos_tokens = ["</s>", "<|end_of_text|>", "<|endoftext|>", "[SEP]"]
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

    user_message: dict[str, str] = dict(role="user", content="X")
    token_ids: list[int] = tokenizer.apply_chat_template(conversation=[user_message])  # type: ignore[assignment]

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


def get_first_label_token_mapping(
    dataset_config: "DatasetConfig",
    model_config: "ModelConfig",
    tokenizer: "PreTrainedTokenizer | None",
    generative_type: "GenerativeType | None",
) -> dict[str, str] | bool:
    """Check if the model should output scores.

    Args:
        dataset_config:
            The dataset configuration.
        model_config:
            The model configuration.
        tokenizer:
            The tokenizer, or None if not available.
        generative_type:
            The generative type, or None if not available.

    Returns:
        A mapping from labels to the first token in each label, or alternatively a
        Boolean value indicating whether the model should output scores (if the mapping
        is outputted then the model will always output scores).
    """
    if generative_type == GenerativeType.REASONING:
        log_once(
            f"The model {model_config.model_id!r} is a reasoning model and "
            "thus does not support logprobs, so we do not enable it.",
            level=logging.DEBUG,
        )
        return False

    # If we do not have any tokenizer, then we cannot check if the model should output
    # scores and we just assume it should if the dataset supports it
    output_scores = dataset_config.task.task_group in TASK_GROUPS_USING_LOGPROBS
    if tokenizer is None:
        if output_scores:
            log_once(
                f"The model {model_config.model_id!r} will output scores, since the "
                "dataset supports it and no tokenizer is available.",
                level=logging.DEBUG,
            )
        else:
            log_once(
                f"The model {model_config.model_id!r} will not output scores, since "
                "the dataset does not support it and no tokenizer is available.",
                level=logging.DEBUG,
            )
        return output_scores

    # If there are labels associated with the dataset, and that the first token of each
    # label is distinct, then we can safely use the logprobs
    if output_scores and dataset_config.labels:
        local_labels = [
            dataset_config.prompt_label_mapping[label].strip()
            for label in dataset_config.labels
        ]

        # Tokenize some text containing each label, which we will use to extract the
        # first token of each label
        all_tokens: list[list[str]]
        if tokenizer.chat_template is None:
            add_prefix_space = should_prefix_space_be_added_to_labels(
                labels_to_be_generated=local_labels, tokenizer=tokenizer
            )
            all_tokens = [
                tokenizer.tokenize(text=f" {label}" if add_prefix_space else label)
                for label in local_labels
            ]
        else:
            all_tokens = [
                tokenizer.convert_ids_to_tokens(
                    ids=tokenizer.apply_chat_template(
                        conversation=[
                            dict(role="user", content=""),
                            dict(role="assistant", content=label),
                        ],
                        add_generation_prompt=True,
                        tokenize=True,
                    )
                )
                for label in local_labels
            ]

        # Remove any non-alphabetic characters from the tokens
        all_tokens = [
            [
                re.sub(
                    pattern=r"^[^a-zæøåüöä]+|[^a-zæøåüöä]+$",
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
            matching_tokens = [
                tok for tok in token_list if tok and label.startswith(tok)
            ]
            if not matching_tokens:
                log_once(
                    f"No matching token found in token_list for label '{label}', so "
                    "we will not output scores.",
                    level=logging.DEBUG,
                )
                return False
            first_tokens.append(matching_tokens[0])

        # Build a mapping from labels to the first token in each label if the first
        # tokens are distinct
        if len(first_tokens) == len(set(first_tokens)):
            log_once(
                "The model will output scores, since the first tokens of the labels "
                "are distinct.",
                level=logging.DEBUG,
            )
            return {
                label: first_token
                for label, first_token in zip(local_labels, first_tokens)
            }
        else:
            log_once(
                "The model will not output scores, since the first tokens of the "
                "labels are not distinct. The first tokens for the labels "
                f"{local_labels} are {first_tokens}"
            )
            return False

    # Otherwise, we assume that the model should not output scores, to avoid potential
    # evaluation errors. This will force the label extraction to rely on word edit
    # distance instead of logprobs.
    log_once(
        "The model will not output scores, since the dataset does not have labels.",
        level=logging.DEBUG,
    )
    return False
