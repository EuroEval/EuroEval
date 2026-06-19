"""Bits-per-character (BPC) scoring logic for language model evaluation.

This module provides functions for computing BPC scores from vLLM prompt_logprobs.
BPC is a character-level metric where lower scores indicate better performance.
"""

import collections.abc as c
import logging
import math
import typing as t

if t.TYPE_CHECKING:
    from vllm.outputs import RequestOutput

from .data_models import DatasetConfig
from .enums import TaskGroup
from .logging_utils import log_once
from .task_group_utils.cloze import letter_to_choice_text
from .types import Tokeniser


def compute_bpc_scores(
    raw_outputs: c.Sequence["RequestOutput"],
    prompts: c.Sequence[str],
    answer_texts: c.Sequence[str],
    answer_start_indices: c.Sequence[int],
    tokeniser: Tokeniser,
) -> list[float]:
    """Compute bits-per-character scores from prompt_logprobs.

    BPC = -sum(log₂(P)) / len(chars). Lower is better. Typical range: 0.5-3.0.
    Logprobs are in natural log base, so we convert to log₂ and negate to get
    positive bits per character.

    This function extracts the logprobs for the answer portion only from the
    prompt_logprobs. The answer is assumed to be appended to the prompt.

    Args:
        raw_outputs: Raw outputs from vLLM model.generate() with prompt_logprobs.
        prompts: The full prompts (prompt + answer text).
        answer_texts: Ground truth answer texts that were appended to prompts.
        answer_start_indices: Pre-computed token indices where answers start in prompts.
        tokeniser: Tokeniser for tokenising prompts and answers.

    Returns:
        BPC scores (positive floats, lower is better).
    """
    bpc_scores: list[float] = []

    for raw_output, prompt, answer, answer_start_idx in zip(
        raw_outputs, prompts, answer_texts, answer_start_indices
    ):
        prompt_logprobs = raw_output.prompt_logprobs
        prompt_token_ids = raw_output.prompt_token_ids

        if prompt_logprobs is None or prompt_token_ids is None:
            # Missing prompt_logprobs/token ids = infinite BPC (worst possible score)
            bpc_scores.append(float("inf"))
            continue

        # Align with the exact tokens vLLM scored: prompt_logprobs[i] corresponds to
        # prompt_token_ids[i]. Re-encoding the prompt ourselves can disagree with vLLM
        # (which prepends special tokens such as BOS for many decoder models), which
        # would misalign the answer span so that no answer logprobs are extracted and
        # every score collapses to infinity.
        full_tokens = list(prompt_token_ids)

        # The precomputed answer-start index was derived from a tokenisation without
        # special tokens; shift it by however many leading special tokens vLLM added so
        # it indexes correctly into full_tokens / prompt_logprobs.
        plain_prompt_length = len(tokeniser.encode(prompt, add_special_tokens=False))
        special_token_offset = max(0, len(full_tokens) - plain_prompt_length)
        start_idx = answer_start_idx + special_token_offset

        # The answer tokens run from start_idx to the end.
        # prompt_logprobs[0] is None (no logprob for the first token), and
        # prompt_logprobs[i] corresponds to token i in the full sequence.
        answer_logprobs: list[float] = []
        for i in range(start_idx, len(prompt_logprobs)):
            if prompt_logprobs[i] is None:
                continue
            logprob_dict = prompt_logprobs[i]
            if isinstance(logprob_dict, dict):
                # Get the logprob for the actual token at this position
                if i < len(full_tokens):
                    token_id = full_tokens[i]
                    if token_id in logprob_dict:
                        lp = logprob_dict[token_id]
                        # Handle both dict objects and raw logprob values
                        if hasattr(lp, "logprob"):
                            answer_logprobs.append(lp.logprob)
                        else:
                            answer_logprobs.append(float(lp))

        if answer_logprobs:
            # Convert from natural log to log₂ and negate to get positive bits.
            # vLLM logprobs are negative (log of probability < 1), so -sum(negative)
            # yields positive BPC. E.g. logprob=-0.693 (ln(0.5)) → BPC=1.0 bits/char.
            total_logprob = -sum(lp / math.log(2) for lp in answer_logprobs)
            bpc = total_logprob / max(1, len(answer))
        else:
            # No answer tokens extracted = infinite BPC (worst possible score)
            bpc = float("inf")

        bpc_scores.append(bpc)

    return bpc_scores


def extract_answer_texts(inputs: dict, dataset_config: DatasetConfig) -> list[str]:
    """Extract ground truth answer texts from inputs based on task group.

    Used by BPC scoring to determine which text spans to evaluate against
    prompt_logprobs.

    Args:
        inputs:
            Model inputs containing labels and task-specific fields.
        dataset_config:
            Configuration for the dataset including labels and task metadata.

    Returns:
        List of answer texts to score. Empty list if the task type is not
        supported or required fields are missing.
    """
    answer_texts: list[str] = []
    task_group = dataset_config.task.task_group
    labels = inputs.get("label", [])

    match task_group:
        case TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION:
            if "raw_choices" not in inputs:
                log_once(
                    "BPC scoring requires 'raw_choices' in inputs for MCQ "
                    "tasks, but they are not present. Skipping BPC "
                    "computation.",
                    level=logging.WARNING,
                )
            else:
                raw_choices = inputs["raw_choices"]
                answer_texts = [
                    letter_to_choice_text(
                        letter=str(label).strip().lower(), raw_choices=raw_choice
                    )
                    for label, raw_choice in zip(labels, raw_choices)
                ]

        case TaskGroup.SEQUENCE_CLASSIFICATION:
            # The answer text must match what was appended to `bpc_prompt` during
            # dataset preparation, which uses the localized prompt label mapping.
            answer_texts = [
                dataset_config.prompt_label_mapping.get(label, label)
                for label in labels
            ]

        case TaskGroup.TEXT_TO_TEXT:
            if "target_text" not in inputs:
                log_once(
                    "BPC scoring requires 'target_text' in inputs for "
                    "text-to-text tasks, but they are not present. "
                    "Skipping BPC computation.",
                    level=logging.WARNING,
                )
            else:
                answer_texts = inputs["target_text"]

        case TaskGroup.QUESTION_ANSWERING:
            if not all(isinstance(lbl, dict) and "answers" in lbl for lbl in labels):
                log_once(
                    "BPC scoring requires structured 'answers' in label "
                    "for QA tasks, but the format is unexpected. Skipping "
                    "BPC computation.",
                    level=logging.WARNING,
                )
            else:
                answer_texts = [lbl["answers"]["text"][0] for lbl in labels]

        case TaskGroup.TOKEN_CLASSIFICATION:
            # Serialize NER tags to text
            if "tokens" not in inputs or "labels" not in inputs:
                log_once(
                    "BPC scoring requires 'tokens' and 'labels' in inputs "
                    "for token classification tasks, but they are not "
                    "present. Skipping BPC computation.",
                    level=logging.WARNING,
                )
            else:
                tokens_list = inputs["tokens"]
                labels_list = inputs["labels"]
                answer_texts = []
                for tokens, example_labels in zip(tokens_list, labels_list):
                    # Group tokens by BIO tags
                    tagged_entities: dict[str, list[str]] = {}
                    current_entity: str | None = None
                    current_tokens: list[str] = []

                    for token, tag in zip(tokens, example_labels):
                        tag_str = str(tag).lower()
                        if tag_str == "o":
                            if current_entity is not None:
                                tagged_entities.setdefault(current_entity, []).append(
                                    " ".join(current_tokens)
                                )
                                current_entity = None
                                current_tokens = []
                            continue

                        if tag_str.startswith("b-"):
                            if current_entity is not None:
                                tagged_entities.setdefault(current_entity, []).append(
                                    " ".join(current_tokens)
                                )
                            current_entity = tag_str[2:]
                            current_tokens = [token]
                        elif tag_str.startswith("i-") and current_entity:
                            current_tokens.append(token)
                        else:
                            if current_entity is not None:
                                tagged_entities.setdefault(current_entity, []).append(
                                    " ".join(current_tokens)
                                )
                            current_entity = (
                                tag_str[2:] if tag_str.startswith("i-") else tag_str
                            )
                            current_tokens = [token]

                    if current_entity is not None:
                        tagged_entities.setdefault(current_entity, []).append(
                            " ".join(current_tokens)
                        )

                    answer_texts.append(str(tagged_entities))

        case _:
            log_once(
                f"BPC scoring not implemented for task group {task_group}. "
                "Skipping BPC computation.",
                level=logging.WARNING,
            )

    return answer_texts


def compute_bpc_scores_for_vllm_outputs(
    raw_outputs: c.Sequence["RequestOutput"],
    inputs: dict,
    dataset_config: "DatasetConfig",
    tokeniser: Tokeniser,
) -> list[float] | None:
    """Compute BPC scores from vLLM outputs.

    High-level wrapper that extracts answer texts and computes BPC scores.

    Args:
        raw_outputs: Raw outputs from vLLM with prompt_logprobs.
        inputs: Model inputs containing bpc_prompt and bpc_answer_start.
        dataset_config: Dataset configuration for task type.
        tokeniser: Tokeniser for tokenisation.

    Returns:
        BPC scores if prompt_logprobs are available, None otherwise.
    """
    # Extract answer texts based on task type
    answer_texts = extract_answer_texts(inputs=inputs, dataset_config=dataset_config)

    # Check if prompt_logprobs are available
    has_logprobs = all(
        hasattr(raw_output, "prompt_logprobs")
        and raw_output.prompt_logprobs is not None
        for raw_output in raw_outputs
    )

    if answer_texts and has_logprobs:
        return compute_bpc_scores(
            raw_outputs=raw_outputs,
            prompts=inputs["bpc_prompt"],
            answer_texts=answer_texts,
            answer_start_indices=inputs["bpc_answer_start"],
            tokeniser=tokeniser,
        )
    elif answer_texts:
        log_once(
            "BPC mode enabled but prompt_logprobs not available. "
            "Skipping BPC computation.",
            level=logging.WARNING,
        )
    return None
