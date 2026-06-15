"""Bits-per-character (BPC) scoring logic for language model evaluation.

This module provides functions for computing BPC scores from vLLM prompt_logprobs.
BPC is a character-level metric where lower scores indicate better performance.
"""

import collections.abc as c
import math
import typing as t

from .types import Tokeniser


def compute_bpc_scores(
    raw_outputs: c.Sequence[t.Any],
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

        if prompt_logprobs is None:
            # Missing prompt_logprobs = infinite BPC (worst possible score)
            bpc_scores.append(float("inf"))
            continue

        # Tokenise the full prompt (including answer) to get all tokens
        full_tokens = tokeniser.encode(prompt, add_special_tokens=False)

        # The answer tokens are from answer_start_idx to end
        # prompt_logprobs[0] is None (no logprob for first token)
        # prompt_logprobs[i] corresponds to token i in the full sequence
        answer_logprobs: list[float] = []
        for i in range(answer_start_idx, len(prompt_logprobs)):
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
