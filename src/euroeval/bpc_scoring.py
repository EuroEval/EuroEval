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

from .exceptions import InvalidModel
from .logging_utils import log_once
from .types import Tokeniser


def compute_bpc_scores(
    raw_outputs: c.Sequence["RequestOutput"],
    prompts: c.Sequence[str],
    answer_texts: c.Sequence[str],
    answer_start_indices: c.Sequence[int],
    tokeniser: Tokeniser,
    answer_char_counts: c.Sequence[int] | None = None,
) -> list[float]:
    """Compute bits-per-character scores from prompt_logprobs.

    BPC = -sum(log₂(P)) / num_chars. Lower is better. Typical range: 0.5-3.0.
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
        answer_char_counts:
            The number of characters to divide each answer's bits by. Defaults to the
            length of each answer text when omitted. For token classification (NER) this
            is the entity-text character count rather than the full serialised-answer
            length, so the predictable JSON scaffolding does not dominate the score.

    Returns:
        BPC scores (positive floats, lower is better).
    """
    if answer_char_counts is None:
        answer_char_counts = [len(answer) for answer in answer_texts]

    bpc_scores: list[float] = []

    for idx, (
        raw_output,
        prompt,
        answer,
        answer_start_idx,
        answer_char_count,
    ) in enumerate(
        zip(
            raw_outputs, prompts, answer_texts, answer_start_indices, answer_char_counts
        )
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
            bpc = total_logprob / max(1, answer_char_count)
        else:
            # No answer tokens extracted = infinite BPC (worst possible score)
            bpc = float("inf")

        # Log the first scored example so the cloze formulation can be eyeballed: the
        # prompt should end with the bare question and the appended answer text, and the
        # answer span should cover exactly that answer.
        if idx == 0:
            scored_token_ids = full_tokens[start_idx:]
            scored_text = tokeniser.decode(scored_token_ids)
            log_once(
                "BPC (cloze) scoring of the first example:\n"
                f"  Full scored prompt: {prompt!r}\n"
                f"  Gold answer text: {answer!r}\n"
                f"  Answer start token index (pre/post special-token shift): "
                f"{answer_start_idx}/{start_idx}\n"
                f"  Decoded scored answer span: {scored_text!r}\n"
                f"  Answer tokens scored: {len(answer_logprobs)}, "
                f"characters divided by: {answer_char_count} "
                f"(full answer length: {len(answer)})\n"
                f"  Bits per character: {bpc:.4f}",
                level=logging.DEBUG,
            )

        bpc_scores.append(bpc)

    return bpc_scores


def compute_bpc_scores_for_vllm_outputs(
    raw_outputs: c.Sequence["RequestOutput"], inputs: dict, tokeniser: Tokeniser
) -> list[float] | None:
    """Compute BPC scores from vLLM outputs.

    High-level wrapper that reads the pre-computed gold answer texts and computes BPC
    scores.

    Args:
        raw_outputs: Raw outputs from vLLM with prompt_logprobs.
        inputs:
            Model inputs containing the ``bpc_prompt``, ``bpc_answer_start``,
            ``bpc_answer_text`` and ``bpc_answer_char_count`` columns produced during
            dataset preparation.
        tokeniser: Tokeniser for tokenisation.

    Returns:
        BPC scores, or None if there are no answer texts to score.

    Raises:
        InvalidModel:
            If the active vLLM backend does not return per-token prompt logprobs,
            which are mandatory for BPC scoring (e.g. the Apple Metal plugin).
    """
    # The gold answer texts are derived once during dataset preparation and carried in
    # the `bpc_answer_text` column, guaranteeing they match the answers appended to
    # `bpc_prompt`. An absent column or all-empty texts means the task group has no
    # scoreable answer, so BPC is skipped.
    answer_texts = inputs.get("bpc_answer_text")
    if not answer_texts or all(answer == "" for answer in answer_texts):
        return None

    # Per-token prompt logprobs are mandatory for BPC. Some vLLM backends (notably the
    # Apple Metal plugin `vllm_metal`) do not compute them and return an empty or
    # degenerate structure (e.g. a single placeholder entry for a multi-token prompt),
    # so detect that and fail fast with clear guidance instead of silently reporting
    # infinite scores.
    backend_provides_prompt_logprobs = all(
        raw_output.prompt_logprobs is not None
        and raw_output.prompt_token_ids is not None
        and len(raw_output.prompt_logprobs) >= len(raw_output.prompt_token_ids)
        for raw_output in raw_outputs
    )
    if not backend_provides_prompt_logprobs:
        raise InvalidModel(
            "Bits-per-character (BPC) scoring requires per-token prompt logprobs, but "
            "the active vLLM backend did not return them. This is a known limitation "
            "of the Apple Metal plugin (`vllm_metal`), which does not compute prompt "
            "logprobs; BPC scoring is currently only supported on the CUDA vLLM "
            "backend."
        )

    return compute_bpc_scores(
        raw_outputs=raw_outputs,
        prompts=inputs["bpc_prompt"],
        answer_texts=answer_texts,
        answer_start_indices=inputs["bpc_answer_start"],
        tokeniser=tokeniser,
        answer_char_counts=inputs.get("bpc_answer_char_count"),
    )
