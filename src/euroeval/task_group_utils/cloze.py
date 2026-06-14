"""Utilities for Cloze Formulation (CF) evaluation of multiple-choice tasks."""

import collections.abc as c
import typing as t

import numpy as np

from ..exceptions import InvalidBenchmark
from ..string_utils import CHOICE_LETTERS

if t.TYPE_CHECKING:
    from datasets import DatasetDict
    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    from ..data_models import DatasetConfig, GenerativeModelOutput, ModelConfig
    from ..types import Tokeniser


def build_cf_prompt(
    bare_input: str,
    few_shot_rendered: c.Sequence[str],
    prompt_template: str,
    prompt_prefix: str,
) -> str:
    r"""Assemble the final CF prompt that the model will score candidates against.

    The prompt ends with whatever trailing marker the template renders for an
    empty `{label}` (typically `"Antwoord: "` or `"Answer: "`), so that
    candidate texts can be appended directly for scoring.

    Args:
        bare_input:
            The bare question text (no enumerated choices).
        few_shot_rendered:
            Pre-rendered few-shot sections, each of the form
            `"Vraag: <q>\nAntwoord: <full-answer-text>"`.
        prompt_template:
            The dataset's few-shot prompt template with `{text}` and `{label}`
            placeholders.
        prompt_prefix:
            The dataset's prompt prefix (may be empty). `{labels_str}`, if
            present, is substituted with an empty string because CF does not
            enumerate the label letters.

    Returns:
        The full prompt string.
    """
    # label="" renders the template's trailing marker (e.g. "Answer: ") without
    # any choice text, so candidates can be appended directly for scoring.
    tail = prompt_template.format(text=bare_input.replace("\n", " ").strip(), label="")
    # Only {labels_str} is currently used in prompt_prefix templates.
    rendered_prefix = prompt_prefix.format(labels_str="") if prompt_prefix else ""
    if rendered_prefix:
        rendered_prefix += "\n\n"
    few_shot_block = "\n\n".join(few_shot_rendered)
    if few_shot_block:
        few_shot_block += "\n\n"
    return rendered_prefix + few_shot_block + tail


def render_cf_few_shot(bare_input: str, answer_text: str, prompt_template: str) -> str:
    """Render a single few-shot example for CF with the full answer text.

    Args:
        bare_input:
            The bare question of the few-shot example.
        answer_text:
            The full text of the correct answer choice.
        prompt_template:
            The dataset's few-shot prompt template.

    Returns:
        The rendered few-shot section.
    """
    return prompt_template.format(
        text=bare_input.replace("\n", " ").strip(),
        label=answer_text.replace("\n", " ").strip(),
    )


def normalize_cf_score(token_logprobs: c.Sequence[float], answer_text: str) -> float:
    """Compute the normalized CF score for a single answer candidate.

    Scores are normalized by the character length of the answer text, matching
    the default used by Llama and the EleutherAI LM Evaluation Harness.

    Args:
        token_logprobs:
            Per-token logprobs of the candidate continuation.
        answer_text:
            The raw candidate text, used for character-length normalization.

    Returns:
        The normalized score. Higher is better.
    """
    total = float(sum(token_logprobs))
    return total / max(1, len(answer_text))


def extract_labels_from_cf(
    input_batch: dict[str, list],
    model_output: "GenerativeModelOutput",
    dataset_config: "DatasetConfig",
    model_config: "ModelConfig",
    first_label_token_mapping: dict[str, str] | bool,
) -> c.Sequence[str]:
    """Pick the highest-scoring choice letter for each sample.

    The signature mirrors
    `euroeval.task_group_utils.sequence_classification.extract_labels_from_generation`
    so that the backend's `extract_labels_from_generation` property can
    drop this in for MCQ samples evaluated with CF.

    Args:
        input_batch:
            The input batch (unused, kept for signature parity).
        model_output:
            The generative model output with `cf_scores` populated (shape
            `(batch_size, num_choices)`).
        dataset_config:
            The dataset configuration (unused, kept for signature parity).
        model_config:
            The model configuration (unused, kept for signature parity).
        first_label_token_mapping:
            Unused, kept for signature parity with the MCF extractor.

    Returns:
        The predicted letter label ("a", "b", ...) for each sample.

    Raises:
        InvalidBenchmark:
            If `cf_scores` is missing from `model_output`.
    """
    del input_batch, dataset_config, model_config, first_label_token_mapping
    if model_output.cf_scores is None:
        raise InvalidBenchmark(
            "CF evaluation expected `cf_scores` on the model output, but none was "
            "produced. This is likely a bug."
        )
    return [CHOICE_LETTERS[int(np.argmax(row))] for row in model_output.cf_scores]


def letter_to_choice_text(letter: str, raw_choices: c.Sequence[str]) -> str:
    """Return the full choice text corresponding to a letter label.

    Args:
        letter:
            A single lowercase letter ("a", "b", ...).
        raw_choices:
            The ordered list of raw choice strings for the example.

    Returns:
        The raw choice string at the index encoded by `letter`.

    Raises:
        InvalidBenchmark:
            If the letter does not correspond to a choice in `raw_choices`.
    """
    letter = letter.strip().lower()
    idx = CHOICE_LETTERS.find(letter)
    if idx == -1 or idx >= len(raw_choices):
        raise InvalidBenchmark(
            f"Could not map label letter {letter!r} to a choice; "
            f"available choices: {list(raw_choices)!r}."
        )
    return raw_choices[idx]


def prepare_cf_dataset(
    dataset: "DatasetDict",
    few_shot_examples: c.Sequence[dict[str, t.Any]],
    prompt_template: str,
    prompt_prefix: str,
) -> "DatasetDict":
    """Prepare a dataset for Cloze Formulation evaluation.

    Builds CF prompts with bare questions and few-shot examples using full
    answer text (not letter labels). Used only by vLLM backend.

    Args:
        dataset:
            The dataset dict with a "test" split.
        few_shot_examples:
            List of few-shot examples to render.
        prompt_template:
            The dataset's prompt template.
        prompt_prefix:
            The dataset's prompt prefix.

    Returns:
        The prepared dataset with CF prompts.
    """

    def _ensure_cf_columns(example: dict) -> dict:
        # Preprocessing always provides these columns; this is a sanity check
        return {
            "bare_input": example["bare_input"],
            "raw_choices": example["raw_choices"],
        }

    few_shot_examples = [{**ex, **_ensure_cf_columns(ex)} for ex in few_shot_examples]
    few_shot_rendered = [
        render_cf_few_shot(
            bare_input=ex["bare_input"],
            answer_text=letter_to_choice_text(
                letter=str(ex["label"]), raw_choices=ex["raw_choices"]
            ),
            prompt_template=prompt_template,
        )
        for ex in few_shot_examples
    ]

    dataset["test"] = dataset["test"].map(
        _ensure_cf_columns, load_from_cache_file=False, keep_in_memory=True
    )

    def _build_cf_prompt(example: dict) -> dict:
        text = build_cf_prompt(
            bare_input=example["bare_input"],
            few_shot_rendered=few_shot_rendered,
            prompt_template=prompt_template,
            prompt_prefix=prompt_prefix,
        )
        return {"text": text, "prompt": text}

    dataset["test"] = dataset["test"].map(
        _build_cf_prompt, load_from_cache_file=False, keep_in_memory=True
    )
    return dataset


def score_completions(
    model: "LLM",
    tokeniser: "Tokeniser",
    prompts: c.Sequence[str],
    completions: c.Sequence[c.Sequence[str]],
    sampling_params: "SamplingParams",
) -> c.Sequence[c.Sequence[c.Sequence[float]]]:
    """Score each (prompt, candidate) pair with vLLM's prompt_logprobs.

    Args:
        model:
            The vLLM model instance.
        tokeniser:
            The tokenizer for the model.
        prompts:
            One bare-question prompt per sample. Candidates are appended
            directly to the prompt before scoring.
        completions:
            Per-sample list of candidate continuation strings.
        sampling_params:
            vLLM sampling parameters for scoring.

    Returns:
        Per-sample, per-candidate, per-token logprobs.

    Raises:
        InvalidBenchmark:
            If a candidate is empty after tokenization.
    """
    flat_prompts: list[str] = []
    spans: list[list[tuple[int, int]]] = []
    for prompt, candidates in zip(prompts, completions):
        prompt_ids = list(tokeniser(prompt, add_special_tokens=False).input_ids)
        group: list[tuple[int, int]] = []
        for candidate in candidates:
            if not candidate:
                raise InvalidBenchmark(
                    f"Candidate is empty after tokenisation: cannot score an "
                    f"empty continuation. Prompt: {prompt!r}, candidate: "
                    f"{candidate!r}."
                )
            full = prompt + candidate
            full_ids = list(tokeniser(full, add_special_tokens=False).input_ids)
            if len(full_ids) <= len(prompt_ids) and (
                full_ids == prompt_ids[: len(full_ids)]
            ):
                raise InvalidBenchmark(
                    f"Candidate is empty after tokenisation: cannot score an "
                    f"empty continuation. Prompt: {prompt!r}, candidate: "
                    f"{candidate!r}."
                )
            prefix_len = len(prompt_ids)
            while prefix_len > 0 and (
                prefix_len >= len(full_ids)
                or full_ids[:prefix_len] != prompt_ids[:prefix_len]
            ):
                prefix_len -= 1
            if prefix_len >= len(full_ids):
                raise InvalidBenchmark(
                    f"Candidate is empty after tokenisation: cannot score an "
                    f"empty continuation. Prompt: {prompt!r}, candidate: "
                    f"{candidate!r}."
                )
            flat_prompts.append(full)
            group.append((prefix_len, len(full_ids)))
        spans.append(group)

    raw_outputs = model.generate(
        prompts=flat_prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
        lora_request=None,
    )

    out: list[list[list[float]]] = []
    cursor = 0
    for group in spans:
        per_sample: list[list[float]] = []
        for prompt_len, full_len in group:
            prompt_logprobs = raw_outputs[cursor].prompt_logprobs or []
            token_lps: list[float] = []
            for pos in range(prompt_len, full_len):
                entry = prompt_logprobs[pos]
                if not entry:
                    token_lps.append(0.0)
                    continue
                token_lps.append(float(next(iter(entry.values())).logprob))
            per_sample.append(token_lps)
            cursor += 1
        out.append(per_sample)
    return out


def generate_cf(
    model: "LLM",
    tokeniser: "Tokeniser",
    inputs: dict[str, t.Any],
    sampling_params: "SamplingParams",
) -> "GenerativeModelOutput":
    """Score each candidate answer with Cloze Formulation using vLLM.

    Args:
        model:
            The vLLM model instance.
        tokeniser:
            The tokenizer for the model.
        inputs:
            Input batch containing "text" (bare-question prompts) and
            "raw_choices" (list of candidate strings per sample).
        sampling_params:
            vLLM sampling parameters for scoring.

    Returns:
        GenerativeModelOutput with empty sequences and populated cf_scores.

    Raises:
        InvalidBenchmark:
            If raw_choices is missing from inputs.
    """
    if "raw_choices" not in inputs:
        raise InvalidBenchmark(
            "Cloze Formulation evaluation requires a `raw_choices` column, "
            "which was not found in the batch."
        )
    prompts: c.Sequence[str] = inputs["text"]
    raw_choices: c.Sequence[c.Sequence[str]] = inputs["raw_choices"]

    per_token_logprobs = score_completions(
        model=model,
        tokeniser=tokeniser,
        prompts=prompts,
        completions=raw_choices,
        sampling_params=sampling_params,
    )

    cf_scores: list[list[float]] = [
        [
            normalize_cf_score(token_logprobs=token_lps, answer_text=answer)
            for token_lps, answer in zip(sample_lps, sample_choices)
        ]
        for sample_lps, sample_choices in zip(per_token_logprobs, raw_choices)
    ]

    return GenerativeModelOutput(sequences=[""] * len(prompts), cf_scores=cf_scores)
