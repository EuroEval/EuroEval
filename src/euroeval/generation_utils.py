"""Utility functions related to generative models."""

import collections.abc as c
import itertools as it
import logging
import random
import re
import typing as t

from datasets import Dataset

from .enums import GenerativeType, TaskGroup
from .exceptions import InvalidBenchmark, InvalidModel
from .logging_utils import log_once
from .string_utils import extract_multiple_choice_labels
from .task_group_utils.cloze import (
    letter_to_choice_text,
    parse_bare_question_and_choices,
)
from .task_group_utils.token_classification import (
    serialise_ner_tags,
    serialised_ner_content_length,
)
from .tokenisation_utils import apply_chat_template, should_prompts_be_stripped

if t.TYPE_CHECKING:
    from datasets import DatasetDict
    from transformers.tokenization_utils import PreTrainedTokenizer

    from .data_models import BenchmarkConfig, DatasetConfig, ModelConfig


def _extract_classification_examples(
    shuffled_train: Dataset,
    num_few_shots: int,
    dataset_config: "DatasetConfig",
    random_seed: int,
) -> list[dict[str, t.Any]]:
    """Extract few-shot examples for classification task groups.

    Args:
        shuffled_train:
            The shuffled training dataset.
        num_few_shots:
            The number of few-shot examples to extract.
        dataset_config:
            The dataset configuration.
        random_seed:
            The random seed for reproducibility.

    Returns:
        A list of few-shot examples.

    Raises:
        InvalidBenchmark:
            If not enough short examples are found.
    """
    # Locate the maximum number of tokens that constitutes a short example
    for max_num_tokens in [512, 1024, 2048, 4096, 8192]:
        train_with_short_examples = shuffled_train.filter(
            lambda example: len(example["text"]) < max_num_tokens
        )
        num_short_examples = len(train_with_short_examples)
        if num_short_examples >= num_few_shots:
            break
    else:
        raise InvalidBenchmark(
            "Could not find enough short examples for few-shot learning."
        )

    shuffled_train = train_with_short_examples.shuffle(seed=random_seed)
    few_shot_examples: list[dict[str, t.Any]] = list()

    if dataset_config.labels:
        labels = it.cycle(dataset_config.labels)
        labels_with_no_samples: set[str] = set()
        while len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0:
            if len(labels_with_no_samples) == len(dataset_config.labels):
                raise InvalidBenchmark(
                    "Could not find enough examples for few-shot learning. "
                    "Please check the dataset and the labels."
                )
            label = next(labels)
            possible_examples = shuffled_train.filter(
                lambda x: str(x["label"]).lower() == label.lower()
            )
            assert isinstance(possible_examples, Dataset), (
                f"Expected `possible_examples` to be a Dataset, but got "
                f"{type(possible_examples)} instead."
            )
            if len(possible_examples) == 0:
                labels_with_no_samples.add(label)
                continue
            example = possible_examples.select(range(1))[0]
            assert isinstance(example, dict), (
                f"Expected `example` to be a dict, but got {type(example)} instead."
            )
            few_shot_examples.append(example)
            shuffled_train = shuffled_train.filter(
                lambda x: x["text"] != example["text"]
            )
    else:
        # No labels defined (e.g. community datasets with variable number of
        # choices) — fall back to random sampling.
        while len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0:
            example = shuffled_train.select(range(1))[0]
            assert isinstance(example, dict), (
                f"Expected `example` to be a dict, but got {type(example)} instead."
            )
            few_shot_examples.append(example)
            shuffled_train = shuffled_train.filter(
                lambda x: x["text"] != example["text"]
            )

    return few_shot_examples


def _extract_text_to_text_examples(
    shuffled_train: Dataset, num_few_shots: int
) -> list[dict[str, t.Any]]:
    """Extract few-shot examples for text-to-text task group.

    Args:
        shuffled_train:
            The shuffled training dataset.
        num_few_shots:
            The number of few-shot examples to extract.

    Returns:
        A list of few-shot examples.
    """
    few_shot_examples: list[dict[str, t.Any]] = list()
    while len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0:
        example = shuffled_train.select(range(1))[0]
        assert isinstance(example, dict), (
            f"Expected `example` to be a dict, but got {type(example)} instead."
        )
        few_shot_examples.append(example)
        shuffled_train = shuffled_train.filter(lambda x: x["text"] != example["text"])
    return few_shot_examples


def _extract_token_classification_examples(
    shuffled_train: Dataset, num_few_shots: int, dataset_config: "DatasetConfig"
) -> list[dict[str, t.Any]]:
    """Extract few-shot examples for token classification task group.

    Args:
        shuffled_train:
            The shuffled training dataset.
        num_few_shots:
            The number of few-shot examples to extract.
        dataset_config:
            The dataset configuration.

    Returns:
        A list of few-shot examples.
    """
    few_shot_examples: list[dict[str, t.Any]] = list()
    labels = it.cycle(
        [
            label.lower()
            for label in dataset_config.labels
            if label.lower().startswith("b-")
        ]
    )
    while len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0:
        label = next(labels)
        possible_examples = shuffled_train.filter(
            lambda x: label in [str(tag).lower() for tag in x["labels"]]
        )
        assert isinstance(possible_examples, Dataset), (
            f"Expected `possible_examples` to be a Dataset, but got "
            f"{type(possible_examples)} instead."
        )
        if len(possible_examples) == 0:
            continue
        example = possible_examples.select(range(1))[0]
        assert isinstance(example, dict), (
            f"Expected `example` to be a dict, but got {type(example)} instead."
        )
        few_shot_examples.append(example)
        shuffled_train = shuffled_train.filter(
            lambda x: x["tokens"] != example["tokens"]
        )
    return few_shot_examples


def _extract_question_answering_examples(
    shuffled_train: Dataset, num_few_shots: int, random_seed: int
) -> list[dict[str, t.Any]]:
    """Extract few-shot examples for question answering task group.

    Args:
        shuffled_train:
            The shuffled training dataset.
        num_few_shots:
            The number of few-shot examples to extract.
        random_seed:
            The random seed for reproducibility.

    Returns:
        A list of few-shot examples.

    Raises:
        InvalidBenchmark:
            If not enough short examples are found.
    """
    # Locate the maximum number of tokens that constitutes a short example
    for max_num_tokens in [512, 1024, 2048, 4096, 8192]:
        train_with_short_examples = shuffled_train.filter(
            lambda example: len(example["context"]) < max_num_tokens
        )
        num_short_examples = len(train_with_short_examples)
        if num_short_examples >= num_few_shots:
            break
    else:
        raise InvalidBenchmark(
            "Could not find enough short examples for few-shot learning."
        )

    shuffled_train = train_with_short_examples.shuffle(seed=random_seed)
    few_shot_examples: list[dict[str, t.Any]] = list()
    while len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0:
        example = shuffled_train.select(range(1))[0]
        assert isinstance(example, dict), (
            f"Expected `example` to be a dict, but got {type(example)} instead."
        )
        few_shot_examples.append(example)
        shuffled_train = shuffled_train.filter(
            lambda x: x["context"] != example["context"]
        )
    return few_shot_examples


def extract_few_shot_examples(
    dataset: "DatasetDict",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
    itr_idx: int,
) -> c.Sequence[dict[str, t.Any]]:
    """Extract few-shot examples from a dataset.

    This will always extract the examples from the training split.

    We ensure that the few-shot examples are unique by picking them one at a time.

    Args:
        dataset:
            The dataset to extract the few-shot examples from.
        dataset_config:
            The dataset configuration.
        benchmark_config:
            The benchmark configuration.
        itr_idx:
            The index of the dataset in the iterator.

    Returns:
        The few-shot examples.
    """
    if "train" not in dataset:
        log_once(
            "There is no training split in the dataset, so we cannot extract any "
            "few-shot examples, even though you requested few-shot evaluation (it's "
            "the default). We will therefore evaluate the model zero-shot.",
            level=logging.DEBUG,
        )
        return list()

    if dataset_config.task.requires_zero_shot and benchmark_config.few_shot:
        msg = (
            "This task only allows zero-shot evaluation, so even though you have "
            "requested few-shot evaluation "
        )
        if benchmark_config.run_with_cli:
            msg += "(by not setting the --zero-shot flag), "
        else:
            msg += "(by setting the default `few_shot=True` argument), "
        msg += "we will run the evaluation in zero-shot mode."
        benchmark_config.few_shot = False
        log_once(msg, level=logging.DEBUG)
        return []

    random_seed = 4242 + itr_idx
    num_few_shots = dataset_config.num_few_shot_examples
    shuffled_train = dataset["train"].shuffle(seed=random_seed)
    assert isinstance(shuffled_train, Dataset), (
        f"Expected `shuffled_train` to be a Dataset, but got {type(shuffled_train)} "
        "instead."
    )

    few_shot_examples: list[dict[str, t.Any]]
    match dataset_config.task.task_group:
        case (
            TaskGroup.SEQUENCE_CLASSIFICATION | TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
        ):
            few_shot_examples = _extract_classification_examples(
                shuffled_train=shuffled_train,
                num_few_shots=num_few_shots,
                dataset_config=dataset_config,
                random_seed=random_seed,
            )
        case TaskGroup.TEXT_TO_TEXT:
            few_shot_examples = _extract_text_to_text_examples(
                shuffled_train=shuffled_train, num_few_shots=num_few_shots
            )
        case TaskGroup.TOKEN_CLASSIFICATION:
            few_shot_examples = _extract_token_classification_examples(
                shuffled_train=shuffled_train,
                num_few_shots=num_few_shots,
                dataset_config=dataset_config,
            )
        case TaskGroup.QUESTION_ANSWERING:
            few_shot_examples = _extract_question_answering_examples(
                shuffled_train=shuffled_train,
                num_few_shots=num_few_shots,
                random_seed=random_seed,
            )
        case _:
            raise NotImplementedError(
                f"Unsupported task group: {dataset_config.task.task_group}."
            )

    random.seed(random_seed)
    random.shuffle(few_shot_examples)
    return few_shot_examples


def _get_sections_builder(
    task_group: TaskGroup, dataset_config: "DatasetConfig", use_bits_per_character: bool
) -> c.Callable[
    [list[dict[str, t.Any]], dict[str, t.Any], c.Callable],
    tuple[list[tuple[str, str]], list[tuple[str, str]]],
]:
    """Get a function that builds few-shot and new sections for a task group.

    Args:
        task_group:
            The task group to build sections for.
        dataset_config:
            The dataset configuration.
        use_bits_per_character:
            Whether to use bits-per-character scoring.

    Returns:
        A function that takes few_shot_examples, examples, and create_prompt,
        and returns (few_shot_sections, new_sections).
    """
    if task_group == TaskGroup.SEQUENCE_CLASSIFICATION:
        return lambda few_shot_examples, examples, create_prompt: (
            _build_sequence_classification_sections(
                few_shot_examples=few_shot_examples,
                examples=examples,
                create_prompt=create_prompt,
                dataset_config=dataset_config,
            )
        )
    elif task_group == TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION:
        return lambda few_shot_examples, examples, create_prompt: _build_mcq_sections(
            few_shot_examples=few_shot_examples,
            examples=examples,
            create_prompt=create_prompt,
            dataset_config=dataset_config,
            use_bits_per_character=use_bits_per_character,
        )
    elif task_group == TaskGroup.TEXT_TO_TEXT:
        return lambda few_shot_examples, examples, create_prompt: (
            _build_text_to_text_sections(
                few_shot_examples=few_shot_examples,
                examples=examples,
                create_prompt=create_prompt,
            )
        )
    elif task_group == TaskGroup.TOKEN_CLASSIFICATION:
        return lambda few_shot_examples, examples, create_prompt: (
            _build_token_classification_sections(
                few_shot_examples=few_shot_examples,
                examples=examples,
                create_prompt=create_prompt,
                dataset_config=dataset_config,
            )
        )
    elif task_group == TaskGroup.QUESTION_ANSWERING:
        return lambda few_shot_examples, examples, create_prompt: (
            _build_question_answering_sections(
                few_shot_examples=few_shot_examples,
                examples=examples,
                create_prompt=create_prompt,
            )
        )
    else:
        raise NotImplementedError(f"Unsupported task group: {task_group}.")


def _build_sequence_classification_sections(
    few_shot_examples: list[dict[str, t.Any]],
    examples: dict[str, t.Any],
    create_prompt: c.Callable,
    dataset_config: "DatasetConfig",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Build sections for sequence classification tasks.

    Args:
        few_shot_examples:
            The few-shot examples.
        examples:
            The examples to evaluate.
        create_prompt:
            The function to create prompts.
        dataset_config:
            The dataset configuration.

    Returns:
        A tuple of (few-shot sections, new sections).
    """
    labels_str = dataset_config.get_labels_str()
    few_shot_sections = [
        create_prompt(
            text=example["text"].replace("\n", " ").strip(),
            label=str(example["label"]).replace("\n", " ").strip(),
            labels_str=labels_str,
        )
        for example in few_shot_examples
    ]
    new_sections = [
        create_prompt(
            text=text.replace("\n", " ").strip(), label="", labels_str=labels_str
        )
        for text in examples["text"]
    ]
    return few_shot_sections, new_sections


def _build_mcq_sections(
    few_shot_examples: list[dict[str, t.Any]],
    examples: dict[str, t.Any],
    create_prompt: c.Callable,
    dataset_config: "DatasetConfig",
    use_bits_per_character: bool,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Build sections for multiple choice classification tasks.

    Args:
        few_shot_examples:
            The few-shot examples.
        examples:
            The examples to evaluate.
        create_prompt:
            The function to create prompts.
        dataset_config:
            The dataset configuration.
        use_bits_per_character:
            Whether to use bits-per-character scoring.

    Returns:
        A tuple of (few-shot sections, new sections).
    """
    if use_bits_per_character:
        few_shot_sections = [
            create_prompt(
                text=example["bare_input"].replace("\n", " ").strip(),
                label=letter_to_choice_text(
                    letter=str(example["label"]).strip().lower(),
                    raw_choices=example["raw_choices"],
                )
                .replace("\n", " ")
                .strip(),
            )
            for example in few_shot_examples
        ]
        new_sections = [
            create_prompt(
                text=examples["bare_input"][i].replace("\n", " ").strip(), label=""
            )
            for i in range(len(examples["text"]))
        ]
    else:
        few_shot_sections = [
            create_prompt(
                text=example["text"].replace("\n", " ").strip(),
                label=str(example["label"]).replace("\n", " ").strip(),
                labels_str=dataset_config.get_labels_str(
                    labels=extract_multiple_choice_labels(
                        prompt=example["text"], candidate_labels=dataset_config.labels
                    )
                ),
            )
            for example in few_shot_examples
        ]
        new_sections = [
            create_prompt(
                text=text.replace("\n", " ").strip(),
                label="",
                labels_str=dataset_config.get_labels_str(
                    labels=extract_multiple_choice_labels(
                        prompt=text, candidate_labels=dataset_config.labels
                    )
                ),
            )
            for text in examples["text"]
        ]
    return few_shot_sections, new_sections


def _build_text_to_text_sections(
    few_shot_examples: list[dict[str, t.Any]],
    examples: dict[str, t.Any],
    create_prompt: c.Callable,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Build sections for text-to-text tasks.

    Args:
        few_shot_examples:
            The few-shot examples.
        examples:
            The examples to evaluate.
        create_prompt:
            The function to create prompts.

    Returns:
        A tuple of (few-shot sections, new sections).
    """
    few_shot_sections = [
        create_prompt(
            text=re.sub(r"\n{2,}", "\n", example["text"]).strip(),
            target_text=re.sub(r"\n{2,}", "\n", example["target_text"]).strip(),
        )
        for example in few_shot_examples
    ]
    new_sections = [
        create_prompt(text=re.sub(r"\n{2,}", "\n", text).strip(), target_text="")
        for text in examples["text"]
    ]
    return few_shot_sections, new_sections


def _build_token_classification_sections(
    few_shot_examples: list[dict[str, t.Any]],
    examples: dict[str, t.Any],
    create_prompt: c.Callable,
    dataset_config: "DatasetConfig",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Build sections for token classification tasks.

    Args:
        few_shot_examples:
            The few-shot examples.
        examples:
            The examples to evaluate.
        create_prompt:
            The function to create prompts.
        dataset_config:
            The dataset configuration.

    Returns:
        A tuple of (few-shot sections, new sections).
    """
    labels_str = dataset_config.get_labels_str()
    few_shot_sections = [
        create_prompt(
            text=" ".join(example["tokens"]).replace("\n", " ").strip(),
            label=serialise_ner_tags(
                tokens=example["tokens"],
                labels=example["labels"],
                prompt_label_mapping=dataset_config.prompt_label_mapping,
            ),
            labels_str=labels_str,
        )
        for example in few_shot_examples
    ]
    new_sections = [
        create_prompt(
            text=" ".join(tokens).replace("\n", " ").strip(),
            label="",
            labels_str=labels_str,
        )
        for tokens in examples["tokens"]
    ]
    return few_shot_sections, new_sections


def _build_bpc_columns(
    task_group: TaskGroup,
    full_prompts: list[str],
    answers: list[str] | None,
    tokeniser: "PreTrainedTokenizer",
    strip_bpc_prompt: bool,
    num_examples: int,
) -> dict[str, list[t.Any]]:
    """Build BPC-related columns for examples.

    Args:
        task_group:
            The task group for determining answer character counting.
        full_prompts:
            List of full prompts.
        answers:
            List of answers, or None if no scoreable answer.
        tokeniser:
            Tokeniser for encoding.
        strip_bpc_prompt:
            Whether to strip prompts when building BPC prompts.
        num_examples:
            Number of examples.

    Returns:
        Dictionary with bpc_prompt, bpc_answer_start, bpc_answer_text,
        and bpc_answer_char_count columns.
    """

    def build_bpc_prompt(prompt: str, answer: str) -> tuple[str, int]:
        """Join a prompt and gold answer.

        Returns:
            A tuple of (full prompt, answer-start token index).
        """
        if strip_bpc_prompt:
            prefix = prompt.rstrip()
            full_prompt = f"{prefix} {answer}"
        else:
            prefix = prompt
            full_prompt = f"{prefix}{answer}"
        answer_start = len(tokeniser.encode(prefix, add_special_tokens=False))
        return full_prompt, answer_start

    if answers is None:
        return {
            "bpc_prompt": list(full_prompts),
            "bpc_answer_start": [0] * num_examples,
            "bpc_answer_text": [""] * num_examples,
            "bpc_answer_char_count": [0] * num_examples,
        }

    # Compute answer character counts
    if task_group == TaskGroup.TOKEN_CLASSIFICATION:
        answer_char_counts = [
            serialised_ner_content_length(answer) or len(answer) for answer in answers
        ]
    else:
        answer_char_counts = [len(answer) for answer in answers]

    # Build BPC prompts and answer starts
    bpc_prompts: list[str] = []
    bpc_answer_starts: list[int] = []
    for full_prompt, answer in zip(full_prompts, answers):
        bpc_prompt, answer_start = build_bpc_prompt(full_prompt, answer)
        bpc_prompts.append(bpc_prompt)
        bpc_answer_starts.append(answer_start)

    return {
        "bpc_prompt": bpc_prompts,
        "bpc_answer_start": bpc_answer_starts,
        "bpc_answer_text": answers,
        "bpc_answer_char_count": answer_char_counts,
    }


def _extract_bpc_answers(
    task_group: TaskGroup,
    examples: dict[str, t.Any],
    dataset_config: "DatasetConfig",
    num_examples: int,
) -> list[str] | None:
    """Extract BPC answers for each example based on task group.

    Args:
        task_group:
            The task group determining how to extract answers.
        examples:
            The examples dictionary containing the data.
        dataset_config:
            The dataset configuration.
        num_examples:
            Number of examples to process.

    Returns:
        List of answer strings, or None if task group has no scoreable answer.
    """
    if task_group == TaskGroup.SEQUENCE_CLASSIFICATION:
        return [
            dataset_config.prompt_label_mapping.get(
                examples["label"][i], examples["label"][i]
            )
            for i in range(num_examples)
        ]
    elif (
        task_group == TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
        and "raw_choices" in examples
    ):
        return [
            letter_to_choice_text(
                letter=str(examples["label"][i]).strip().lower(),
                raw_choices=examples["raw_choices"][i],
            )
            for i in range(num_examples)
        ]
    elif task_group == TaskGroup.TEXT_TO_TEXT and "target_text" in examples:
        return [str(examples["target_text"][i]) for i in range(num_examples)]
    elif task_group == TaskGroup.QUESTION_ANSWERING and "answers" in examples:
        return [examples["answers"][i]["text"][0] for i in range(num_examples)]
    elif (
        task_group == TaskGroup.TOKEN_CLASSIFICATION
        and "tokens" in examples
        and "labels" in examples
    ):
        return [
            serialise_ner_tags(
                tokens=examples["tokens"][i],
                labels=examples["labels"][i],
                prompt_label_mapping=dataset_config.prompt_label_mapping,
            )
            for i in range(num_examples)
        ]
    else:
        return None


def _build_question_answering_sections(
    few_shot_examples: list[dict[str, t.Any]],
    examples: dict[str, t.Any],
    create_prompt: c.Callable,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Build sections for question answering tasks.

    Args:
        few_shot_examples:
            The few-shot examples.
        examples:
            The examples to evaluate.
        create_prompt:
            The function to create prompts.

    Returns:
        A tuple of (few-shot sections, new sections).
    """
    few_shot_sections = [
        create_prompt(
            text=example["context"].replace("\n", " ").strip(),
            question=example["question"].replace("\n", " ").strip(),
            label=example["answers"]["text"][0].replace("\n", " "),
        )
        for example in few_shot_examples
    ]
    new_sections = [
        create_prompt(
            text=context.replace("\n", " ").strip(),
            question=question.replace("\n", " ").strip(),
            label="",
        )
        for context, question in zip(examples["context"], examples["question"])
    ]
    return few_shot_sections, new_sections


def _create_prompt_creator(
    dataset_config: "DatasetConfig", generative_type: GenerativeType | None
) -> c.Callable[..., tuple[str, str]]:
    """Create a function that builds prompts from keyword arguments.

    Args:
        dataset_config:
            The dataset configuration.
        generative_type:
            The generative type of the model.

    Returns:
        A function that builds prompts from keyword arguments.
    """

    def create_prompt(**kwargs: str) -> tuple[str, str]:
        """Create a prompt from the given keyword arguments.

        Args:
            kwargs:
                The keyword arguments to use in the prompt.

        Returns:
            A pair (prompt, label), where "label" is an empty string if the model is
            not instruction tuned (as in this case it is included in the prompt).
        """
        label_key = "label" if "label" in kwargs else "target_text"
        label = kwargs.pop(label_key)
        assert label is not None, (
            f"Found a None label for the prompt: {kwargs}. This should not happen."
        )
        label_mapping = dataset_config.prompt_label_mapping
        label = label_mapping.get(label, label)
        if generative_type in {
            GenerativeType.INSTRUCTION_TUNED,
            GenerativeType.REASONING,
        }:
            prompt = dataset_config.instruction_prompt.format(**kwargs)
            return prompt, label
        else:
            kwargs[label_key] = label
            return dataset_config.prompt_template.format(**kwargs), ""

    return create_prompt


def _add_bare_inputs_for_bpc(
    examples: dict[str, t.Any], few_shot_examples: list[dict[str, t.Any]]
) -> None:
    """Add bare_input and raw_choices for BPC scoring on MCQ tasks."""
    bare_inputs: list[str] = []
    raw_choices_list: list[list[str]] = []
    for text in examples["text"]:
        bare_input, raw_choices = parse_bare_question_and_choices(text)
        bare_inputs.append(bare_input)
        raw_choices_list.append(raw_choices)
    examples["bare_input"] = bare_inputs
    examples["raw_choices"] = raw_choices_list
    for fs_example in few_shot_examples:
        if "raw_choices" not in fs_example:
            fs_bare, fs_choices = parse_bare_question_and_choices(fs_example["text"])
            fs_example["bare_input"] = fs_bare
            fs_example["raw_choices"] = fs_choices


def _build_instruction_tuned_outputs(
    few_shot_sections: list[tuple[str, str]],
    new_sections: list[tuple[str, str]],
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    generative_type: GenerativeType | None,
    tokeniser: "PreTrainedTokenizer",
    always_populate_text_field: bool,
) -> dict[str, t.Any]:
    """Build outputs for instruction-tuned/reasoning models.

    Args:
        few_shot_sections:
            The few-shot sections.
        new_sections:
            The new sections.
        model_config:
            The model configuration.
        dataset_config:
            The dataset configuration.
        generative_type:
            The generative type of the model.
        tokeniser:
            The tokeniser.
        always_populate_text_field:
            Whether to always populate the text field.

    Returns:
        A dictionary of outputs.
    """
    few_shot_messages = [
        dict(role=role, content=content)
        for prompt, label in few_shot_sections
        for role, content in [("user", prompt), ("assistant", label)]
    ]
    messages_list = [
        few_shot_messages + [dict(role="user", content=prompt)]
        for prompt, _ in new_sections
    ]

    outputs: dict[str, t.Any] = {}
    if not always_populate_text_field:
        outputs["messages"] = messages_list
    else:
        chat_template = _select_chat_template(tokeniser, dataset_config, model_config)
        chat_template_kwargs = _build_chat_template_kwargs(model_config)

        texts = [
            apply_chat_template(
                conversation=messages,
                tokeniser=tokeniser,
                tokenise=False,
                add_generation_prompt=True,
                enable_thinking=(generative_type == GenerativeType.REASONING),
                chat_template=chat_template,
                **chat_template_kwargs,
            )
            for messages in messages_list
        ]
        outputs["text"] = texts
        outputs["messages"] = messages_list

    return outputs


def _select_chat_template(
    tokeniser: "PreTrainedTokenizer",
    dataset_config: "DatasetConfig",
    model_config: "ModelConfig",
) -> str | None:
    """Select chat template matching dataset language if available.

    Args:
        tokeniser:
            The tokeniser.
        dataset_config:
            The dataset configuration.
        model_config:
            The model configuration.

    Returns:
        The chat template, or None if no matching template is found.
    """
    if not (
        hasattr(tokeniser, "chat_template")
        and isinstance(tokeniser.chat_template, dict)
    ):
        return None

    language_codes = [language.code for language in dataset_config.languages]
    for name, candidate_template in tokeniser.chat_template.items():
        if name.lower() in language_codes:
            log_once(
                f"Using the {name!r} chat template for the tokeniser for "
                f"model {model_config.model_id!r}.",
                level=logging.DEBUG,
            )
            return candidate_template
    return None


def _build_chat_template_kwargs(model_config: "ModelConfig") -> dict[str, t.Any]:
    """Build chat template kwargs for reasoning effort.

    Args:
        model_config:
            The model configuration.

    Returns:
        A dictionary of chat template kwargs.
    """
    if model_config.param in {"low", "medium", "high"}:
        log_once(f"Set reasoning mode to {model_config.param!r}.", level=logging.DEBUG)
        return {"reasoning_effort": model_config.param}
    return {}


def _build_standard_outputs(
    few_shot_sections: list[tuple[str, str]],
    new_sections: list[tuple[str, str]],
    dataset_config: "DatasetConfig",
) -> dict[str, t.Any]:
    """Build outputs for non-instruction-tuned models.

    Args:
        few_shot_sections:
            The few-shot sections.
        new_sections:
            The new sections.
        dataset_config:
            The dataset configuration.

    Returns:
        A dictionary of outputs.
    """
    prompt_prefix = ""
    if dataset_config.prompt_prefix:
        labels_str = dataset_config.get_labels_str()
        prompt_prefix = (
            dataset_config.prompt_prefix.format(labels_str=labels_str) + "\n\n"
        )

    few_shot_prompt = "\n\n".join([prompt for prompt, _ in few_shot_sections])
    if few_shot_prompt:
        few_shot_prompt += "\n\n"

    return {
        "text": [
            prompt_prefix + few_shot_prompt + new_prompt
            for new_prompt, _ in new_sections
        ]
    }


def _build_bpc_outputs(
    dataset_config: "DatasetConfig",
    examples: dict[str, t.Any],
    tokeniser: "PreTrainedTokenizer",
    num_examples: int,
) -> dict[str, t.Any]:
    """Build BPC prompt and answer_start columns.

    Args:
        dataset_config:
            The dataset configuration.
        examples:
            The examples to evaluate.
        tokeniser:
            The tokeniser.
        num_examples:
            The number of examples.

    Returns:
        A dictionary of BPC outputs.
    """
    labels_for_spacing = list(dataset_config.prompt_label_mapping.values()) or [
        "negative",
        "positive",
    ]
    strip_bpc_prompt = should_prompts_be_stripped(
        labels_to_be_generated=labels_for_spacing, tokeniser=tokeniser
    )

    full_prompts: list[str] = [str(text) for text in examples["text"]]

    answers = _extract_bpc_answers(
        task_group=dataset_config.task.task_group,
        examples=examples,
        dataset_config=dataset_config,
        num_examples=num_examples,
    )

    bpc_data = _build_bpc_columns(
        task_group=dataset_config.task.task_group,
        full_prompts=full_prompts,
        answers=answers,
        tokeniser=tokeniser,
        strip_bpc_prompt=strip_bpc_prompt,
        num_examples=num_examples,
    )
    return bpc_data


def apply_prompt(
    examples: dict[str, t.Any],
    few_shot_examples: c.Sequence[dict[str, t.Any]],
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    generative_type: GenerativeType | None,
    always_populate_text_field: bool,
    tokeniser: "PreTrainedTokenizer | None",
    use_bits_per_character: bool = False,
) -> dict[str, t.Any]:
    """Apply prompt template to an example, potentially with few-shot examples.

    Args:
        examples:
            The examples to apply the few-shot examples to.
        few_shot_examples:
            The few-shot examples to apply.
        model_config:
            The model configuration.
        dataset_config:
            The dataset configuration.
        generative_type:
            The generative type of the model.
        always_populate_text_field:
            Whether to always populate the 'text' field in the examples, as opposed to
            the 'messages' field.
        tokeniser:
            The tokeniser to use for the model. If None, the tokeniser is not used.
        use_bits_per_character:
            Whether to use bits-per-character (BPC) scoring. For multiple-choice tasks,
            treats benchmark as text-to-text with bare question → full answer text.
            Defaults to False.

    Returns:
        The example with the few-shot examples applied.

    Raises:
        ValueError:
            If the `tokeniser` argument is not provided when the model is instruction
            tuned and when we are not just returning the raw messages.
    """
    # Sanity check
    is_instruction_tuned = generative_type in {
        GenerativeType.INSTRUCTION_TUNED,
        GenerativeType.REASONING,
    }
    if is_instruction_tuned and always_populate_text_field and tokeniser is None:
        raise ValueError(
            "The `tokeniser` argument must be provided when the model is instruction "
            "tuned and when we are not just returning the raw messages."
        )

    create_prompt = _create_prompt_creator(dataset_config, generative_type)

    # Add bare inputs for BPC on MCQ tasks
    if (
        use_bits_per_character
        and dataset_config.task.task_group == TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
        and "raw_choices" not in examples
    ):
        _add_bare_inputs_for_bpc(examples, list(few_shot_examples))

    sections_builder = _get_sections_builder(
        task_group=dataset_config.task.task_group,
        dataset_config=dataset_config,
        use_bits_per_character=use_bits_per_character,
    )
    few_shot_sections, new_sections = sections_builder(
        list(few_shot_examples), examples, create_prompt
    )

    # Build outputs based on model type
    if is_instruction_tuned:
        assert tokeniser is not None
        outputs = _build_instruction_tuned_outputs(
            few_shot_sections=few_shot_sections,
            new_sections=new_sections,
            model_config=model_config,
            dataset_config=dataset_config,
            generative_type=generative_type,
            tokeniser=tokeniser,
            always_populate_text_field=always_populate_text_field,
        )
        examples.update(outputs)
    else:
        outputs = _build_standard_outputs(
            few_shot_sections=few_shot_sections,
            new_sections=new_sections,
            dataset_config=dataset_config,
        )
        examples.update(outputs)

    # Always add the final prompts without few-shot examples, too, for analysis
    examples["prompt"] = [new_prompt for new_prompt, _ in new_sections]

    # Create bpc_prompt column for BPC scoring when requested
    if use_bits_per_character:
        assert tokeniser is not None, (
            "tokeniser must be provided when use_bits_per_character=True"
        )
        num_examples = len(new_sections)
        bpc_data = _build_bpc_outputs(
            dataset_config=dataset_config,
            examples=examples,
            tokeniser=tokeniser,
            num_examples=num_examples,
        )
        examples.update(bpc_data)

    return examples


def raise_if_wrong_params(
    model_config: "ModelConfig", allowed_params: c.Mapping[re.Pattern[str], list[str]]
) -> None:
    """Raise an error if the model configuration has invalid parameters.

    Args:
        model_config:
            The model configuration.
        allowed_params:
            The allowed parameters for the model, being a dictionary mapping a regex
            pattern matching the model ID to a list of allowed parameters for those
            models.

    Raises:
        InvalidModel:
            If the model configuration has invalid parameters.
    """
    # Do nothing if there are no parameters to check
    if model_config.param is None:
        return

    # Make list of all allowed parameters for the model
    all_allowed_params: set[str] = set()
    for model_regex, allowed_params_list in allowed_params.items():
        if re.fullmatch(pattern=model_regex, string=model_config.model_id):
            all_allowed_params.update(allowed_params_list)

    # Raise error if the parameter is not allowed
    if model_config.param not in all_allowed_params:
        msg = (
            f"Invalid parameter {model_config.param!r} for model "
            f"{model_config.model_id!r}."
        )
        if all_allowed_params:
            msg += f" Allowed parameters are: {', '.join(all_allowed_params)}."
        else:
            msg += " No parameters are allowed."
        raise InvalidModel(msg)
