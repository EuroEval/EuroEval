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
from .task_group_utils.token_classification import serialise_ner_tags
from .tokenisation_utils import apply_chat_template, should_prompts_be_stripped

if t.TYPE_CHECKING:
    from datasets import DatasetDict
    from transformers.tokenization_utils import PreTrainedTokenizer

    from .data_models import BenchmarkConfig, DatasetConfig, ModelConfig


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

    Raises:
        InvalidBenchmark:
            If there are not enough short examples for few-shot learning.
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
    few_shot_examples: list[dict[str, t.Any]] = list()
    shuffled_train = dataset["train"].shuffle(seed=random_seed)
    assert isinstance(shuffled_train, Dataset), (
        f"Expected `shuffled_train` to be a Dataset, but got {type(shuffled_train)} "
        "instead."
    )

    match dataset_config.task.task_group:
        case (
            TaskGroup.SEQUENCE_CLASSIFICATION | TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
        ):
            # Locate the maximum number of tokens that constitutes a short example
            for max_num_tokens in [512, 1024, 2048, 4096, 8192]:
                train_with_short_examples = dataset["train"].filter(
                    lambda example: len(example["text"]) < max_num_tokens
                )
                num_short_examples = len(train_with_short_examples)
                if num_short_examples >= num_few_shots:
                    break
            else:
                raise InvalidBenchmark(
                    "Could not find enough short examples for few-shot learning."
                )

            if dataset_config.labels:
                labels = it.cycle(dataset_config.labels)
                labels_with_no_samples: set[str] = set()
                while (
                    len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0
                ):
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
                        f"Expected `example` to be a dict, but got "
                        f"{type(example)} instead."
                    )
                    few_shot_examples.append(example)
                    shuffled_train = shuffled_train.filter(
                        lambda x: x["text"] != example["text"]
                    )
            else:
                # No labels defined (e.g. community datasets with variable number of
                # choices) — fall back to random sampling.
                while (
                    len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0
                ):
                    example = shuffled_train.select(range(1))[0]
                    assert isinstance(example, dict), (
                        f"Expected `example` to be a dict, but got "
                        f"{type(example)} instead."
                    )
                    few_shot_examples.append(example)
                    shuffled_train = shuffled_train.filter(
                        lambda x: x["text"] != example["text"]
                    )

        case TaskGroup.TEXT_TO_TEXT:
            while len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0:
                example = shuffled_train.select(range(1))[0]
                assert isinstance(example, dict), (
                    f"Expected `example` to be a dict, but got {type(example)} instead."
                )
                few_shot_examples.append(example)
                shuffled_train = shuffled_train.filter(
                    lambda x: x["text"] != example["text"]
                )

        case TaskGroup.TOKEN_CLASSIFICATION:
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

        case TaskGroup.QUESTION_ANSWERING:
            # Locate the maximum number of tokens that constitutes a short example
            for max_num_tokens in [512, 1024, 2048, 4096, 8192]:
                train_with_short_examples = dataset["train"].filter(
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
            assert isinstance(shuffled_train, Dataset), (
                f"Expected `shuffled_train` to be a Dataset, but got "
                f"{type(shuffled_train)} instead."
            )
            while len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0:
                example = shuffled_train.select(range(1))[0]
                assert isinstance(example, dict), (
                    f"Expected `example` to be a dict, but got {type(example)} instead."
                )
                few_shot_examples.append(example)
                shuffled_train = shuffled_train.filter(
                    lambda x: x["context"] != example["context"]
                )

        case _:
            raise NotImplementedError(
                f"Unsupported task group: {dataset_config.task.task_group}."
            )

    random.seed(random_seed)
    random.shuffle(few_shot_examples)
    return few_shot_examples


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
    if (
        generative_type in {GenerativeType.INSTRUCTION_TUNED, GenerativeType.REASONING}
        and always_populate_text_field
        and tokeniser is None
    ):
        raise ValueError(
            "The `tokeniser` argument must be provided when the model is instruction "
            "tuned and when we are not just returning the raw messages."
        )

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

    # Multiple-choice datasets currently expose only a pre-formatted `text` column, so
    # when BPC (cloze) scoring is requested we recover the bare question and the choice
    # texts it needs and attach them to both the batch and the few-shot examples.
    if (
        use_bits_per_character
        and dataset_config.task.task_group == TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
        and "raw_choices" not in examples
    ):
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
                fs_bare, fs_choices = parse_bare_question_and_choices(
                    fs_example["text"]
                )
                fs_example["bare_input"] = fs_bare
                fs_example["raw_choices"] = fs_choices

    match dataset_config.task.task_group:
        case TaskGroup.SEQUENCE_CLASSIFICATION:
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
                    text=text.replace("\n", " ").strip(),
                    label="",
                    labels_str=labels_str,
                )
                for text in examples["text"]
            ]

        case TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION:
            if use_bits_per_character:
                # BPC (cloze) scoring: present the bare question and score the full
                # answer text, which is appended when building `bpc_prompt` below. This
                # mirrors the sequence-classification BPC path: few-shot examples show
                # the complete question → answer, while the sections to be scored leave
                # the answer empty. The answer fills the template's `{label}` slot (the
                # MCQ template has no `{target_text}` placeholder).
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
                        text=examples["bare_input"][i].replace("\n", " ").strip(),
                        label="",
                    )
                    for i in range(len(examples["text"]))
                ]
            else:
                # Standard scoring: enumerated choices
                few_shot_sections = [
                    create_prompt(
                        text=example["text"].replace("\n", " ").strip(),
                        label=str(example["label"]).replace("\n", " ").strip(),
                        labels_str=dataset_config.get_labels_str(
                            labels=extract_multiple_choice_labels(
                                prompt=example["text"],
                                candidate_labels=dataset_config.labels,
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

        case TaskGroup.TEXT_TO_TEXT:
            few_shot_sections = [
                create_prompt(
                    text=example["text"].replace("\n", " ").strip(),
                    target_text=example["target_text"].replace("\n", " ").strip(),
                )
                for example in few_shot_examples
            ]
            new_sections = [
                create_prompt(text=text.replace("\n", " ").strip(), target_text="")
                for text in examples["text"]
            ]

        case TaskGroup.TOKEN_CLASSIFICATION:
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

        case TaskGroup.QUESTION_ANSWERING:
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

        case _:
            raise NotImplementedError(
                f"Unsupported task group: {dataset_config.task.task_group}."
            )

    if generative_type in {GenerativeType.INSTRUCTION_TUNED, GenerativeType.REASONING}:
        few_shot_messages = [
            dict(role=role, content=content)
            for prompt, label in few_shot_sections
            for role, content in [("user", prompt), ("assistant", label)]
        ]

        messages_list = [
            few_shot_messages + [dict(role="user", content=prompt)]
            for prompt, _ in new_sections
        ]

        if not always_populate_text_field:
            examples["messages"] = messages_list
        else:
            assert tokeniser is not None

            # Pick the chat template that matches the language of the dataset, if such a
            # template exists
            chat_template: str | None = None
            if hasattr(tokeniser, "chat_template") and isinstance(
                tokeniser.chat_template, dict
            ):
                language_codes = [
                    language.code for language in dataset_config.languages
                ]
                for name, candidate_template in tokeniser.chat_template.items():
                    if name.lower() in language_codes:
                        chat_template = candidate_template
                        log_once(
                            f"Using the {name!r} chat template for the tokeniser for "
                            f"model {model_config.model_id!r}.",
                            level=logging.DEBUG,
                        )
                        break

            # Custom chat template kwargs
            chat_template_kwargs: dict[str, t.Any] = dict()
            if model_config.param in {"low", "medium", "high"}:
                chat_template_kwargs["reasoning_effort"] = model_config.param
                log_once(
                    f"Set reasoning mode to {model_config.param!r}.",
                    level=logging.DEBUG,
                )

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

            examples["text"] = texts

    else:
        prompt_prefix = ""
        if dataset_config.prompt_prefix:
            labels_str = dataset_config.get_labels_str()
            prompt_prefix = (
                dataset_config.prompt_prefix.format(labels_str=labels_str) + "\n\n"
            )

        few_shot_prompt = "\n\n".join([prompt for prompt, _ in few_shot_sections])
        if few_shot_prompt:
            few_shot_prompt += "\n\n"

        examples["text"] = [
            prompt_prefix + few_shot_prompt + new_prompt
            for new_prompt, _ in new_sections
        ]

    # Always add the final prompts without few-shot examples, too, for analysis
    examples["prompt"] = [new_prompt for new_prompt, _ in new_sections]

    # Create bpc_prompt column for BPC scoring when requested
    # bpc_prompt = prompt + answer text (for scoring with prompt_logprobs)
    # Also track bpc_answer_start: token index where answer begins in bpc_prompt
    if use_bits_per_character:
        assert tokeniser is not None, (
            "tokeniser must be provided when use_bits_per_character=True"
        )
        bpc_tokeniser = tokeniser

        # Decide, once for this tokeniser, how to join a prompt with its gold answer.
        # This mirrors the generation path (see `should_prompts_be_stripped`): if the
        # tokeniser merges a leading space into the following token, the prompt's
        # trailing whitespace must be stripped and a single space placed before the
        # answer so the answer's first token carries its natural leading space.
        # Otherwise the prompt keeps its trailing whitespace (emitted as its own token)
        # and the answer is appended directly. Either way the prefix tokenises stably,
        # so its token count is the exact answer-start index.
        labels_for_spacing = list(dataset_config.prompt_label_mapping.values()) or [
            "negative",
            "positive",
        ]
        strip_bpc_prompt = should_prompts_be_stripped(
            labels_to_be_generated=labels_for_spacing, tokeniser=bpc_tokeniser
        )

        def build_bpc_prompt(prompt: str, answer: str) -> tuple[str, int]:
            """Join a prompt and gold answer, returning the answer-start token index.

            Args:
                prompt:
                    The full prompt for the example (including any prompt prefix and
                    few-shot examples), ending with the answer prefix (e.g.
                    ``"Svar: "``).
                answer:
                    The gold answer text to be scored.

            Returns:
                A pair ``(bpc_prompt, answer_start_token_index)``.
            """
            if strip_bpc_prompt:
                prefix = prompt.rstrip()
                full_prompt = f"{prefix} {answer}"
            else:
                prefix = prompt
                full_prompt = f"{prefix}{answer}"
            answer_start = len(bpc_tokeniser.encode(prefix, add_special_tokens=False))
            return full_prompt, answer_start

        # Score the answer against the same full prompt the model is conditioned on
        # during generation (prompt prefix + few-shot examples + question), which is
        # already assembled in `examples["text"]`. Few-shot context is automatically
        # absent here when zero-shot evaluation is requested, since `few_shot_examples`
        # is then empty.
        full_prompts: list[str] = [str(text) for text in examples["text"]]
        bpc_prompts: list[str] = []
        bpc_answer_starts: list[int] = []
        match dataset_config.task.task_group:
            case TaskGroup.SEQUENCE_CLASSIFICATION:
                for i in range(len(new_sections)):
                    label = examples["label"][i]
                    answer = dataset_config.prompt_label_mapping.get(label, label)
                    bpc_prompt, answer_start = build_bpc_prompt(full_prompts[i], answer)
                    bpc_prompts.append(bpc_prompt)
                    bpc_answer_starts.append(answer_start)
            case TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION:
                if "raw_choices" in examples:
                    for i in range(len(new_sections)):
                        label = examples["label"][i]
                        raw_choice = examples["raw_choices"][i]
                        answer = letter_to_choice_text(
                            letter=str(label).strip().lower(), raw_choices=raw_choice
                        )
                        bpc_prompt, answer_start = build_bpc_prompt(
                            full_prompts[i], answer
                        )
                        bpc_prompts.append(bpc_prompt)
                        bpc_answer_starts.append(answer_start)
                else:
                    bpc_prompts = list(full_prompts)
                    bpc_answer_starts = [0] * len(new_sections)
            case TaskGroup.TEXT_TO_TEXT:
                if "target_text" in examples:
                    for i in range(len(new_sections)):
                        target = examples["target_text"][i]
                        bpc_prompt, answer_start = build_bpc_prompt(
                            full_prompts[i], target
                        )
                        bpc_prompts.append(bpc_prompt)
                        bpc_answer_starts.append(answer_start)
                else:
                    bpc_prompts = list(full_prompts)
                    bpc_answer_starts = [0] * len(new_sections)
            case TaskGroup.QUESTION_ANSWERING:
                if "answers" in examples:
                    for i in range(len(new_sections)):
                        answer_dct = examples["answers"][i]
                        answer = answer_dct["answers"]["text"][0]
                        bpc_prompt, answer_start = build_bpc_prompt(
                            full_prompts[i], answer
                        )
                        bpc_prompts.append(bpc_prompt)
                        bpc_answer_starts.append(answer_start)
                else:
                    bpc_prompts = list(full_prompts)
                    bpc_answer_starts = [0] * len(new_sections)
            case TaskGroup.TOKEN_CLASSIFICATION:
                if "tokens" in examples and "labels" in examples:
                    for i in range(len(new_sections)):
                        answer = serialise_ner_tags(
                            tokens=examples["tokens"][i],
                            labels=examples["labels"][i],
                            prompt_label_mapping=dataset_config.prompt_label_mapping,
                        )
                        bpc_prompt, answer_start = build_bpc_prompt(
                            full_prompts[i], answer
                        )
                        bpc_prompts.append(bpc_prompt)
                        bpc_answer_starts.append(answer_start)
                else:
                    bpc_prompts = list(full_prompts)
                    bpc_answer_starts = [0] * len(new_sections)
            case _:
                bpc_prompts = list(full_prompts)
                bpc_answer_starts = [0] * len(new_sections)

        examples["bpc_prompt"] = bpc_prompts
        examples["bpc_answer_start"] = bpc_answer_starts

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
