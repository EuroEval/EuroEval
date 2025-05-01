"""Utility functions related to generative models."""

import itertools as it
import json
import logging
import random
import typing as t

from .enums import TaskGroup
from .exceptions import InvalidBenchmark
from .utils import log_once

if t.TYPE_CHECKING:
    from datasets import DatasetDict
    from transformers.tokenization_utils import PreTrainedTokenizer

    from .data_models import DatasetConfig, ModelConfig

logger = logging.getLogger("euroeval")


def extract_few_shot_examples(
    dataset: "DatasetDict", dataset_config: "DatasetConfig", itr_idx: int
) -> list[dict[str, t.Any]]:
    """Extract few-shot examples from a dataset.

    This will always extract the examples from the training split.

    We ensure that the few-shot examples are unique by picking them one at a time.

    Args:
        dataset:
            The dataset to extract the few-shot examples from.
        dataset_config:
            The dataset configuration.
        itr_idx:
            The index of the dataset in the iterator.

    Returns:
        The few-shot examples.
    """
    random_seed = 4242 + itr_idx
    num_few_shots = dataset_config.num_few_shot_examples
    few_shot_examples: list[dict[str, t.Any]] = list()
    shuffled_train = dataset["train"].shuffle(seed=random_seed)

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
                if num_short_examples >= dataset_config.num_few_shot_examples:
                    break
            else:
                raise InvalidBenchmark(
                    "Could not find enough short examples for few-shot learning."
                )

            shuffled_train = train_with_short_examples.shuffle(seed=random_seed)
            labels = it.cycle(dataset_config.labels)
            while len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0:
                label = next(labels)
                possible_examples = shuffled_train.filter(
                    lambda x: x["label"].lower() == label.lower()
                )
                if len(possible_examples) == 0:
                    continue
                example = possible_examples.select(range(1))[0]
                few_shot_examples.append(example)
                shuffled_train = shuffled_train.filter(
                    lambda x: x["text"] != example["text"]
                )

        case TaskGroup.TEXT_TO_TEXT:
            while len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0:
                example = shuffled_train.select(range(1))[0]
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
                    lambda x: label in [tag.lower() for tag in x["labels"]]
                )
                if len(possible_examples) == 0:
                    continue
                example = possible_examples.select(range(1))[0]
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
                if num_short_examples >= dataset_config.num_few_shot_examples:
                    break
            else:
                raise InvalidBenchmark(
                    "Could not find enough short examples for few-shot learning."
                )

            shuffled_train = train_with_short_examples.shuffle(seed=random_seed)
            while len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0:
                example = shuffled_train.select(range(1))[0]
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
    few_shot_examples: list[dict[str, t.Any]],
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    instruction_model: bool,
    always_populate_text_field: bool,
    tokenizer: "PreTrainedTokenizer | None",
) -> dict[str, t.Any]:
    """Apply prompt template to an example, potentially with few-shot examples.

    Args:
        examples:
            The examples to apply the few-shot examples to.
        few_shot_examples:
            The few-shot examples to apply.
        dataset_config:
            The dataset configuration.
        instruction_model:
            Whether the model is instruction-tuned.
        always_populate_text_field:
            Whether to always populate the 'text' field in the examples, as opposed to
            the 'messages' field.
        tokenizer:
            The tokenizer to use for the model. If None, the tokenizer is not used.

    Returns:
        The example with the few-shot examples applied.
    """
    # Sanity check
    if instruction_model and always_populate_text_field and tokenizer is None:
        raise ValueError(
            "The `tokenizer` argument must be provided when the model is instruction "
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
        if instruction_model:
            prompt = dataset_config.instruction_prompt.format(**kwargs)
            return prompt, label
        else:
            kwargs[label_key] = label
            return dataset_config.prompt_template.format(**kwargs), ""

    match dataset_config.task.task_group:
        case (
            TaskGroup.SEQUENCE_CLASSIFICATION | TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
        ):
            few_shot_sections = [
                create_prompt(
                    text=example["text"].replace("\n", " ").strip(),
                    label=example["label"].replace("\n", " ").strip(),
                )
                for example in few_shot_examples
            ]
            new_sections = [
                create_prompt(text=text.replace("\n", " ").strip(), label="")
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

            def create_label(example: dict) -> str:
                prompt_labels = dataset_config.prompt_label_mapping.values()
                labels: dict[str, list[str]] = {
                    prompt_label: list() for prompt_label in prompt_labels
                }
                for token, label in zip(example["tokens"], example["labels"]):
                    label = label.lower()
                    if label == "o":
                        continue
                    prompt_label = dataset_config.prompt_label_mapping[label]
                    if label.startswith("b-"):
                        labels[prompt_label].append(token)
                    elif label.startswith("i-"):
                        labels[prompt_label][-1] += " " + token
                return json.dumps(labels, ensure_ascii=False)

            few_shot_sections = [
                create_prompt(
                    text=" ".join(example["tokens"]).replace("\n", " ").strip(),
                    label=create_label(example=example),
                )
                for example in few_shot_examples
            ]
            new_sections = [
                create_prompt(
                    text=" ".join(tokens).replace("\n", " ").strip(), label=""
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

    if instruction_model:
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
            assert tokenizer is not None

            # Pick the chat template that matches the language of the dataset, if such a
            # template exists
            chat_template: str | None = None
            if isinstance(tokenizer.chat_template, dict):
                language_codes = [
                    language.code for language in dataset_config.languages
                ]
                for name, candidate_template in tokenizer.chat_template.items():
                    if name.lower() in language_codes:
                        chat_template = candidate_template
                        log_once(
                            f"Using the {name!r} chat template for the tokenizer for "
                            f"model {model_config.model_id!r}.",
                            level=logging.DEBUG,
                        )
                        break

            texts = [
                tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=chat_template,
                )
                for messages in messages_list
            ]

            examples["text"] = texts

    else:
        prompt_prefix = ""
        if dataset_config.prompt_prefix:
            prompt_prefix = dataset_config.prompt_prefix + "\n\n"

        few_shot_prompt = "\n\n".join([prompt for prompt, _ in few_shot_sections])
        if few_shot_prompt:
            few_shot_prompt += "\n\n"

        examples["text"] = [
            prompt_prefix + few_shot_prompt + new_prompt
            for new_prompt, _ in new_sections
        ]

    return examples
