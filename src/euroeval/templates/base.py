"""Base classes for task-specific prompting across languages."""

from dataclasses import dataclass, field


@dataclass
class BasePromptConfig:
    """Base configuration for task-specific prompting across languages.

    Attributes:
        num_few_shot_examples:
            The number of examples to use when benchmarking the dataset using few-shot
            evaluation. For a classification task, these will be drawn evenly from
            each label.
        max_generated_tokens:
            The maximum number of tokens to generate when benchmarking the dataset
            using few-shot evaluation.
        labels (optional):
            The labels in the dataset.
        prompt_label_mapping (optional):
            A mapping from the labels to another phrase which is used as a substitute
            for the label in few-shot evaluation. Defaults to an empty dictionary. If
            set to "auto", the mapping will be set to a 1:1 mapping between the labels
            and themselves.

    """

    num_few_shot_examples: int
    max_generated_tokens: int
    labels: list[str] = field(default_factory=list)
    prompt_label_mapping: dict[str, str] | str = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post Initialisation setup of defaults."""
        if isinstance(self.prompt_label_mapping, str):
            if self.prompt_label_mapping == "auto":
                self.prompt_label_mapping = {label: label for label in self.labels}
            else:
                raise ValueError(
                    f"Invalid value '{self.prompt_label_mapping}' "
                    "for prompt_label_mapping. Must be 'auto' or a dictionary."
                )


@dataclass
class PromptConfig:
    """Configuration for task-specific prompting across languages.

    Defines the prompt templates needed for evaluating a specific task in a
    given language.

    Attributes:
        labels:
            The labels in the dataset. Defaults to an empty list.
        prompt_label_mapping:
            A mapping from the labels to another phrase which is used as a substitute
            for the label in few-shot evaluation. Defaults to an empty dictionary.
        num_few_shot_examples:
            The number of examples to use when benchmarking the dataset using few-shot
            evaluation. For a classification task, these will be drawn evenly from
            each label.
        max_generated_tokens:
            The maximum number of tokens to generate when benchmarking the dataset
            using few-shot evaluation.
        prompt_prefix:
            The prefix to use in the few-shot prompt.
        prompt_template:
            The template for the prompt to use when benchmarking the dataset using
            few-shot evaluation.
        instruction_prompt:
            The prompt to use when benchmarking the dataset using instruction-based
            evaluation.
        prompt_label_mapping (optional):
            A mapping from the labels to another phrase which is used as a substitute
            for the label in few-shot evaluation. Defaults to an empty dictionary. If
            set to "auto", the mapping will be set to a 1:1 mapping between the labels
            and themselves.
    """

    labels: list[str]
    prompt_label_mapping: dict[str, str]
    num_few_shot_examples: int
    max_generated_tokens: int
    prompt_prefix: str
    prompt_template: str
    instruction_prompt: str
