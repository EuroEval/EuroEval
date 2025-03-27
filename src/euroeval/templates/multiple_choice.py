"""Templates for the Multiple Choice Reading Comprehension task."""

from dataclasses import asdict

from ..types import TemplateDict
from .base import BasePromptConfig, PromptConfig

MULTIPLE_CHOICE_DEFAULTS = BasePromptConfig(
    labels=["a", "b", "c", "d"],
    num_few_shot_examples=5,
    max_generated_tokens=5,
    prompt_label_mapping="auto",
)

MCRC_TEMPLATES: TemplateDict = {
    "da": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "de": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "en": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "es": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "fo": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "fr": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "is": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "it": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nb": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nl": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nn": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "no": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "sv": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
}
