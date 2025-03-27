"""Templates for the Summarization task."""

from dataclasses import asdict

from ..types import TemplateDict
from .base import BasePromptConfig, PromptConfig

SUMMARIZATION_DEFAULTS = BasePromptConfig(
    num_few_shot_examples=1, max_generated_tokens=256
)

SUMM_TEMPLATES: TemplateDict = {
    "da": PromptConfig(
        **asdict(SUMMARIZATION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "de": PromptConfig(
        **asdict(SUMMARIZATION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "en": PromptConfig(
        **asdict(SUMMARIZATION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "es": PromptConfig(
        **asdict(SUMMARIZATION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "fo": PromptConfig(
        **asdict(SUMMARIZATION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "fr": PromptConfig(
        **asdict(SUMMARIZATION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "is": PromptConfig(
        **asdict(SUMMARIZATION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "it": PromptConfig(
        **asdict(SUMMARIZATION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nb": PromptConfig(
        **asdict(SUMMARIZATION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nl": PromptConfig(
        **asdict(SUMMARIZATION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nn": PromptConfig(
        **asdict(SUMMARIZATION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "no": PromptConfig(
        **asdict(SUMMARIZATION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "sv": PromptConfig(
        **asdict(SUMMARIZATION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
}
