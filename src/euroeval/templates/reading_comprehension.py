"""Templates for the Reading Comprehension task."""

from dataclasses import asdict

from ..types import TemplateDict
from .base import BasePromptConfig, PromptConfig

READING_COMPREHENSION_DEFAULTS = BasePromptConfig(
    labels=["a", "b", "c", "d"],
    num_few_shot_examples=2,
    max_generated_tokens=32,
    prompt_label_mapping="auto",
)

RC_TEMPLATES: TemplateDict = {
    "da": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="Følgende er tekster med tilhørende spørgsmål og svar.",
        prompt_template="Tekst: {text}\nSpørgsmål: {question}\nSvar med maks. 3 ord: "
        "{label}",
        instruction_prompt="Tekst: {text}\n\nBesvar følgende spørgsmål om teksten "
        "ovenfor med maks. 3 ord.\n\nSpørgsmål: {question}",
    ),
    "de": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "en": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "es": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "fo": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "fr": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "is": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "it": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nb": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nl": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nn": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "no": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "sv": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
}
