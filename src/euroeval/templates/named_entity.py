"""Templates for the Named Entity Recognition task."""

from dataclasses import asdict

from ..types import TemplateDict
from .base import BasePromptConfig, PromptConfig
from .constants import NER_TAG_MAPPING

NER_DEFAULTS = BasePromptConfig(
    labels=[
        "o",
        "b-loc",
        "i-loc",
        "b-org",
        "i-org",
        "b-per",
        "i-per",
        "b-misc",
        "i-misc",
    ],
    num_few_shot_examples=8,
    max_generated_tokens=128,
)
NER_DEFAULTS_DICT = asdict(NER_DEFAULTS)
NER_DEFAULTS_DICT.pop("prompt_label_mapping")


def get_ner_mapping(language_code: str) -> dict[str, str]:
    """Get the translations for all NER tags in the specified language.

    Args:
        language_code: The language code for the target language.

    Returns:
        The translated tags for the specified language.
    """
    try:
        # Add 'b-' or 'i-' prefix to get the actual tag
        return {
            f"{prefix}{tag}": NER_TAG_MAPPING[tag][language_code]
            for tag in NER_TAG_MAPPING.keys()
            for prefix in ["b-", "i-"]
        }
    except KeyError:
        raise KeyError(f"No NER tag mapping found for language '{language_code}'.")


NER_TEMPLATES: TemplateDict = {
    "da": PromptConfig(
        **NER_DEFAULTS_DICT,
        prompt_label_mapping=get_ner_mapping("da"),
        prompt_prefix="Følgende er sætninger og JSON-ordbøger med de navngivne "
        "enheder, som forekommer i den givne sætning.",
        prompt_template="Sætning: {text}\nNavngivne enheder: {label}",
        instruction_prompt="Sætning: {text}\n\nIdentificér de navngivne enheder i "
        "sætningen. Du skal outputte dette som en JSON-ordbog med nøglerne 'person', "
        "'sted', 'organisation' og 'diverse'. Værdierne skal være lister over de "
        "navngivne enheder af den type, præcis som de forekommer i sætningen.",
    ),
    "de": PromptConfig(
        **NER_DEFAULTS_DICT,
        prompt_label_mapping=get_ner_mapping("de"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "en": PromptConfig(
        **NER_DEFAULTS_DICT,
        prompt_label_mapping=get_ner_mapping("en"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "es": PromptConfig(
        **NER_DEFAULTS_DICT,
        prompt_label_mapping=get_ner_mapping("es"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "fo": PromptConfig(
        **NER_DEFAULTS_DICT,
        prompt_label_mapping=get_ner_mapping("fo"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "fr": PromptConfig(
        **NER_DEFAULTS_DICT,
        prompt_label_mapping=get_ner_mapping("fr"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "is": PromptConfig(
        **NER_DEFAULTS_DICT,
        prompt_label_mapping=get_ner_mapping("is"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "it": PromptConfig(
        **NER_DEFAULTS_DICT,
        prompt_label_mapping=get_ner_mapping("it"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nb": PromptConfig(
        **NER_DEFAULTS_DICT,
        prompt_label_mapping=get_ner_mapping("nb"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nl": PromptConfig(
        **NER_DEFAULTS_DICT,
        prompt_label_mapping=get_ner_mapping("nl"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nn": PromptConfig(
        **NER_DEFAULTS_DICT,
        prompt_label_mapping=get_ner_mapping("nn"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "no": PromptConfig(
        **NER_DEFAULTS_DICT,
        prompt_label_mapping=get_ner_mapping("no"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "sv": PromptConfig(
        **NER_DEFAULTS_DICT,
        prompt_label_mapping=get_ner_mapping("sv"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
}
