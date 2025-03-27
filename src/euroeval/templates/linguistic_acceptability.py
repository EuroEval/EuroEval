"""Templates for the Linguistic Acceptability task."""

from dataclasses import asdict

from ..types import TemplateDict
from .base import BasePromptConfig, PromptConfig
from .constants import LA_LABEL_MAPPING

LA_DEFAULTS = BasePromptConfig(
    labels=["incorrect", "correct"], num_few_shot_examples=12, max_generated_tokens=5
)
LA_DEFAULTS_DICT = asdict(LA_DEFAULTS)
LA_DEFAULTS_DICT.pop("prompt_label_mapping")


def get_label_mapping(language_code: str) -> dict[str, str]:
    """Get the translations for all labels in the specified language.

    Args:
        language_code: The language code for the target language.

    Returns:
        The translated labels for the specified language.
    """
    try:
        return {
            f"{label}": LA_LABEL_MAPPING[label][language_code]
            for label in LA_LABEL_MAPPING.keys()
        }
    except KeyError:
        raise KeyError(f"No LA label mapping found for language '{language_code}'.")


LA_TEMPLATES: TemplateDict = {
    "da": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("da"),
        prompt_prefix="Følgende er sætninger og om de er grammatisk korrekte.",
        prompt_template="Sætning: {text}\nGrammatisk korrekt: {label}",
        instruction_prompt="Sætning: {text}\n\nBestem om sætningen er grammatisk "
        "korrekt eller ej. Svar med 'ja', hvis sætningen er korrekt, og 'nej', hvis "
        "den ikke er, og intet andet.",
    ),
    "de": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("de"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "en": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("en"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "es": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("es"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "fo": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("fo"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "fr": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("fr"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "is": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("is"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "it": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("it"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nb": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("nb"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nl": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("nl"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nn": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("nn"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "no": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("no"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "sv": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("sv"),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
}
