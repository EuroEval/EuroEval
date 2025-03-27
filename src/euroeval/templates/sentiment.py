"""Templates for the Sentiment Analysis task."""

from dataclasses import asdict

from ..types import TemplateDict
from .base import BasePromptConfig, PromptConfig

SENTIMENT_DEFAULTS = BasePromptConfig(
    labels=["positive", "neutral", "negative"],
    num_few_shot_examples=12,
    max_generated_tokens=5,
)


def get_label_mapping(language_code: str) -> dict[str, str]:
    """Get the translations for all labels in the specified language.

    Args:
        language_code: The language code for the target language.

    Returns:
        The translated labels for the specified language.
    """
    try:
        return {
            f"{label}": SENT_LABEL_MAPPING[label][language_code]
            for label in SENT_LABEL_MAPPING.keys()
        }
    except KeyError:
        raise KeyError(f"No SENT label mapping found for language '{language_code}'.")


SENT_TEMPLATES: TemplateDict = {
    "da": PromptConfig(
        **asdict(SENTIMENT_DEFAULTS),
        prompt_prefix="Følgende er tekster og deres sentiment, som kan være 'positiv', "
        "'neutral' eller 'negativ'.",
        prompt_template="Tekst: {text}\nSentiment: {label}",
        instruction_prompt="Tekst: {text}\n\nKlassificer sentimentet i teksten. Svar "
        "kun med {labels_str}, og intet andet.",
    ),
    "de": PromptConfig(
        **asdict(SENTIMENT_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "en": PromptConfig(
        **asdict(SENTIMENT_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "es": PromptConfig(
        **asdict(SENTIMENT_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "fo": PromptConfig(
        **asdict(SENTIMENT_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "fr": PromptConfig(
        **asdict(SENTIMENT_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "is": PromptConfig(
        **asdict(SENTIMENT_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "it": PromptConfig(
        **asdict(SENTIMENT_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nb": PromptConfig(
        **asdict(SENTIMENT_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nl": PromptConfig(
        **asdict(SENTIMENT_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nn": PromptConfig(
        **asdict(SENTIMENT_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "no": PromptConfig(
        **asdict(SENTIMENT_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "sv": PromptConfig(
        **asdict(SENTIMENT_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
}

SENT_LABEL_MAPPING = {
    "positive": {
        "da": "positiv",
        "de": "positiv",
        "en": "positive",
        "es": "positivo",
        "fo": "positivt",
        "fr": "positif",
        "is": "jákvætt",
        "it": "positivo",
        "nb": "positiv",
        "nl": "positief",
        "nn": "positiv",
        "no": "positiv",
        "sv": "positiv",
    },
    "neutral": {
        "da": "neutral",
        "de": "neutral",
        "en": "neutral",
        "es": "neutral",
        "fo": "neutralt",
        "fr": "neutre",
        "is": "hlutlaust",
        "it": "neutro",
        "nb": "nøytral",
        "nl": "neutraal",
        "nn": "nøytral",
        "no": "nøytral",
        "sv": "neutral",
    },
    "negative": {
        "da": "negativ",
        "de": "negativ",
        "en": "negative",
        "es": "negativo",
        "fo": "negativt",
        "fr": "négatif",
        "is": "neikvætt",
        "it": "negativo",
        "nb": "negativ",
        "nl": "negatief",
        "nn": "negativ",
        "no": "negativ",
        "sv": "negativ",
    },
}
