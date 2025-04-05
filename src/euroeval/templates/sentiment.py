"""Templates for the Sentiment Analysis task."""

from dataclasses import asdict

from ..types import TemplateDict
from .base import BasePromptConfig, PromptConfig
from .constants import SENT_LABEL_MAPPING

SENTIMENT_DEFAULTS = BasePromptConfig(
    labels=["positive", "neutral", "negative"],
    num_few_shot_examples=12,
    max_generated_tokens=5,
)
SENT_DEFAULTS_DICT = asdict(SENTIMENT_DEFAULTS)
SENT_DEFAULTS_DICT.pop("prompt_label_mapping")


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
        **SENT_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("da"),
        prompt_prefix="Følgende er dokumenter og deres sentiment, som kan være "
        "{labels_str}.",
        prompt_template="Dokument: {text}\nSentiment: {label}",
        instruction_prompt="Dokument: {text}\n\nKlassificer sentimentet i dokumentet. "
        "Svar kun med {labels_str}, og intet andet.",
    ),
    "de": PromptConfig(
        **SENT_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("de"),
        prompt_prefix="Nachfolgend finden Sie Dokumente und ihre Bewertung, die "
        "{labels_str} sein kann.",
        prompt_template="Dokument: {text}\nStimmung: {label}",
        instruction_prompt="Dokument: {text}\n\nKlassifizieren Sie die Stimmung im "
        "Dokument. Antworten Sie mit {labels_str}, und nichts anderes.",
    ),
    "en": PromptConfig(
        **SENT_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("en"),
        prompt_prefix="The following are documents and their sentiment, which can be "
        "{labels_str}.",
        prompt_template="Document: {text}\nSentiment: {label}",
        instruction_prompt="Document: {text}\n\nClassify the sentiment in the "
        "document. Answer with {labels_str}, and nothing else.",
    ),
    "es": PromptConfig(
        **SENT_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("es"),
        prompt_prefix="A continuación se muestran los documentos y su sentimiento, que "
        "puede ser {labels_str}.",
        prompt_template="Documento: {text}\nSentimiento: {label}",
        instruction_prompt="Documento: {text}\n\nClasifica el sentimiento del "
        "documento. Responde con {labels_str}, y nada más.",
    ),
    "fo": PromptConfig(
        **SENT_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("fo"),
        prompt_prefix="Niðanfyri eru skjøl og teirra kenslur, sum kunnu vera "
        "{labels_str}.",
        prompt_template="Skjal: {text}\nKensla: {label}",
        instruction_prompt="Skjal: {text}\n\nFlokka kensluna í skjalinum. Svara við "
        "{labels_str}, og einki annað.",
    ),
    "fr": PromptConfig(
        **SENT_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("fr"),
        prompt_prefix="Les documents suivants sont accompagnés de leur sentiment, "
        "qui peut être {labels_str}.",
        prompt_template="Document: {text}\nSentiment: {label}",
        instruction_prompt="Document: {text}\n\nClassez le sentiment dans le document. "
        "Répondez par {labels_str}, et rien d'autre.",
    ),
    "is": PromptConfig(
        **SENT_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("is"),
        prompt_prefix="Eftirfarandi eru skjöl og viðhorf þeirra, sem geta verið "
        "{labels_str}.",
        prompt_template="Skjal: {text}\nViðhorf: {label}",
        instruction_prompt="Skjal: {text}\n\nFlokkaðu viðhorfið í skjalinu. Svaraðu "
        "með {labels_str}, og ekkert annað.",
    ),
    "it": PromptConfig(
        **SENT_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("it"),
        prompt_prefix="Di seguito sono riportati i documenti e il loro sentiment, "
        "che può essere {labels_str}.",
        prompt_template="Documento: {text}\nSentimento: {label}",
        instruction_prompt="Documento: {text}\n\nClassificare il sentiment del "
        "documento. Rispondere con {labels_str}, e nient'altro.",
    ),
    "nb": PromptConfig(
        **SENT_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("nb"),
        prompt_prefix="Her følger dokumenter og deres sentiment, som kan være "
        "{labels_str}",
        prompt_template="Dokument: {text}\nSentiment: {label}",
        instruction_prompt="Dokument: {text}\n\nKlassifiser følelsen i teksten. Svar "
        "med {labels_str}, og ikke noe annet.",
    ),
    "nl": PromptConfig(
        **SENT_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("nl"),
        prompt_prefix="Hieronder volgen documenten en hun sentiment, dat "
        "{labels_str} kan zijn.",
        prompt_template="Document: {text}\nSentiment: {label}",
        instruction_prompt="Document: {text}\n\nClassificeer het sentiment in het "
        "document. Antwoord met {labels_str}, en verder niets.",
    ),
    "nn": PromptConfig(
        **SENT_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("nn"),
        prompt_prefix="Her følger dokumenter og deres sentiment, som kan være "
        "{labels_str}",
        prompt_template="Dokument: {text}\nSentiment: {label}",
        instruction_prompt="Dokument: {text}\n\nKlassifiser følelsen i teksten. Svar "
        "med {labels_str}, og ikke noe annet.",
    ),
    "no": PromptConfig(
        **SENT_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("no"),
        prompt_prefix="Her følger dokumenter og deres sentiment, som kan være "
        "{labels_str}",
        prompt_template="Dokument: {text}\nSentiment: {label}",
        instruction_prompt="Dokument: {text}\n\nKlassifiser følelsen i teksten. Svar "
        "med {labels_str}, og ikke noe annet.",
    ),
    "sv": PromptConfig(
        **SENT_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("sv"),
        prompt_prefix="Nedan följer dokument och deras sentiment, som kan vara "
        "{labels_str}.",
        prompt_template="Dokument: {text}\nSentiment: {label}",
        instruction_prompt="Dokument: {text}\n\nKlassificera känslan i dokumentet. "
        "Svara med {labels_str}, och inget annat.",
    ),
}
