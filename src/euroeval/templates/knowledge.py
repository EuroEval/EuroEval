"""Templates for the Knowledge task."""

from dataclasses import asdict

from ..types import TemplateDict
from .base import BasePromptConfig, PromptConfig

KNOWLEDGE_DEFAULTS = BasePromptConfig(
    labels=["a", "b", "c", "d"],
    num_few_shot_examples=5,
    max_generated_tokens=5,
    prompt_label_mapping="auto",
)

KNOW_TEMPLATES: TemplateDict = {
    "da": PromptConfig(
        **asdict(KNOWLEDGE_DEFAULTS),
        prompt_prefix="Følgende er multiple choice spørgsmål (med svar).",
        prompt_template="Spørgsmål: {text}\nSvar: {label}",
        instruction_prompt="Spørgsmål: {text}\n\nBesvar ovenstående spørgsmål ved at "
        "svare med {labels_str}, og intet andet.",
    ),
    "de": PromptConfig(
        **asdict(KNOWLEDGE_DEFAULTS),
        prompt_prefix="Die folgenden Fragen sind Multiple-Choice-Fragen "
        "(mit Antworten).",
        prompt_template="Frage: {text}\nAntwort: {label}",
        instruction_prompt="Frage: {text}\n\nBeantworten Sie die obige Frage mit "
        "{labels_str}, und nichts anderes.",
    ),
    "en": PromptConfig(
        **asdict(KNOWLEDGE_DEFAULTS),
        prompt_prefix="The following are multiple choice questions (with answers).",
        prompt_template="Question: {text}\nAnswer: {label}",
        instruction_prompt="Question: {text}\n\nAnswer the above question by replying "
        "with {labels_str}, and nothing else.",
    ),
    "es": PromptConfig(
        **asdict(KNOWLEDGE_DEFAULTS),
        prompt_prefix="Las siguientes son preguntas de opción múltiple "
        "(con respuestas).",
        prompt_template="Pregunta: {text}\nRespuesta: {label}",
        instruction_prompt="Pregunta: {text}\n\nResponda la pregunta anterior usando "
        "solo {labels_str}, y nada más.",
    ),
    "fo": PromptConfig(
        **asdict(KNOWLEDGE_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "fr": PromptConfig(
        **asdict(KNOWLEDGE_DEFAULTS),
        prompt_prefix="Les questions suivantes sont des questions à choix multiples "
        "(avec réponses).",
        prompt_template="Question: {text}\nRéponse: {label}",
        instruction_prompt="Question: {text}\n\nRépondez à la question ci-dessus par "
        "{labels_str}, et rien d'autre.",
    ),
    "is": PromptConfig(
        **asdict(KNOWLEDGE_DEFAULTS),
        prompt_prefix="Eftirfarandi eru fjölvalsspurningar (með svörum).",
        prompt_template="Spurningar: {text}\nSvara: {label}",
        instruction_prompt="Spurningar: {text}\n\nSvaraðu eftirfarandi spurningum með "
        "{labels_str}, og engu öðru.",
    ),
    "it": PromptConfig(
        **asdict(KNOWLEDGE_DEFAULTS),
        prompt_prefix="Le seguenti sono domande a scelta multipla "
        "(con relative risposte).",
        prompt_template="Domanda: {text}\nRéponse: {label}",
        instruction_prompt="Domanda: {text}\n\nRispondete alla domanda precedente con "
        "{labels_str}, e nient'altro.",
    ),
    "nb": PromptConfig(
        **asdict(KNOWLEDGE_DEFAULTS),
        prompt_prefix="Følgende er flervalgsspørsmål (med svar).",
        prompt_template="Spørsmål: {text}\nSvar: {label}",
        instruction_prompt="Spørsmål: {text}\n\nBesvar følgende spørsmål med "
        "{labels_str}, og ikke noe annet.",
    ),
    "nl": PromptConfig(
        **asdict(KNOWLEDGE_DEFAULTS),
        prompt_prefix="Hieronder staan meerkeuzevragen (met antwoorden).",
        prompt_template="Vraag: {text}\nAntwoord: {label}",
        instruction_prompt="Vraag: {text}\n\nBeantwoord de bovenstaande vraag met "
        "{labels_str}, en niets anders.",
    ),
    "nn": PromptConfig(
        **asdict(KNOWLEDGE_DEFAULTS),
        prompt_prefix="Følgende er flervalgsspørsmål (med svar).",
        prompt_template="Spørsmål: {text}\nSvar: {label}",
        instruction_prompt="Spørsmål: {text}\n\nBesvar følgende spørsmål med "
        "{labels_str}, og ikke noe annet.",
    ),
    "no": PromptConfig(
        **asdict(KNOWLEDGE_DEFAULTS),
        prompt_prefix="Følgende er flervalgsspørsmål (med svar).",
        prompt_template="Spørsmål: {text}\nSvar: {label}",
        instruction_prompt="Spørsmål: {text}\n\nBesvar følgende spørsmål med "
        "{labels_str}, og ikke noe annet.",
    ),
    "sv": PromptConfig(
        **asdict(KNOWLEDGE_DEFAULTS),
        prompt_prefix="Följande är flervalsfrågor (med svar).",
        prompt_template="Fråga: {text}\nSvar: {label}",
        instruction_prompt="Fråga: {text}\n\nBesvara följande fråga med {labels_str}, "
        "och inget annat.",
    ),
}
