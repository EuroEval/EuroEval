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
        prompt_prefix="Følgende er tekster med tilhørende multiple choice spørgsmål "
        "og svar.",
        prompt_template="{text}\nSvar: {label}",
        instruction_prompt="{text}\n\nBesvar ovenstående spørgsmål ved at svare med "
        "{labels_str}, og intet andet.",
    ),
    "de": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="Die folgenden Texte sind mit dazugehörigen "
        "Multiple-Choice-Fragen und Antworten.",
        prompt_template="{text}\nAntwort: {label}",
        instruction_prompt="{text}\n\nBeantworten Sie die obige Frage mit "
        "{labels_str} , und nichts anderes.",
    ),
    "en": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="The following are texts with accompanying multiple choice "
        "questions and answers.",
        prompt_template="{text}\nAnswer: {label}",
        instruction_prompt="{text}\n\nAnswer the above question by replying with "
        "{labels_str}, and nothing else.",
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
        prompt_prefix="Les textes suivants sont accompagnés de questions à choix "
        "multiples et de réponses.",
        prompt_template="{text}\nRéponse: {label}",
        instruction_prompt="{text}\n\nRépondez à la question ci-dessus par "
        "{labels_str}, et rien d'autre.",
    ),
    "is": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="Eftirfarandi eru textar með tilheyrandi fjölvalsspurningum og "
        "svörum.",
        prompt_template="{text}\nSvara: {label}",
        instruction_prompt="{text}\n\nSvaraðu eftirfarandi spurningum með "
        "{labels_str}, og engu öðru.",
    ),
    "it": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
    ),
    "nb": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="Her følger tekster med tilhørende multiple choice spørsmål "
        "og svar.",
        prompt_template="{text}\nSvar: {label}",
        instruction_prompt="{text}\n\nBesvar følgende spørsmål med {labels_str}, "
        "og ikke noe annet.",
    ),
    "nl": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="Hieronder staan teksten met bijbehorende multiple choice "
        "vragen en antwoorden.",
        prompt_template="{text}\nAntwoord: {label}",
        instruction_prompt="{text}\n\nBeantwoord de bovenstaande vraag met "
        "{label_str}, en niets anders.",
    ),
    "nn": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="Her følger tekster med tilhørende multiple choice spørsmål "
        "og svar.",
        prompt_template="{text}\nSvar: {label}",
        instruction_prompt="{text}\n\nBesvar følgende spørsmål med {labels_str}, "
        "og ikke noe annet.",
    ),
    "no": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="Her følger tekster med tilhørende multiple choice spørsmål "
        "og svar.",
        prompt_template="{text}\nSvar: {label}",
        instruction_prompt="{text}\n\nBesvar følgende spørsmål med {labels_str}, "
        "og ikke noe annet.",
    ),
    "sv": PromptConfig(
        **asdict(MULTIPLE_CHOICE_DEFAULTS),
        prompt_prefix="Nedan följer texter med tillhörande multiple choice frågor och "
        "svar.",
        prompt_template="{text}\nSvar: {label}",
        instruction_prompt="{text}\n\nBesvara följande fråga med {labels_str}, "
        "och inget annat.",
    ),
}
