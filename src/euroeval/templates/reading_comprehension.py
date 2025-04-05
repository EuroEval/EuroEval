"""Templates for the Reading Comprehension task."""

from dataclasses import asdict

from ..types import TemplateDict
from .base import BasePromptConfig, PromptConfig

READING_COMPREHENSION_DEFAULTS = BasePromptConfig(
    labels=["start_positions", "end_positions"],
    num_few_shot_examples=2,
    max_generated_tokens=32,
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
        prompt_prefix="Im Folgenden finden Sie Texte mit den dazugehörigen Fragen und "
        "Antworten.",
        prompt_template="Text: {text}\nFragen: {question}\nFragen Antwort in maximal 3 "
        "Wörtern: {label}",
        instruction_prompt="Text: {text}\n\nBeantworten Sie die folgende Frage zum "
        "obigen Text in höchstens 3 Wörtern.\n\nFrage: {question}",
    ),
    "en": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="The following are texts with accompanying questions and "
        "answers.",
        prompt_template="Text: {text}\nQuestion: {question}\nAnswer in max 3 words: "
        "{label}",
        instruction_prompt="Text: {text}\n\nAnswer the following question about the "
        "above text in at most 3 words.\n\nQuestion: {question}",
    ),
    "es": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="A continuación se presentan textos con sus preguntas y "
        "respuestas correspondientes.",
        prompt_template="Texto: {text}\nPregunta: {question}\nRespuesta en máximo 3 "
        "palabras: {label}",
        instruction_prompt="Texto: {text}\n\nResponda la siguiente pregunta sobre el "
        "texto anterior en máximo 3 palabras.\n\nPregunta: {question}",
    ),
    "fo": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="Hetta eru tekstir saman við spurningum og svar.",
        prompt_template="Tekstur: {text}\nSpurningur: {question}\nSvara við í mesta "
        "lagi trimum orðum: {label}",
        instruction_prompt="Tekstur: {text}\n\nSvara hesum spurninginum um tekstin "
        "uppiyvir við í mesta lagi trimum orðum.\n\nSpurningur: {question}",
    ),
    "fr": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="Les textes suivants sont accompagnés de questions et de "
        "réponses.",
        prompt_template="Texte: {text}\nQuestion: {question}\nRéponse en 3 mots "
        "maximum: {label}",
        instruction_prompt="Texte: {text}\n\nRépondez à la question suivante sur le "
        "texte ci-dessus en 3 mots maximum.\n\nQuestion: {question}",
    ),
    "is": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="Eftirfarandi eru textar með tilheyrandi spurningum og svörum.",
        prompt_template="Texti: {text}\nSpurning: {question}\nSvaraðu með að hámarki 3 "
        "orðum: {label}",
        instruction_prompt="Texti: {text}\n\nSvaraðu eftirfarandi spurningu um textann "
        "að hámarki í 3 orðum.\n\nSpurning: {question}",
    ),
    "it": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="I testi che seguono sono accompagnati da domande e risposte.",
        prompt_template="Testo: {text}\nDomanda: {question}\nRispondere in massimo "
        "3 parole: {label}",
        instruction_prompt="Testo: {text}\n\nRispondi alla seguente domanda sul "
        "in un massimo di 3 parole.\n\nDomanda: {question}",
    ),
    "nb": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="Her følger tekster med tilhørende spørsmål og svar.",
        prompt_template="Tekst: {text}\nSpørsmål: {question}\nSvar på maks 3 ord: "
        "{label}",
        instruction_prompt="Tekst: {text}\n\nBesvar følgende spørsmål om teksten "
        "ovenfor med maks 3 ord.\n\nSpørsmål: {question}",
    ),
    "nl": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="Hieronder volgen teksten met bijbehorende vragen en antwoorden.",
        prompt_template="Tekst: {text}\nVraag: {question}\nAntwoord in max 3 woorden: "
        "{label}",
        instruction_prompt="Tekst: {text}\n\nBeantwoord de volgende vraag over de "
        "bovenstaande tekst in maximaal 3 woorden.\n\nVraag: {question}",
    ),
    "nn": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="Her følger tekster med tilhørende spørsmål og svar.",
        prompt_template="Tekst: {text}\nSpørsmål: {question}\nSvar på maks 3 ord: "
        "{label}",
        instruction_prompt="Tekst: {text}\n\nBesvar følgende spørsmål om teksten "
        "ovenfor med maks 3 ord.\n\nSpørsmål: {question}",
    ),
    "no": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="Her følger tekster med tilhørende spørsmål og svar.",
        prompt_template="Tekst: {text}\nSpørsmål: {question}\nSvar på maks 3 ord: "
        "{label}",
        instruction_prompt="Tekst: {text}\n\nBesvar følgende spørsmål om teksten "
        "ovenfor med maks 3 ord.\n\nSpørsmål: {question}",
    ),
    "sv": PromptConfig(
        **asdict(READING_COMPREHENSION_DEFAULTS),
        prompt_prefix="Nedan följer texter med tillhörande frågor och svar.",
        prompt_template="Text: {text}\nFråga: {question}\nSvar på max 3 ord: {label}",
        instruction_prompt="Text: {text}\n\nBesvara följande fråga om texten ovan med "
        "högst 3 ord.\n\nFråga: {question}",
    ),
}
