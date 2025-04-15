"""Templates for the Reading Comprehension task."""

from ..data_models import PromptConfig
from ..languages import DA, DE, EN, ES, FO, FR, IS, IT, NB, NL, NN, NO, SV

RC_TEMPLATES = {
    DA: PromptConfig(
        default_prompt_prefix="Følgende er tekster med tilhørende spørgsmål og svar.",
        default_prompt_template="Tekst: {text}\nSpørgsmål: {question}\nSvar med maks. "
        "3 ord: {label}",
        default_instruction_prompt="Tekst: {text}\n\nBesvar følgende spørgsmål om "
        "teksten ovenfor med maks. 3 ord.\n\nSpørgsmål: {question}",
        default_prompt_label_mapping=dict(),
    ),
    DE: PromptConfig(
        default_prompt_prefix="Im Folgenden finden Sie Texte mit den dazugehörigen "
        "Fragen und Antworten.",
        default_prompt_template="Text: {text}\nFragen: {question}\nFragen Antwort in "
        "maximal 3 Wörtern: {label}",
        default_instruction_prompt="Text: {text}\n\nBeantworten Sie die folgende Frage "
        "zum obigen Text in höchstens 3 Wörtern.\n\nFrage: {question}",
        default_prompt_label_mapping=dict(),
    ),
    EN: PromptConfig(
        default_prompt_prefix="The following are texts with accompanying questions and "
        "answers.",
        default_prompt_template="Text: {text}\nQuestion: {question}\nAnswer in max "
        "3 words: {label}",
        default_instruction_prompt="Text: {text}\n\nAnswer the following question "
        "about the above text in at most 3 words.\n\nQuestion: {question}",
        default_prompt_label_mapping=dict(),
    ),
    ES: PromptConfig(
        default_prompt_prefix="A continuación se presentan textos con sus preguntas y "
        "respuestas correspondientes.",
        default_prompt_template="Texto: {text}\nPregunta: {question}\nRespuesta en "
        "máximo 3 palabras: {label}",
        default_instruction_prompt="Texto: {text}\n\nResponda la siguiente pregunta "
        "sobre el texto anterior en máximo 3 palabras.\n\nPregunta: {question}",
        default_prompt_label_mapping=dict(),
    ),
    FO: PromptConfig(
        default_prompt_prefix="Hetta eru tekstir saman við spurningum og svar.",
        default_prompt_template="Tekstur: {text}\nSpurningur: {question}\nSvara við í "
        "mesta lagi trimum orðum: {label}",
        default_instruction_prompt="Tekstur: {text}\n\nSvara hesum spurninginum um "
        "tekstin uppiyvir við í mesta lagi trimum orðum.\n\nSpurningur: {question}",
        default_prompt_label_mapping=dict(),
    ),
    FR: PromptConfig(
        default_prompt_prefix="Les textes suivants sont accompagnés de questions et de "
        "réponses.",
        default_prompt_template="Texte: {text}\nQuestion: {question}\nRéponse en 3 "
        "mots maximum: {label}",
        default_instruction_prompt="Texte: {text}\n\nRépondez à la question suivante "
        "sur le texte ci-dessus en 3 mots maximum.\n\nQuestion: {question}",
        default_prompt_label_mapping=dict(),
    ),
    IS: PromptConfig(
        default_prompt_prefix="Eftirfarandi eru textar með tilheyrandi spurningum og "
        "svörum.",
        default_prompt_template="Texti: {text}\nSpurning: {question}\nSvaraðu með að "
        "hámarki 3 orðum: {label}",
        default_instruction_prompt="Texti: {text}\n\nSvaraðu eftirfarandi spurningu um "
        "textann að hámarki í 3 orðum.\n\nSpurning: {question}",
        default_prompt_label_mapping=dict(),
    ),
    IT: PromptConfig(
        default_prompt_prefix="I testi che seguono sono accompagnati da domande e "
        "risposte.",
        default_prompt_template="Testo: {text}\nDomanda: {question}\nRispondere in "
        "massimo 3 parole: {label}",
        default_instruction_prompt="Testo: {text}\n\nRispondi alla seguente domanda "
        "sul in un massimo di 3 parole.\n\nDomanda: {question}",
        default_prompt_label_mapping=dict(),
    ),
    NB: PromptConfig(
        default_prompt_prefix="Her følger tekster med tilhørende spørsmål og svar.",
        default_prompt_template="Tekst: {text}\nSpørsmål: {question}\nSvar på maks 3 "
        "ord: {label}",
        default_instruction_prompt="Tekst: {text}\n\nBesvar følgende spørsmål om "
        "teksten ovenfor med maks 3 ord.\n\nSpørsmål: {question}",
        default_prompt_label_mapping=dict(),
    ),
    NL: PromptConfig(
        default_prompt_prefix="Hieronder volgen teksten met bijbehorende vragen en "
        "antwoorden.",
        default_prompt_template="Tekst: {text}\nVraag: {question}\nAntwoord in max "
        "3 woorden: {label}",
        default_instruction_prompt="Tekst: {text}\n\nBeantwoord de volgende vraag "
        "over de bovenstaande tekst in maximaal 3 woorden.\n\nVraag: {question}",
        default_prompt_label_mapping=dict(),
    ),
    NN: PromptConfig(
        default_prompt_prefix="Her følger tekster med tilhørende spørsmål og svar.",
        default_prompt_template="Tekst: {text}\nSpørsmål: {question}\nSvar på maks 3 "
        "ord: {label}",
        default_instruction_prompt="Tekst: {text}\n\nBesvar følgende spørsmål om "
        "teksten ovenfor med maks 3 ord.\n\nSpørsmål: {question}",
        default_prompt_label_mapping=dict(),
    ),
    NO: PromptConfig(
        default_prompt_prefix="Her følger tekster med tilhørende spørsmål og svar.",
        default_prompt_template="Tekst: {text}\nSpørsmål: {question}\nSvar på maks 3 "
        "ord: {label}",
        default_instruction_prompt="Tekst: {text}\n\nBesvar følgende spørsmål om "
        "teksten ovenfor med maks 3 ord.\n\nSpørsmål: {question}",
        default_prompt_label_mapping=dict(),
    ),
    SV: PromptConfig(
        default_prompt_prefix="Nedan följer texter med tillhörande frågor och svar.",
        default_prompt_template="Text: {text}\nFråga: {question}\nSvar på max 3 ord: "
        "{label}",
        default_instruction_prompt="Text: {text}\n\nBesvara följande fråga om texten "
        "ovan med högst 3 ord.\n\nFråga: {question}",
        default_prompt_label_mapping=dict(),
    ),
}
