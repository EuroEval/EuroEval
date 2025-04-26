"""Templates for all multiple choice tasks."""

from ..data_models import PromptConfig
from ..languages import DA, DE, EN, ES, FI, FR, IS, IT, NB, NL, NN, NO, SV

# TODO: Missing Faroese
MULTIPLE_CHOICE_TEMPLATES = {
    DA: PromptConfig(
        default_prompt_prefix="Følgende er multiple choice spørgsmål (med svar).",
        default_prompt_template="Spørgsmål: {text}\nSvar: {label}",
        default_instruction_prompt="Spørgsmål: {text}\n\nBesvar ovenstående spørgsmål "
        "ved at svare med {labels_str}, og intet andet.",
        default_prompt_label_mapping="auto",
    ),
    DE: PromptConfig(
        default_prompt_prefix="Die folgenden Fragen sind Multiple-Choice-Fragen "
        "(mit Antworten).",
        default_prompt_template="Frage: {text}\nAntwort: {label}",
        default_instruction_prompt="Frage: {text}\n\nBeantworten Sie die obige Frage "
        "mit {labels_str}, und nichts anderes.",
        default_prompt_label_mapping="auto",
    ),
    EN: PromptConfig(
        default_prompt_prefix="The following are multiple choice questions (with "
        "answers).",
        default_prompt_template="Question: {text}\nAnswer: {label}",
        default_instruction_prompt="Question: {text}\n\nAnswer the above question by "
        "replying with {labels_str}, and nothing else.",
        default_prompt_label_mapping="auto",
    ),
    ES: PromptConfig(
        default_prompt_prefix="Las siguientes son preguntas de opción múltiple "
        "(con respuestas).",
        default_prompt_template="Pregunta: {text}\nRespuesta: {label}",
        default_instruction_prompt="Pregunta: {text}\n\nResponda la pregunta anterior "
        "usando solo {labels_str}, y nada más.",
        default_prompt_label_mapping="auto",
    ),
    FI: PromptConfig(
        default_prompt_prefix="Seuraavat ovat monivalintakysymyksiä (vastauksineen).",
        default_prompt_template="Kysymys: {text}\nVastaus: {label}",
        default_instruction_prompt="Kysymys: {text}\n\nVastaa yllä olevaan kysymykseen "
        "käyttämällä {labels_str}, äläkä mitään muuta.",
        default_prompt_label_mapping="auto",
    ),
    FR: PromptConfig(
        default_prompt_prefix="Les questions suivantes sont des questions à choix "
        "multiples (avec réponses).",
        default_prompt_template="Question: {text}\nRéponse: {label}",
        default_instruction_prompt="Question: {text}\n\nRépondez à la question "
        "ci-dessus par {labels_str}, et rien d'autre.",
        default_prompt_label_mapping="auto",
    ),
    IS: PromptConfig(
        default_prompt_prefix="Eftirfarandi eru fjölvalsspurningar (með svörum).",
        default_prompt_template="Spurningar: {text}\nSvara: {label}",
        default_instruction_prompt="Spurningar: {text}\n\nSvaraðu eftirfarandi "
        "spurningum með {labels_str}, og engu öðru.",
        default_prompt_label_mapping="auto",
    ),
    IT: PromptConfig(
        default_prompt_prefix="Le seguenti sono domande a scelta multipla "
        "(con relative risposte).",
        default_prompt_template="Domanda: {text}\nRéponse: {label}",
        default_instruction_prompt="Domanda: {text}\n\nRispondete alla domanda "
        "precedente con {labels_str}, e nient'altro.",
        default_prompt_label_mapping="auto",
    ),
    NB: PromptConfig(
        default_prompt_prefix="Følgende er flervalgsspørsmål (med svar).",
        default_prompt_template="Spørsmål: {text}\nSvar: {label}",
        default_instruction_prompt="Spørsmål: {text}\n\nBesvar følgende spørsmål med "
        "{labels_str}, og ikke noe annet.",
        default_prompt_label_mapping="auto",
    ),
    NL: PromptConfig(
        default_prompt_prefix="Hieronder staan meerkeuzevragen (met antwoorden).",
        default_prompt_template="Vraag: {text}\nAntwoord: {label}",
        default_instruction_prompt="Vraag: {text}\n\nBeantwoord de bovenstaande vraag "
        "met {labels_str}, en niets anders.",
        default_prompt_label_mapping="auto",
    ),
    NN: PromptConfig(
        default_prompt_prefix="Følgende er flervalgsspørsmål (med svar).",
        default_prompt_template="Spørsmål: {text}\nSvar: {label}",
        default_instruction_prompt="Spørsmål: {text}\n\nBesvar følgende spørsmål med "
        "{labels_str}, og ikke noe annet.",
        default_prompt_label_mapping="auto",
    ),
    NO: PromptConfig(
        default_prompt_prefix="Følgende er flervalgsspørsmål (med svar).",
        default_prompt_template="Spørsmål: {text}\nSvar: {label}",
        default_instruction_prompt="Spørsmål: {text}\n\nBesvar følgende spørsmål med "
        "{labels_str}, og ikke noe annet.",
        default_prompt_label_mapping="auto",
    ),
    SV: PromptConfig(
        default_prompt_prefix="Följande är flervalsfrågor (med svar).",
        default_prompt_template="Fråga: {text}\nSvar: {label}",
        default_instruction_prompt="Fråga: {text}\n\nBesvara följande fråga med "
        "{labels_str}, och inget annat.",
        default_prompt_label_mapping="auto",
    ),
}
