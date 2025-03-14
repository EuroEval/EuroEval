"""All templates for all tasks used in EuroEval."""

from .constants import (
    LA_LABELS,
    MULTIPLE_CHOICE_LABELS,
    MULTIPLE_CHOICE_MAPPING,
    NER_LABELS,
    RC_LABELS,
    SENT_LABELS,
)
from .data_models import Language, PromptTemplate, Task
from .languages import DA, DE, EN, FO, FR, IS, IT, NB, NL, NN, NO, SV
from .tasks import COMMON_SENSE, KNOW, LA, MCRC, NER, RC, SENT, SPEED, SUMM


def get_task_templates(task: Task, language: Language) -> PromptTemplate:
    """Gets the template for a specific task and language.

    Args:
        task (Task): The `Task` to get templates for.
        language (Language): The `Language` for the task.

    Raises:
        NotImplementedError: If the task is not supported/implemented
        KeyError: If the language doesn't have a template for the task

    Returns:
        PromptTemplate: The `PromptTemplate` dataclass for the given task and language
    """
    try:
        match task.name:
            case COMMON_SENSE.name:
                return COMMON_SENSE_TEMPLATE[language]
            case KNOW.name:
                return KNOW_TEMPLATE[language]
            case LA.name:
                return LA_TEMPLATE[language]
            case MCRC.name:
                return MCRC_TEMPLATE[language]
            case NER.name:
                return NER_TEMPLATE[language]
            case RC.name:
                return RC_TEMPLATE[language]
            case SENT.name:
                return SENT_TEMPLATE[language]
            case SPEED.name:
                return PromptTemplate(
                    prompt_prefix="", prompt_template="", instruction_prompt=""
                )
            case SUMM.name:
                return SUMM_TEMPLATE[language]
            case _:
                raise NotImplementedError(f"Unsupported task: {task}.")
    except KeyError:
        raise KeyError(f"No template found for language '{language}' in task '{task}'")


### COMMON SENSE REASONING TEMPLATES ###

COMMON_SENSE_TEMPLATE = {
    DA: PromptTemplate(
        prompt_prefix="Følgende er multiple choice spørgsmål (med svar).",
        prompt_template="Spørgsmål: {text}\nSvar: {label}",
        instruction_prompt="Spørgsmål: {text}\n\nBesvar ovenstående spørgsmål ved at "
        "svare med 'a', 'b', 'c' eller 'd', og intet andet.",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    DE: PromptTemplate(
        prompt_prefix="Die folgenden Fragen sind Multiple-Choice-Fragen "
        "(mit Antworten).",
        prompt_template="Frage: {text}\nAntwort: {label}",
        instruction_prompt="Frage: {text}\n\nBeantworten Sie die obige Frage mit 'a', "
        "'b', 'c' oder 'd', und nichts anderes.",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    EN: PromptTemplate(
        prompt_prefix="The following are multiple choice questions (with answers).",
        prompt_template="Question: {text}\nAnswer: {label}",
        instruction_prompt="Question: {text}\n\nAnswer the above question by replying "
        "with 'a', 'b', 'c' or 'd', and nothing else.",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    # FO: PromptTemplate(
    #     prompt_prefix="",
    #     prompt_template="",
    #     instruction_prompt="",
    #     prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
    #     labels=MULTIPLE_CHOICE_LABELS,
    # ),
    FR: PromptTemplate(
        prompt_prefix="Les questions suivantes sont des questions à choix multiples "
        "(avec réponses).",
        prompt_template="Question: {text}\nRéponse: {label}",
        instruction_prompt="Question: {text}\n\nRépondez à la question ci-dessus par "
        "'a', 'b', 'c' ou 'd', et rien d'autre.",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    IS: PromptTemplate(
        prompt_prefix="Eftirfarandi eru fjölvalsspurningar (með svörum).",
        prompt_template="Spurningar: {text}\nSvara: {label}",
        instruction_prompt="Spurningar: {text}\n\nSvaraðu eftirfarandi spurningum með "
        "'a', 'b', 'c' eða 'd', og engu öðru.",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    IT: PromptTemplate(
        prompt_prefix="Le seguenti sono domande a scelta multipla "
        "(con relative risposte).",
        prompt_template="Domanda: {text}\nRéponse: {label}",
        instruction_prompt="Domanda: {text}\n\nRispondete alla domanda precedente con "
        "'a', 'b', 'c' o 'd' e nient'altro.",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    NB: PromptTemplate(
        prompt_prefix="Følgende er flervalgsspørsmål (med svar).",
        prompt_template="Spørsmål: {text}\nSvar: {label}",
        instruction_prompt="Spørsmål: {text}\n\nBesvar følgende spørsmål med 'a', 'b', "
        "'c' eller 'd', og ikke noe annet.",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    NL: PromptTemplate(
        prompt_prefix="Hieronder staan meerkeuzevragen (met antwoorden).",
        prompt_template="Vraag: {text}\nAntwoord: {label}",
        instruction_prompt="Vraag: {text}\n\nBeantwoord de bovenstaande vraag met 'a', "
        "'b', 'c' of 'd', en niets anders.",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    NN: PromptTemplate(
        prompt_prefix="Følgende er flervalgsspørsmål (med svar).",
        prompt_template="Spørsmål: {text}\nSvar: {label}",
        instruction_prompt="Spørsmål: {text}\n\nBesvar følgende spørsmål med 'a', 'b', "
        "'c' eller 'd', og ikke noe annet.",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    NO: PromptTemplate(
        prompt_prefix="Følgende er flervalgsspørsmål (med svar).",
        prompt_template="Spørsmål: {text}\nSvar: {label}",
        instruction_prompt="Spørsmål: {text}\n\nBesvar følgende spørsmål med 'a', 'b', "
        "'c' eller 'd', og ikke noe annet.",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    SV: PromptTemplate(
        prompt_prefix="Följande är flervalsfrågor (med svar).",
        prompt_template="Fråga: {text}\nSvar: {label}",
        instruction_prompt="Fråga: {text}\n\nBesvara följande fråga med 'a', 'b', 'c' "
        "eller 'd', och inget annat.",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
}

### KNOWLEDGE TEMPLATES ###

KNOW_TEMPLATE = {
    DA: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    DE: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    EN: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    # FO: PromptTemplate(
    #     prompt_prefix="",
    #     prompt_template="",
    #     instruction_prompt="",
    #     prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
    #     labels=MULTIPLE_CHOICE_LABELS
    # ),
    FR: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    IS: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    IT: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    NB: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    NL: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    NN: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    NO: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    SV: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
}

### MULTIPLE CHOICE READING COMPREHENSION TEMPLATES ###

MCRC_TEMPLATE = {
    DA: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    DE: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    EN: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    FO: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    FR: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    IS: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    IT: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    NB: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    NL: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    NN: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    NO: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
    SV: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=MULTIPLE_CHOICE_MAPPING,
        labels=MULTIPLE_CHOICE_LABELS,
    ),
}

### LINGUISTIC ACCEPTABILITY TEMPLATES ###

LA_TEMPLATE = {
    DA: PromptTemplate(
        prompt_prefix="Følgende er sætninger og om de er grammatisk korrekte.",
        prompt_template="Sætning: {text}\nGrammatisk korrekt: {label}",
        instruction_prompt="Sætning: {text}\n\nBestem om sætningen er grammatisk "
        "korrekt eller ej. Svar med 'ja', hvis sætningen er korrekt, og 'nej', hvis "
        "den ikke er, og intet andet.",
        prompt_label_mapping=dict(incorrect="nej", correct="ja"),
        labels=LA_LABELS,
    ),
    DE: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=LA_LABELS,
    ),
    EN: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=LA_LABELS,
    ),
    # FO: PromptTemplate(
    #     prompt_prefix="",
    #     prompt_template="",
    #     instruction_prompt="",
    #     prompt_label_mapping=dict(a=""), #TODO: INSERT LABEL MAPPING
    #     labels=LA_LABELS
    # ),
    FR: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=LA_LABELS,
    ),
    IS: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=LA_LABELS,
    ),
    IT: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=LA_LABELS,
    ),
    NB: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=LA_LABELS,
    ),
    NL: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=LA_LABELS,
    ),
    NN: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=LA_LABELS,
    ),
    NO: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=LA_LABELS,
    ),
    SV: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=LA_LABELS,
    ),
}

### NAMED ENTITY RECOGNITION TEMPLATES ###

NER_TEMPLATE = {
    DA: PromptTemplate(
        prompt_prefix="Følgende er sætninger og JSON-ordbøger med de navngivne "
        "enheder, som forekommer i den givne sætning.",
        prompt_template="Sætning: {text}\nNavngivne enheder: {label}",
        instruction_prompt="Sætning: {text}\n\nIdentificér de navngivne enheder i "
        "sætningen. Du skal outputte dette som en JSON-ordbog med nøglerne 'person', "
        "'sted', 'organisation' og 'diverse'. Værdierne skal være lister over de "
        "navngivne enheder af den type, præcis som de forekommer i sætningen.",
        labels=NER_LABELS,
        prompt_label_mapping={
            "b-per": "person",
            "i-per": "person",
            "b-loc": "sted",
            "i-loc": "sted",
            "b-org": "organisation",
            "i-org": "organisation",
            "b-misc": "diverse",
            "i-misc": "diverse",
        },
    ),
    DE: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=NER_LABELS,
    ),
    EN: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=NER_LABELS,
    ),
    # FO: PromptTemplate(
    #     prompt_prefix="",
    #     prompt_template="",
    #     instruction_prompt="",
    #     prompt_label_mapping=dict(a=""), #TODO: INSERT LABEL MAPPING
    #     labels=NER_LABELS
    # ),
    FR: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=NER_LABELS,
    ),
    IS: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=NER_LABELS,
    ),
    IT: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=NER_LABELS,
    ),
    NB: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=NER_LABELS,
    ),
    NL: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=NER_LABELS,
    ),
    NN: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=NER_LABELS,
    ),
    NO: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=NER_LABELS,
    ),
    SV: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=NER_LABELS,
    ),
}

### READING COMPREHENSION DATASETS ###

RC_TEMPLATE = {
    DA: PromptTemplate(
        prompt_prefix="Følgende er tekster med tilhørende spørgsmål og svar.",
        prompt_template="Tekst: {text}\nSpørgsmål: {question}\nSvar med maks. 3 ord: "
        "{label}",
        instruction_prompt="Tekst: {text}\n\nBesvar følgende spørgsmål om teksten "
        "ovenfor med maks. 3 ord.\n\nSpørgsmål: {question}",
        labels=RC_LABELS,
    ),
    DE: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=RC_LABELS,
    ),
    EN: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=RC_LABELS,
    ),
    # FO: PromptTemplate(
    #     prompt_prefix="",
    #     prompt_template="",
    #     instruction_prompt="",
    #     prompt_label_mapping=dict(a=""), #TODO: INSERT LABEL MAPPING
    #     labels=RC_LABELS
    # ),
    FR: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=RC_LABELS,
    ),
    IS: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=RC_LABELS,
    ),
    IT: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=RC_LABELS,
    ),
    NB: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=RC_LABELS,
    ),
    NL: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=RC_LABELS,
    ),
    NN: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=RC_LABELS,
    ),
    NO: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=RC_LABELS,
    ),
    SV: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=RC_LABELS,
    ),
}

### SENTIMENT TEMPLATES ###

SENT_TEMPLATE = {
    DA: PromptTemplate(
        prompt_prefix="Følgende er tekster og deres sentiment, som kan være 'positiv', "
        "'neutral' eller 'negativ'.",
        prompt_template="Tekst: {text}\nSentiment: {label}",
        instruction_prompt="Tekst: {text}\n\nKlassificer sentimentet i teksten. Svar "
        "kun med 'positiv', 'neutral' eller 'negativ', og intet andet.",
        labels=SENT_LABELS,
    ),
    DE: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=SENT_LABELS,
    ),
    EN: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=SENT_LABELS,
    ),
    # FO: PromptTemplate(
    #     prompt_prefix="",
    #     prompt_template="",
    #     instruction_prompt="",
    #     prompt_label_mapping=dict(a=""), #TODO: INSERT LABEL MAPPING
    #     labels=SENT_LABELS
    # ),
    FR: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=SENT_LABELS,
    ),
    IS: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=SENT_LABELS,
    ),
    IT: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=SENT_LABELS,
    ),
    NB: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=SENT_LABELS,
    ),
    NL: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=SENT_LABELS,
    ),
    NN: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=SENT_LABELS,
    ),
    NO: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=SENT_LABELS,
    ),
    SV: PromptTemplate(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(a=""),  # TODO: INSERT LABEL MAPPING
        labels=SENT_LABELS,
    ),
}

### SUMMARIZATION TEMPLATES ###

SUMM_TEMPLATE = {
    DA: PromptTemplate(prompt_prefix="", prompt_template="", instruction_prompt=""),
    DE: PromptTemplate(prompt_prefix="", prompt_template="", instruction_prompt=""),
    EN: PromptTemplate(prompt_prefix="", prompt_template="", instruction_prompt=""),
    # FO: PromptTemplate(
    #     prompt_prefix="",
    #     prompt_template="",
    #     instruction_prompt=""
    # ),
    FR: PromptTemplate(prompt_prefix="", prompt_template="", instruction_prompt=""),
    IS: PromptTemplate(prompt_prefix="", prompt_template="", instruction_prompt=""),
    IT: PromptTemplate(prompt_prefix="", prompt_template="", instruction_prompt=""),
    NB: PromptTemplate(prompt_prefix="", prompt_template="", instruction_prompt=""),
    NL: PromptTemplate(prompt_prefix="", prompt_template="", instruction_prompt=""),
    NN: PromptTemplate(prompt_prefix="", prompt_template="", instruction_prompt=""),
    NO: PromptTemplate(prompt_prefix="", prompt_template="", instruction_prompt=""),
    SV: PromptTemplate(prompt_prefix="", prompt_template="", instruction_prompt=""),
}
