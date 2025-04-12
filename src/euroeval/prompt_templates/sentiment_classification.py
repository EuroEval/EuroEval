"""Templates for the Sentiment Analysis task."""

from ..data_models import PromptConfig
from ..languages import DA, DE, EN, ES, FO, FR, IS, IT, NB, NL, NN, NO, SV

SENT_TEMPLATES = {
    DA: PromptConfig(
        prompt_label_mapping=dict(
            positive="positiv", neutral="neutral", negative="negativ"
        ),
        prompt_prefix="Følgende er dokumenter og deres sentiment, som kan være "
        "{labels_str}.",
        prompt_template="Dokument: {text}\nSentiment: {label}",
        instruction_prompt="Dokument: {text}\n\nKlassificer sentimentet i "
        "dokumentet. Svar kun med {labels_str}, og intet andet.",
    ),
    DE: PromptConfig(
        prompt_label_mapping=dict(
            positive="positiv", neutral="neutral", negative="negativ"
        ),
        prompt_prefix="Nachfolgend finden Sie Dokumente und ihre Bewertung, die "
        "{labels_str} sein kann.",
        prompt_template="Dokument: {text}\nStimmung: {label}",
        instruction_prompt="Dokument: {text}\n\nKlassifizieren Sie die Stimmung im "
        "Dokument. Antworten Sie mit {labels_str}, und nichts anderes.",
    ),
    EN: PromptConfig(
        prompt_label_mapping=dict(
            positive="positive", neutral="neutral", negative="negative"
        ),
        prompt_prefix="The following are documents and their sentiment, which can "
        "be {labels_str}.",
        prompt_template="Document: {text}\nSentiment: {label}",
        instruction_prompt="Document: {text}\n\nClassify the sentiment in the "
        "document. Answer with {labels_str}, and nothing else.",
    ),
    ES: PromptConfig(
        prompt_label_mapping=dict(
            positive="positivo", neutral="neutral", negative="negativo"
        ),
        prompt_prefix="A continuación se muestran los documentos y su "
        "sentimiento, que puede ser {labels_str}.",
        prompt_template="Documento: {text}\nSentimiento: {label}",
        instruction_prompt="Documento: {text}\n\nClasifica el sentimiento del "
        "documento. Responde con {labels_str}, y nada más.",
    ),
    FO: PromptConfig(
        prompt_label_mapping=dict(
            positive="positivt", neutral="neutralt", negative="negativt"
        ),
        prompt_prefix="Niðanfyri eru skjøl og teirra kenslur, sum kunnu vera "
        "{labels_str}.",
        prompt_template="Skjal: {text}\nKensla: {label}",
        instruction_prompt="Skjal: {text}\n\nFlokka kensluna í skjalinum. Svara "
        "við {labels_str}, og einki annað.",
    ),
    FR: PromptConfig(
        prompt_label_mapping=dict(
            positive="positif", neutral="neutre", negative="négatif"
        ),
        prompt_prefix="Les documents suivants sont accompagnés de leur sentiment, "
        "qui peut être {labels_str}.",
        prompt_template="Document: {text}\nSentiment: {label}",
        instruction_prompt="Document: {text}\n\nClassez le sentiment dans le "
        "document. Répondez par {labels_str}, et rien d'autre.",
    ),
    IS: PromptConfig(
        prompt_label_mapping=dict(
            positive="jákvætt", neutral="hlutlaust", negative="neikvætt"
        ),
        prompt_prefix="Eftirfarandi eru skjöl og viðhorf þeirra, sem geta verið "
        "{labels_str}.",
        prompt_template="Skjal: {text}\nViðhorf: {label}",
        instruction_prompt="Skjal: {text}\n\nFlokkaðu viðhorfið í skjalinu. "
        "Svaraðu með {labels_str}, og ekkert annað.",
    ),
    IT: PromptConfig(
        prompt_label_mapping=dict(
            positive="positivo", neutral="neutro", negative="negativo"
        ),
        prompt_prefix="Di seguito sono riportati i documenti e il loro sentiment, "
        "che può essere {labels_str}.",
        prompt_template="Documento: {text}\nSentimento: {label}",
        instruction_prompt="Documento: {text}\n\nClassificare il sentiment del "
        "documento. Rispondere con {labels_str}, e nient'altro.",
    ),
    NB: PromptConfig(
        prompt_label_mapping=dict(
            positive="positiv", neutral="nøytral", negative="negativ"
        ),
        prompt_prefix="Her følger dokumenter og deres sentiment, som kan være "
        "{labels_str}",
        prompt_template="Dokument: {text}\nSentiment: {label}",
        instruction_prompt="Dokument: {text}\n\nKlassifiser følelsen i teksten. "
        "Svar med {labels_str}, og ikke noe annet.",
    ),
    NL: PromptConfig(
        prompt_label_mapping=dict(
            positive="positief", neutral="neutraal", negative="negatief"
        ),
        prompt_prefix="Hieronder volgen documenten en hun sentiment, dat "
        "{labels_str} kan zijn.",
        prompt_template="Document: {text}\nSentiment: {label}",
        instruction_prompt="Document: {text}\n\nClassificeer het sentiment in het "
        "document. Antwoord met {labels_str}, en verder niets.",
    ),
    NN: PromptConfig(
        prompt_label_mapping=dict(
            positive="positiv", neutral="nøytral", negative="negativ"
        ),
        prompt_prefix="Her følger dokumenter og deres sentiment, som kan være "
        "{labels_str}",
        prompt_template="Dokument: {text}\nSentiment: {label}",
        instruction_prompt="Dokument: {text}\n\nKlassifiser følelsen i teksten. "
        "Svar med {labels_str}, og ikke noe annet.",
    ),
    NO: PromptConfig(
        prompt_label_mapping=dict(
            positive="positiv", neutral="nøytral", negative="negativ"
        ),
        prompt_prefix="Her følger dokumenter og deres sentiment, som kan være "
        "{labels_str}",
        prompt_template="Dokument: {text}\nSentiment: {label}",
        instruction_prompt="Dokument: {text}\n\nKlassifiser følelsen i teksten. "
        "Svar med {labels_str}, og ikke noe annet.",
    ),
    SV: PromptConfig(
        prompt_label_mapping=dict(
            positive="positiv", neutral="neutral", negative="negativ"
        ),
        prompt_prefix="Nedan följer dokument och deras sentiment, som kan vara "
        "{labels_str}.",
        prompt_template="Dokument: {text}\nSentiment: {label}",
        instruction_prompt="Dokument: {text}\n\nKlassificera känslan i dokumentet. "
        "Svara med {labels_str}, och inget annat.",
    ),
}
