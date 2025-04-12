"""Templates for the Summarization task."""

from ..data_models import PromptConfig
from ..languages import DA, DE, EN, ES, FO, FR, IS, IT, NB, NL, NN, NO, SV

SUMM_TEMPLATES = {
    DA: PromptConfig(
        prompt_prefix="Følgende er dokumenter med tilhørende resuméer.",
        prompt_template="Dokument: {text}\nResumé: {target_text}",
        instruction_prompt="Dokument: {text}\n\nSkriv et resumé af ovenstående "
        "dokument.",
        prompt_label_mapping=dict(),
    ),
    DE: PromptConfig(
        prompt_prefix="Nachstehend finden Sie Dokumente mit zugehörigen "
        "Zusammenfassungen.",
        prompt_template="Dokument: {text}\nZusammenfassung: {target_text}",
        instruction_prompt="Nachrichtenartikel: {text}\n\nSchreiben Sie eine "
        "Zusammenfassung des oben genannten Dokuments.",
        prompt_label_mapping=dict(),
    ),
    EN: PromptConfig(
        prompt_prefix="The following are documents with accompanying summaries.",
        prompt_template="Document: {text}\nSummary: {target_text}",
        instruction_prompt="Document: {text}\n\nWrite a summary of the above document.",
        prompt_label_mapping=dict(),
    ),
    ES: PromptConfig(
        prompt_prefix="A continuación se presentan documentos con resúmenes adjuntos.",
        prompt_template="Documento: {text}\nResumen: {target_text}",
        instruction_prompt="Documento: {text}\n\nEscriba un resumen del "
        "documento anterior.",
        prompt_label_mapping=dict(),
    ),
    FO: PromptConfig(
        prompt_prefix="",
        prompt_template="",
        instruction_prompt="",
        prompt_label_mapping=dict(),
    ),
    FR: PromptConfig(
        prompt_prefix="Les documents suivants sont accompagnés d'un résumé.",
        prompt_template="Document: {text}\nRésumé: {target_text}",
        instruction_prompt="Document: {text}\n\nRédigez un résumé du "
        "document ci-dessus.",
        prompt_label_mapping=dict(),
    ),
    IS: PromptConfig(
        prompt_prefix="Eftirfarandi eru skjöl með meðfylgjandi samantektum.",
        prompt_template="Skjal: {text}\nSamantekt: {target_text}",
        instruction_prompt="Skjal: {text}\n\nSkrifaðu samantekt á ofangreindu skjali.",
        prompt_label_mapping=dict(),
    ),
    IT: PromptConfig(
        prompt_prefix="Di seguito sono riportati i documenti con le relative sintesi.",
        prompt_template="Documento: {text}\nSintesi: {target_text}",
        instruction_prompt="Documento: {text}\n\nScrivete una sintesi del "
        "documento di cui sopra.",
        prompt_label_mapping=dict(),
    ),
    NB: PromptConfig(
        prompt_prefix="Nedenfor følger dokumenter med tilhørende sammendrag.",
        prompt_template="Dokument: {text}\nSammendrag: {target_text}",
        instruction_prompt="Dokument: {text}\n\nSkriv et sammendrag av "
        "dokumentet ovenfor.",
        prompt_label_mapping=dict(),
    ),
    NL: PromptConfig(
        prompt_prefix="Hieronder volgen documenten met bijbehorende samenvattingen.",
        prompt_template="Document: {text}\nSamenvatting: {target_text}",
        instruction_prompt="Document: {text}\n\nSchrijf een samenvatting van het "
        "bovenstaande document.",
        prompt_label_mapping=dict(),
    ),
    NN: PromptConfig(
        prompt_prefix="Nedenfor følger dokumenter med tilhørende sammendrag.",
        prompt_template="Dokument: {text}\nSammendrag: {target_text}",
        instruction_prompt="Dokument: {text}\n\nSkriv et sammendrag av "
        "dokumentet ovenfor.",
        prompt_label_mapping=dict(),
    ),
    NO: PromptConfig(
        prompt_prefix="Nedenfor følger dokumenter med tilhørende sammendrag.",
        prompt_template="Dokument: {text}\nSammendrag: {target_text}",
        instruction_prompt="Dokument: {text}\n\nSkriv et sammendrag av "
        "dokumentet ovenfor.",
        prompt_label_mapping=dict(),
    ),
    SV: PromptConfig(
        prompt_prefix="Nedan följer dokument med tillhörande sammanfattningar.",
        prompt_template="Dokument: {text}\nSammanfattning: {target_text}",
        instruction_prompt="Dokument: {text}\n\nSkriv en sammanfattning av "
        "ovanstående dokument.",
        prompt_label_mapping=dict(),
    ),
}
