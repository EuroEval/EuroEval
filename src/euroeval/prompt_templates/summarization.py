"""Templates for the Summarization task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import (
    CZECH,
    DANISH,
    DUTCH,
    ENGLISH,
    ESTONIAN,
    FINNISH,
    FRENCH,
    GERMAN,
    GREEK,
    ICELANDIC,
    ITALIAN,
    LATVIAN,
    LITHUANIAN,
    NORWEGIAN,
    NORWEGIAN_BOKMÅL,
    NORWEGIAN_NYNORSK,
    POLISH,
    PORTUGUESE,
    SERBIAN,
    SPANISH,
    SWEDISH,
    UKRAINIAN,
)

if t.TYPE_CHECKING:
    from ..languages import Language

# TODO: Missing Faroese
SUMM_TEMPLATES: dict["Language", PromptConfig] = {
    CZECH: PromptConfig(
        default_prompt_prefix=("Následující jsou dokumenty s přiloženými souhrny."),
        default_prompt_template=("Dokument: {text}\nSouhrn: {target_text}"),
        default_instruction_prompt=(
            "Dokument: {text}\n\nNapište souhrn výše uvedeného dokumentu."
        ),
        default_prompt_label_mapping=dict(),
    ),
    DANISH: PromptConfig(
        default_prompt_prefix="Følgende er dokumenter med tilhørende resuméer.",
        default_prompt_template="Dokument: {text}\nResumé: {target_text}",
        default_instruction_prompt="Dokument: {text}\n\nSkriv et resumé af ovenstående "
        "dokument.",
        default_prompt_label_mapping=dict(),
    ),
    GERMAN: PromptConfig(
        default_prompt_prefix="Nachstehend finden Sie Dokumente mit zugehörigen "
        "Zusammenfassungen.",
        default_prompt_template="Dokument: {text}\nZusammenfassung: {target_text}",
        default_instruction_prompt="Nachrichtenartikel: {text}\n\nSchreiben Sie eine "
        "Zusammenfassung des oben genannten Dokuments.",
        default_prompt_label_mapping=dict(),
    ),
    GREEK: PromptConfig(
        default_prompt_prefix="Ακολουθούν έγγραφα με συνοδευτικές περιλήψεις.",
        default_prompt_template="Έγγραφο: {text}\nΠερίληψη: {target_text}",
        default_instruction_prompt="Έγγραφο: {text}\n\nΓράψτε μια περίληψη του "
        "παραπάνω εγγράφου.",
        default_prompt_label_mapping=dict(),
    ),
    ENGLISH: PromptConfig(
        default_prompt_prefix="The following are documents with accompanying "
        "summaries.",
        default_prompt_template="Document: {text}\nSummary: {target_text}",
        default_instruction_prompt="Document: {text}\n\nWrite a summary of the above "
        "document.",
        default_prompt_label_mapping=dict(),
    ),
    SPANISH: PromptConfig(
        default_prompt_prefix="A continuación se presentan documentos con resúmenes "
        "adjuntos.",
        default_prompt_template="Documento: {text}\nResumen: {target_text}",
        default_instruction_prompt="Documento: {text}\n\n",
        default_prompt_label_mapping=dict(),
    ),
    ESTONIAN: PromptConfig(
        default_prompt_prefix="Allpool on dokumendid koos kokkuvõtetega.",
        default_prompt_template="Dokument: {text}\nKokkuvõte: {target_text}",
        default_instruction_prompt="Dokument: {text}\n\nKoosta ülaltoodud dokumendi "
        "kokkuvõte.",
        default_prompt_label_mapping=dict(),
    ),
    PORTUGUESE: PromptConfig(
        default_prompt_prefix="Abaixo encontras documentos com resumos associados.",
        default_prompt_template="Documento: {text}\nResumo: {target_text}",
        default_instruction_prompt="Documento: {text}\n\nEscreve um resumo do "
        "documento anterior.",
        default_prompt_label_mapping=dict(),
    ),
    FINNISH: PromptConfig(
        default_prompt_prefix="Seuraavassa on artikkeleita ja niihin liittyviä "
        "tiivistelmiä.",
        default_prompt_template="Uutisartikkeli: {text}\nTiivistelmä: {target_text}",
        default_instruction_prompt="Uutisartikkeli: {text}\n\nKirjoita tiivistelmä "
        "yllä olevasta artikkelista.",
        default_prompt_label_mapping=dict(),
    ),
    FRENCH: PromptConfig(
        default_prompt_prefix="Les documents suivants sont accompagnés d'un résumé.",
        default_prompt_template="Document: {text}\nRésumé: {target_text}",
        default_instruction_prompt="Document: {text}\n\nRédigez un résumé du "
        "document ci-dessus.",
        default_prompt_label_mapping=dict(),
    ),
    LATVIAN: PromptConfig(
        default_prompt_prefix="Tālāk ir dokumenti ar pievienotām kopsavilkumiem.",
        default_prompt_template="Dokuments: {text}\nKopsavilkums: {target_text}",
        default_instruction_prompt=(
            "Dokuments: {text}\n\n"
            "Uzrakstiet kopsavilkumu par iepriekš minēto dokumentu."
        ),
        default_prompt_label_mapping=dict(),
    ),
    LITHUANIAN: PromptConfig(
        default_prompt_prefix=(
            "Žemiau pateikiami dokumentai su pridėtomis santraukomis."
        ),
        default_prompt_template=("Dokumentas: {text}\nSantrauka: {target_text}"),
        default_instruction_prompt=(
            "Dokumentas: {text}\n\nParašykite aukščiau pateikto dokumento santrauką."
        ),
        default_prompt_label_mapping=dict(),
    ),
    ITALIAN: PromptConfig(
        default_prompt_prefix="Di seguito sono riportati i documenti con le relative "
        "sintesi.",
        default_prompt_template="Documento: {text}\nSintesi: {target_text}",
        default_instruction_prompt="Documento: {text}\n\nScrivete una sintesi del "
        "documento di cui sopra.",
        default_prompt_label_mapping=dict(),
    ),
    ICELANDIC: PromptConfig(
        default_prompt_prefix="Eftirfarandi eru skjöl með meðfylgjandi samantektum.",
        default_prompt_template="Skjal: {text}\nSamantekt: {target_text}",
        default_instruction_prompt="Skjal: {text}\n\nSkrifaðu samantekt á ofangreindu "
        "skjali.",
        default_prompt_label_mapping=dict(),
    ),
    NORWEGIAN_BOKMÅL: PromptConfig(
        default_prompt_prefix="Nedenfor følger dokumenter med tilhørende sammendrag.",
        default_prompt_template="Dokument: {text}\nSammendrag: {target_text}",
        default_instruction_prompt="Dokument: {text}\n\nSkriv et sammendrag av "
        "dokumentet ovenfor.",
        default_prompt_label_mapping=dict(),
    ),
    DUTCH: PromptConfig(
        default_prompt_prefix="Hieronder volgen documenten met bijbehorende "
        "samenvattingen.",
        default_prompt_template="Document: {text}\nSamenvatting: {target_text}",
        default_instruction_prompt="Document: {text}\n\nSchrijf een samenvatting van "
        "het bovenstaande document.",
        default_prompt_label_mapping=dict(),
    ),
    NORWEGIAN_NYNORSK: PromptConfig(
        default_prompt_prefix="Nedenfor følger dokumenter med tilhørende sammendrag.",
        default_prompt_template="Dokument: {text}\nSammendrag: {target_text}",
        default_instruction_prompt="Dokument: {text}\n\nSkriv et sammendrag av "
        "dokumentet ovenfor.",
        default_prompt_label_mapping=dict(),
    ),
    NORWEGIAN: PromptConfig(
        default_prompt_prefix="Nedenfor følger dokumenter med tilhørende sammendrag.",
        default_prompt_template="Dokument: {text}\nSammendrag: {target_text}",
        default_instruction_prompt="Dokument: {text}\n\nSkriv et sammendrag av "
        "dokumentet ovenfor.",
        default_prompt_label_mapping=dict(),
    ),
    POLISH: PromptConfig(
        default_prompt_prefix="Poniżej znajdują się artykuły z towarzyszącymi "
        "im streszczeniami.",
        default_prompt_template="Artykuł: {text}\nStreszczenie: {target_text}",
        default_instruction_prompt="Artykuł: {text}\n\nNapisz streszczenie "
        "powyższego artykułu.",
        default_prompt_label_mapping=dict(),
    ),
    SERBIAN: PromptConfig(
        default_prompt_prefix="Slede dokumenti sa odgovarajućim sažecima.",
        default_prompt_template="Dokument: {text}\nSažetak: {target_text}",
        default_instruction_prompt="Dokument: {text}\n\nNapišite sažetak "
        "gorenavedenog dokumenta.",
        default_prompt_label_mapping=dict(),
    ),
    SWEDISH: PromptConfig(
        default_prompt_prefix="Nedan följer dokument med tillhörande sammanfattningar.",
        default_prompt_template="Dokument: {text}\nSammanfattning: {target_text}",
        default_instruction_prompt="Dokument: {text}\n\nSkriv en sammanfattning av "
        "ovanstående dokument.",
        default_prompt_label_mapping=dict(),
    ),
    UKRAINIAN: PromptConfig(
        default_prompt_prefix="Нижче наведено документи з супровідними резюме.",
        default_prompt_template="Документ: {text}\nРезюме: {target_text}",
        default_instruction_prompt=(
            "Документ: {text}\n\nНапишіть резюме наведеного вище документа."
        ),
        default_prompt_label_mapping=dict(),
    ),
}
