"""Templates for the Linguistic Acceptability task."""

from ..data_models import PromptConfig
from ..languages import DA, DE, EN, ES, FI, FO, FR, IS, IT, NB, NL, NN, NO, SV

LA_TEMPLATES = {
    DA: PromptConfig(
        default_prompt_label_mapping=dict(correct="ja", incorrect="nej"),
        default_prompt_prefix="Følgende er sætninger og om de er grammatisk korrekte.",
        default_prompt_template="Sætning: {text}\nGrammatisk korrekt: {label}",
        default_instruction_prompt="Sætning: {text}\n\nBestem om sætningen er "
        "grammatisk korrekt eller ej. Svar kun med {labels_str}, og intet andet.",
    ),
    DE: PromptConfig(
        default_prompt_label_mapping=dict(correct="ja", incorrect="nein"),
        default_prompt_prefix="Die folgenden Sätze und ob sie grammatikalisch korrekt "
        "sind.",
        default_prompt_template="Satz: {text}\nGrammatikalisch richtig: {label}",
        default_instruction_prompt="Satz: {text}\n\nBestimmen Sie, ob der Satz "
        "grammatikalisch korrekt ist oder nicht. Antworten Sie mit {labels_str}, und "
        "nichts anderes.",
    ),
    EN: PromptConfig(
        default_prompt_label_mapping=dict(correct="yes", incorrect="no"),
        default_prompt_prefix="The following are sentences and whether they are "
        "grammatically correct.",
        default_prompt_template="Sentence: {text}\nGrammatically correct: {label}",
        default_instruction_prompt="Sentence: {text}\n\nDetermine whether the sentence "
        "is grammatically correct or not. Answer with {labels_str}, and nothing else.",
    ),
    ES: PromptConfig(
        default_prompt_label_mapping=dict(correct="sí", incorrect="no"),
        default_prompt_prefix="Lo siguiente son textos y si son gramaticalmente "
        "correctos.",
        default_prompt_template="Texto: {text}\nGramaticalmente correcto: {label}",
        default_instruction_prompt="Texto: {text}\n\nDetermina si el texto es "
        "gramaticalmente correcto o no. Responde con {labels_str}, y nada más.",
    ),
    FI: PromptConfig(
        default_prompt_label_mapping=dict(correct="kyllä", incorrect="ei"),
        default_prompt_prefix="Seuraavat ovat lauseita ja ovatko ne "
        "kieliopillisesti oikein.",
        default_prompt_template="Lause: {text}\nKieliopillisesti oikein: {label}",
        default_instruction_prompt="Lause: {text}\n\nMääritä onko lause "
        "oikein vai ei. Vastaa {labels_str}, ja ei mitään muuta.",
    ),
    FO: PromptConfig(
        default_prompt_label_mapping=dict(correct="ja", incorrect="nei"),
        default_prompt_prefix="Hetta eru nakrir setningar og um teir eru mállæruliga "
        "rættir.",
        default_prompt_template="Setningur: {text}\nMállæruliga rættur: {label}",
        default_instruction_prompt="Setningur: {text}\n\nGreinið hvort setningurin er "
        "mállæruliga rættur ella ikki. Svara við {labels_str}, og einki annað.",
    ),
    FR: PromptConfig(
        default_prompt_label_mapping=dict(correct="oui", incorrect="non"),
        default_prompt_prefix="Les phrases suivantes indiquent si elles sont "
        "grammaticalement correctes.",
        default_prompt_template="Phrase : {text}\nCorrect du point de vue grammatical: "
        "{label}",
        default_instruction_prompt="Phrase: {text}\n\nDéterminez si la phrase est "
        "grammaticalement correcte ou non. Répondez par {labels_str}, et rien d'autre.",
    ),
    IS: PromptConfig(
        default_prompt_label_mapping=dict(correct="já", incorrect="nei"),
        default_prompt_prefix="Eftirfarandi eru setningar og hvort þær eru "
        "málfræðilega réttar.",
        default_prompt_template="Setning: {text}\nMálfræðilega rétt: {label}",
        default_instruction_prompt="Setning: {text}\n\nGreinið hvort setningin er "
        "málfræðilega rétt eða ekki. Svaraðu með {labels_str}, og ekkert annað.",
    ),
    IT: PromptConfig(
        default_prompt_label_mapping=dict(correct="si", incorrect="no"),
        default_prompt_prefix="Di seguito sono riportate le frasi e la loro "
        "correttezza grammaticale.",
        default_prompt_template="Frase : {text}\nGrammaticalmente corretto : {label}",
        default_instruction_prompt="Frase: {text}\n\nStabilite se la frase è "
        "grammaticalmente corretta o meno. Rispondere con {labels_str}, e nient'altro.",
    ),
    NB: PromptConfig(
        default_prompt_label_mapping=dict(correct="ja", incorrect="nei"),
        default_prompt_prefix="Følgende er setninger og hvorvidt de er grammatisk "
        "korrekte.",
        default_prompt_template="Setning: {text}\nGrammatisk korrekt: {label}",
        default_instruction_prompt="Setning: {text}\n\nBestem om setningen er "
        "grammatisk korrekt eller ikke. Svar med {labels_str}, og ikke noe annet.",
    ),
    NL: PromptConfig(
        default_prompt_label_mapping=dict(correct="ja", incorrect="nee"),
        default_prompt_prefix="Hieronder staan zinnen en of ze grammaticaal correct "
        "zijn.",
        default_prompt_template="Zin: {text}\nGrammaticaal correct: {label}",
        default_instruction_prompt="Zin: {text}\n\nBepaal of de zin grammaticaal "
        "correct is of niet. Antwoord met {labels_str}, en verder niets.",
    ),
    NN: PromptConfig(
        default_prompt_label_mapping=dict(correct="ja", incorrect="nei"),
        default_prompt_prefix="Følgende er setninger og hvorvidt de er grammatisk "
        "korrekte.",
        default_prompt_template="Setning: {text}\nGrammatisk korrekt: {label}",
        default_instruction_prompt="Setning: {text}\n\nBestem om setningen er "
        "grammatisk korrekt eller ikke. Svar med {labels_str}, og ikke noe annet.",
    ),
    NO: PromptConfig(
        default_prompt_label_mapping=dict(correct="ja", incorrect="nei"),
        default_prompt_prefix="Følgende er setninger og hvorvidt de er grammatisk "
        "korrekte.",
        default_prompt_template="Setning: {text}\nGrammatisk korrekt: {label}",
        default_instruction_prompt="Setning: {text}\n\nBestem om setningen er "
        "grammatisk korrekt eller ikke. Svar med {labels_str}, og ikke noe annet.",
    ),
    SV: PromptConfig(
        default_prompt_label_mapping=dict(correct="ja", incorrect="nej"),
        default_prompt_prefix="Följande är meningar och huruvida de är grammatiskt "
        "korrekta.",
        default_prompt_template="Mening: {text}\nGrammatisk korrekt: {label}",
        default_instruction_prompt="Mening: {text}\n\nBestäm om meningen är "
        "grammatiskt korrekt eller inte. Svara med {labels_str}, och inget annat.",
    ),
}
