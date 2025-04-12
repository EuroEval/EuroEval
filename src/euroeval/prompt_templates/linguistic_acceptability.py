"""Templates for the Linguistic Acceptability task."""

from ..data_models import PromptConfig
from ..languages import DA, DE, EN, ES, FO, FR, IS, IT, NB, NL, NN, NO, SV

LA_TEMPLATES = {
    DA: PromptConfig(
        prompt_label_mapping=dict(correct="ja", incorrect="nej"),
        prompt_prefix="Følgende er sætninger og om de er grammatisk korrekte.",
        prompt_template="Sætning: {text}\nGrammatisk korrekt: {label}",
        instruction_prompt="Sætning: {text}\n\nBestem om sætningen er grammatisk "
        "korrekt eller ej. Svar med 'ja', hvis sætningen er korrekt, og 'nej', "
        "hvis den ikke er, og intet andet.",
    ),
    DE: PromptConfig(
        prompt_label_mapping=dict(correct="ja", incorrect="nein"),
        prompt_prefix="Die folgenden Sätze und ob sie grammatikalisch korrekt sind.",
        prompt_template="Satz: {text}\nGrammatikalisch richtig: {label}",
        instruction_prompt="Satz: {text}\n\nBestimmen Sie, ob der Satz "
        "grammatikalisch korrekt ist oder nicht. Antworten Sie mit 'ja', wenn der "
        "Satz korrekt ist und 'nein', wenn er es nicht ist, und nichts anderes.",
    ),
    EN: PromptConfig(
        prompt_label_mapping=dict(correct="yes", incorrect="no"),
        prompt_prefix="The following are sentences and whether they are "
        "grammatically correct.",
        prompt_template="Sentence: {text}\nGrammatically correct: {label}",
        instruction_prompt="Sentence: {text}\n\nDetermine whether the sentence is "
        "grammatically correct or not. Reply with 'yes' if the sentence is "
        "correct and 'no' if it is not, and nothing else.",
    ),
    ES: PromptConfig(
        prompt_label_mapping=dict(correct="sí", incorrect="no"),
        prompt_prefix="Lo siguiente son textos y si son gramaticalmente correctos.",
        prompt_template="Texto: {text}\nGramaticalmente correcto: {label}",
        instruction_prompt="Texto: {text}\n\nDetermina si el texto es "
        "gramaticalmente correcto o no. Responde con 'sí' si el texto es "
        "correcto, y 'no' si no lo es.",
    ),
    FO: PromptConfig(
        prompt_label_mapping=dict(correct="ja", incorrect="nei"),
        prompt_prefix="Hetta eru nakrir setningar og um teir eru mállæruliga rættir.",
        prompt_template="Setningur: {text}\nMállæruliga rættur: {label}",
        instruction_prompt="Setningur: {text}\n\nGreinið hvort setningurin er "
        "mállæruliga rættur ella ikki. Svarið skal vera 'ja' um setningurin er "
        "rættur og 'nei' um hann ikki er, og einki annað.",
    ),
    FR: PromptConfig(
        prompt_label_mapping=dict(correct="oui", incorrect="non"),
        prompt_prefix="Les phrases suivantes indiquent si elles sont "
        "grammaticalement correctes.",
        prompt_template="Phrase : {text}\nCorrect du point de vue grammatical: {label}",
        instruction_prompt="Phrase: {text}\n\nDéterminez si la phrase est "
        "grammaticalement correcte ou non. Répondez par 'oui' si la phrase est "
        "correcte et par 'non' si elle ne l'est pas, et rien d'autre.",
    ),
    IS: PromptConfig(
        prompt_label_mapping=dict(correct="já", incorrect="nei"),
        prompt_prefix="Eftirfarandi eru setningar og hvort þær eru málfræðilega "
        "réttar.",
        prompt_template="Setning: {text}\nMálfræðilega rétt: {label}",
        instruction_prompt="Setning: {text}\n\nGreinið hvort setningin er "
        "málfræðilega rétt eða ekki. Svarið skal vera 'já' ef setningin er rétt "
        "og 'nei' ef hún er ekki, og engu öðru.",
    ),
    IT: PromptConfig(
        prompt_label_mapping=dict(correct="si", incorrect="no"),
        prompt_prefix="Di seguito sono riportate le frasi e la loro correttezza "
        "grammaticale.",
        prompt_template="Frase : {text}\nGrammaticalmente corretto : {label}",
        instruction_prompt="Frase: {text}\n\nStabilite se la frase è "
        "grammaticalmente corretta o meno. Rispondete con 'si' se la frase è "
        "corretta e con 'no' se non lo è, e nient'altro.",
    ),
    NB: PromptConfig(
        prompt_label_mapping=dict(correct="ja", incorrect="nei"),
        prompt_prefix="Følgende er setninger og hvorvidt de er grammatisk korrekte.",
        prompt_template="Setning: {text}\nGrammatisk korrekt: {label}",
        instruction_prompt="Setning: {text}\n\nBestem om setningen er grammatisk "
        "korrekt eller ikke. Svar med 'ja' hvis setningen er korrekt og 'nei' "
        "hvis den ikke er, og ikke noe annet.",
    ),
    NL: PromptConfig(
        prompt_label_mapping=dict(correct="ja", incorrect="nee"),
        prompt_prefix="Hieronder staan zinnen en of ze grammaticaal correct zijn.",
        prompt_template="Zin: {text}\nGrammaticaal correct: {label}",
        instruction_prompt="Zin: {text}\n\nBepaal of de zin grammaticaal correct "
        "is of niet. Antwoord met 'ja' als de zin correct is en 'nee' als dat "
        "niet het geval is, en niets anders.",
    ),
    NN: PromptConfig(
        prompt_label_mapping=dict(correct="ja", incorrect="nei"),
        prompt_prefix="Følgende er setninger og hvorvidt de er grammatisk korrekte.",
        prompt_template="Setning: {text}\nGrammatisk korrekt: {label}",
        instruction_prompt="Setning: {text}\n\nBestem om setningen er grammatisk "
        "korrekt eller ikke. Svar med 'ja' hvis setningen er korrekt og 'nei' "
        "hvis den ikke er, og ikke noe annet.",
    ),
    NO: PromptConfig(
        prompt_label_mapping=dict(correct="ja", incorrect="nei"),
        prompt_prefix="Følgende er setninger og hvorvidt de er grammatisk korrekte.",
        prompt_template="Setning: {text}\nGrammatisk korrekt: {label}",
        instruction_prompt="Setning: {text}\n\nBestem om setningen er grammatisk "
        "korrekt eller ikke. Svar med 'ja' hvis setningen er korrekt og 'nei' "
        "hvis den ikke er, og ikke noe annet.",
    ),
    SV: PromptConfig(
        prompt_label_mapping=dict(correct="ja", incorrect="nej"),
        prompt_prefix="Följande är meningar och huruvida de är grammatiskt korrekta.",
        prompt_template="Mening: {text}\nGrammatisk korrekt: {label}",
        instruction_prompt="Mening: {text}\n\nBestäm om meningen är grammatiskt "
        "korrekt eller inte. Svara med 'ja' om meningen är korrekt och 'nej' "
        "om den inte är, och inget annat.",
    ),
}
