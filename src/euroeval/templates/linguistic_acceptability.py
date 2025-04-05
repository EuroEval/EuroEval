"""Templates for the Linguistic Acceptability task."""

from dataclasses import asdict

from ..types import TemplateDict
from .base import BasePromptConfig, PromptConfig
from .constants import LA_LABEL_MAPPING

LA_DEFAULTS = BasePromptConfig(
    labels=["incorrect", "correct"], num_few_shot_examples=12, max_generated_tokens=5
)
LA_DEFAULTS_DICT = asdict(LA_DEFAULTS)
LA_DEFAULTS_DICT.pop("prompt_label_mapping")


def get_label_mapping(language_code: str) -> dict[str, str]:
    """Get the translations for all labels in the specified language.

    Args:
        language_code: The language code for the target language.

    Returns:
        The translated labels for the specified language.
    """
    try:
        return {
            f"{label}": LA_LABEL_MAPPING[label][language_code]
            for label in LA_LABEL_MAPPING.keys()
        }
    except KeyError:
        raise KeyError(f"No LA label mapping found for language '{language_code}'.")


LA_TEMPLATES: TemplateDict = {
    "da": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("da"),
        prompt_prefix="Følgende er sætninger og om de er grammatisk korrekte.",
        prompt_template="Sætning: {text}\nGrammatisk korrekt: {label}",
        instruction_prompt="Sætning: {text}\n\nBestem om sætningen er grammatisk "
        "korrekt eller ej. Svar med 'ja', hvis sætningen er korrekt, og 'nej', hvis "
        "den ikke er, og intet andet.",
    ),
    "de": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("de"),
        prompt_prefix="Die folgenden Sätze und ob sie grammatikalisch korrekt sind.",
        prompt_template="Satz: {text}\nGrammatikalisch richtig: {label}",
        instruction_prompt="Satz: {text}\n\nBestimmen Sie, ob der Satz grammatikalisch "
        "korrekt ist oder nicht. Antworten Sie mit 'ja', wenn der Satz korrekt ist und "
        "'nein', wenn er es nicht ist, und nichts anderes.",
    ),
    "en": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("en"),
        prompt_prefix="The following are sentences and whether they are grammatically "
        "correct.",
        prompt_template="Sentence: {text}\nGrammatically correct: {label}",
        instruction_prompt="Sentence: {text}\n\nDetermine whether the sentence is "
        "grammatically correct or not. Reply with 'yes' if the sentence is correct and "
        "'no' if it is not, and nothing else.",
    ),
    "es": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("es"),
        prompt_prefix="Lo siguiente son textos y si son gramaticalmente correctos.",
        prompt_template="Texto: {text}\nGramaticalmente correcto: {label}",
        instruction_prompt="Texto: {text}\n\nDetermina si el texto es gramaticalmente "
        "correcto o no. Responde con 'sí' si el texto es correcto, y 'no' si no lo es.",
    ),
    "fo": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("fo"),
        prompt_prefix="Hetta eru nakrir setningar og um teir eru mállæruliga rættir.",
        prompt_template="Setningur: {text}\nMállæruliga rættur: {label}",
        instruction_prompt="Setningur: {text}\n\nGreinið hvort setningurin er "
        "mállæruliga rættur ella ikki. Svarið skal vera 'ja' um setningurin er rættur "
        "og 'nei' um hann ikki er, og einki annað.",
    ),
    "fr": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("fr"),
        prompt_prefix="Les phrases suivantes indiquent si elles sont grammaticalement "
        "correctes.",
        prompt_template="Phrase : {text}\nCorrect du point de vue grammatical: {label}",
        instruction_prompt="Phrase: {text}\n\nDéterminez si la phrase est "
        "grammaticalement correcte ou non. Répondez par 'oui' si la phrase est "
        "correcte et par 'non' si elle ne l'est pas, et rien d'autre.",
    ),
    "is": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("is"),
        prompt_prefix="Eftirfarandi eru setningar og hvort þær eru málfræðilega "
        "réttar.",
        prompt_template="Setning: {text}\nMálfræðilega rétt: {label}",
        instruction_prompt="Setning: {text}\n\nGreinið hvort setningin er málfræðilega "
        "rétt eða ekki. Svarið skal vera 'já' ef setningin er rétt og 'nei' ef hún er "
        "ekki, og engu öðru.",
    ),
    "it": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("it"),
        prompt_prefix="Di seguito sono riportate le frasi e la loro correttezza "
        "grammaticale.",
        prompt_template="Frase : {text}\nGrammaticalmente corretto : {label}",
        instruction_prompt="Frase: {text}\n\nStabilite se la frase è grammaticalmente "
        "corretta o meno. Rispondete con 'si' se la frase è corretta e con 'no' se "
        "non lo è, e nient'altro.",
    ),
    "nb": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("nb"),
        prompt_prefix="Følgende er setninger og hvorvidt de er grammatisk korrekte.",
        prompt_template="Setning: {text}\nGrammatisk korrekt: {label}",
        instruction_prompt="Setning: {text}\n\nBestem om setningen er grammatisk "
        "korrekt eller ikke. Svar med 'ja' hvis setningen er korrekt og 'nei' "
        "hvis den ikke er, og ikke noe annet.",
    ),
    "nl": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("nl"),
        prompt_prefix="Hieronder staan zinnen en of ze grammaticaal correct zijn.",
        prompt_template="Zin: {text}\nGrammaticaal correct: {label}",
        instruction_prompt="Zin: {text}\n\nBepaal of de zin grammaticaal correct is of "
        "niet. Antwoord met 'ja' als de zin correct is en 'nee' als dat niet het geval "
        "is, en niets anders.",
    ),
    "nn": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("nb"),
        prompt_prefix="Følgende er setninger og hvorvidt de er grammatisk korrekte.",
        prompt_template="Setning: {text}\nGrammatisk korrekt: {label}",
        instruction_prompt="Setning: {text}\n\nBestem om setningen er grammatisk "
        "korrekt eller ikke. Svar med 'ja' hvis setningen er korrekt og 'nei' "
        "hvis den ikke er, og ikke noe annet.",
    ),
    "no": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("nb"),
        prompt_prefix="Følgende er setninger og hvorvidt de er grammatisk korrekte.",
        prompt_template="Setning: {text}\nGrammatisk korrekt: {label}",
        instruction_prompt="Setning: {text}\n\nBestem om setningen er grammatisk "
        "korrekt eller ikke. Svar med 'ja' hvis setningen er korrekt og 'nei' "
        "hvis den ikke er, og ikke noe annet.",
    ),
    "sv": PromptConfig(
        **LA_DEFAULTS_DICT,
        prompt_label_mapping=get_label_mapping("sv"),
        prompt_prefix="Följande är meningar och huruvida de är grammatiskt korrekta.",
        prompt_template="Mening: {text}\nGrammatisk korrekt: {label}",
        instruction_prompt="Mening: {text}\n\nBestäm om meningen är grammatiskt "
        "korrekt eller inte. Svara med 'ja' om meningen är korrekt och 'nej' "
        "om den inte är, och inget annat.",
    ),
}
