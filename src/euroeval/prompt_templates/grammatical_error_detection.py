"""Templates for the Grammatical Error Detection task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import (
    DANISH,
    DUTCH,
    FAROESE,
    GERMAN,
    ICELANDIC,
    NORWEGIAN,
    NORWEGIAN_BOKMÅL,
    NORWEGIAN_NYNORSK,
    SWEDISH,
)

if t.TYPE_CHECKING:
    from ..languages import Language


GED_TEMPLATES: dict["Language", PromptConfig] = {
    DANISH: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "fejl",
            "i-err": "fejl",
        },
        default_prompt_prefix="Nedenstående er sætninger og JSON-ordbøger med de "
        "grammatiske fejl, der forekommer i den givne sætning.",
        default_prompt_template="Sætning: {text}\nGrammatiske fejl: {label}",
        default_instruction_prompt="Sætning: {text}\n\nIdentificer de grammatiske "
        "fejl i sætningen. Du skal outputte dette som en JSON-ordbog med nøglen "
        "'fejl'. Værdien skal være en liste over de forkert placerede ord, præcis "
        "som de forekommer i sætningen.",
    ),
    DUTCH: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "fout",
            "i-err": "fout",
        },
        default_prompt_prefix="Hieronder staan zinnen en JSON-woordenboeken met de "
        "grammaticale fouten die in de gegeven zin voorkomen.",
        default_prompt_template="Zin: {text}\nGrammaticale fouten: {label}",
        default_instruction_prompt="Zin: {text}\n\nIdentificeer de grammaticale "
        "fouten in de zin. Je moet dit weergeven als een JSON-woordenboek met de "
        "sleutel 'fout'. De waarde moet een lijst zijn van de foutief geplaatste "
        "woorden, precies zoals ze in de zin voorkomen.",
    ),
    FAROESE: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "villa",
            "i-err": "villa",
        },
        default_prompt_prefix="Niðanfyri eru setningar og JSON orðabøkur við "
        "málvillum, ið eru í givnu setningunni.",
        default_prompt_template="Setning: {text}\nMálvillur: {label}",
        default_instruction_prompt="Setning: {text}\n\nKenn aftur málvillurnar í "
        "setningunni. Tú skalt prenta hetta sum ein JSON orðabók við lyklinum "
        "'villa'. Virðið skal vera listi yvir rangt sett orð, beint sum tey "
        "síggjast í setningunni.",
    ),
    GERMAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "fehler",
            "i-err": "fehler",
        },
        default_prompt_prefix="Unten sind Sätze und JSON-Wörterbücher mit den "
        "grammatischen Fehlern, die im jeweiligen Satz vorkommen.",
        default_prompt_template="Satz: {text}\nGrammatische Fehler: {label}",
        default_instruction_prompt="Satz: {text}\n\nIdentifiziere die grammatischen "
        "Fehler im Satz. Du solltest dies als JSON-Wörterbuch mit dem Schlüssel "
        "'fehler' ausgeben. Der Wert soll eine Liste der falsch platzierten Wörter "
        "sein, genau so, wie sie im Satz erscheinen.",
    ),
    ICELANDIC: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "villa",
            "i-err": "villa",
        },
        default_prompt_prefix="Hér fyrir neðan eru setningar og JSON orðabækur með "
        "málfræðilegum villum sem koma fyrir í viðkomandi setningu.",
        default_prompt_template="Setning: {text}\nMálfræðilegar villur: {label}",
        default_instruction_prompt="Setning: {text}\n\nFinndu málfræðilegar villur í "
        "setningunni. Þú átt að prenta þetta sem JSON orðabók með lyklinum 'villa'. "
        "Gildið á að vera listi yfir rangt staðsett orð, nákvæmlega eins og þau "
        "koma fyrir í setningunni.",
    ),
    NORWEGIAN_BOKMÅL: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "feil",
            "i-err": "feil",
        },
        default_prompt_prefix="Nedenfor er setninger og JSON-ordbøker med de "
        "grammatiske feilene som forekommer i den gitte setningen.",
        default_prompt_template="Setning: {text}\nGrammatiske feil: {label}",
        default_instruction_prompt="Setning: {text}\n\nIdentifiser de grammatiske "
        "feilene i setningen. Du skal skrive dette ut som en JSON-ordbok med "
        "nøkkelen 'feil'. Verdien skal være en liste over feilplasserte ord, akkurat "
        "som de vises i setningen.",
    ),
    NORWEGIAN_NYNORSK: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "feil",
            "i-err": "feil",
        },
        default_prompt_prefix="Nedanfor er setningar og JSON-ordbøker med dei "
        "grammatiske feila som førekjem i den gitte setninga.",
        default_prompt_template="Setning: {text}\nGrammatiske feil: {label}",
        default_instruction_prompt="Setning: {text}\n\nIdentifiser dei grammatiske "
        "feila i setninga. Du skal skrive dette ut som ein JSON-ordbok med nøkkelen "
        "'feil'. Verdien skal vere ei liste over feilplasserte ord, akkurat som dei "
        "viser seg i setninga.",
    ),
    NORWEGIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "feil",
            "i-err": "feil",
        },
        default_prompt_prefix="Nedenfor er setninger og JSON-ordbøker med de "
        "grammatiske feilene som forekommer i den gitte setningen.",
        default_prompt_template="Setning: {text}\nGrammatiske feil: {label}",
        default_instruction_prompt="Setning: {text}\n\nIdentifiser de grammatiske "
        "feilene i setningen. Du skal skrive dette ut som en JSON-ordbok med "
        "nøkkelen 'feil'. Verdien skal være en liste over feilplasserte ord, akkurat "
        "som de vises i setningen.",
    ),
    SWEDISH: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "fel",
            "i-err": "fel",
        },
        default_prompt_prefix="Nedan är meningar och JSON-ordböcker med de "
        "grammatiska fel som förekommer i den givna meningen.",
        default_prompt_template="Mening: {text}\nGrammatiska fel: {label}",
        default_instruction_prompt="Mening: {text}\n\nIdentifiera de grammatiska "
        "felen i meningen. Du ska skriva ut detta som en JSON-ordbok med nyckeln "
        "'fel'. Värdet ska vara en lista över felplacerade ord, precis som de "
        "visas i meningen.",
    ),
}
