"""Templates for the Named Entity Recognition task."""

from ..data_models import PromptConfig
from ..languages import DA, DE, EN, ES, FO, FR, IS, IT, NB, NL, NN, NO, SV

NER_TEMPLATES = {
    DA: PromptConfig(
        prompt_label_mapping=dict(
            per="person", loc="sted", org="organisation", misc="diverse"
        ),
        prompt_prefix="Følgende er sætninger og JSON-ordbøger med de navngivne "
        "enheder, som forekommer i den givne sætning.",
        prompt_template="Sætning: {text}\nNavngivne enheder: {label}",
        instruction_prompt="Sætning: {text}\n\nIdentificér de navngivne enheder i "
        "sætningen. Du skal outputte dette som en JSON-ordbog med nøglerne "
        "{labels_str}. Værdierne skal være lister over de navngivne enheder af den "
        "type, præcis som de forekommer i sætningen.",
    ),
    DE: PromptConfig(
        prompt_label_mapping=dict(
            per="person", loc="ort", org="organisation", misc="verschiedenes"
        ),
        prompt_prefix="Es folgen Sätze und JSON-Wörterbücher mit den benannten "
        "Entitäten, die in der angegebenen Phrase vorkommen.",
        prompt_template="Satz: {text}\nBenannte Entitäten: {label}",
        instruction_prompt="Satz: {text}\n\nIdentifizieren Sie die benannten "
        "Entitäten im Satz. Sie sollten dies als JSON-Wörterbuch mit den "
        "Schlüsseln {labels_str} ausgeben. Die Werte sollten Listen der "
        "benannten Entitäten dieses Typs sein, genau wie sie im Satz erscheinen.",
    ),
    EN: PromptConfig(
        prompt_label_mapping=dict(
            per="person", loc="location", org="organization", misc="miscellaneous"
        ),
        prompt_prefix="Below are sentences and JSON dictionaries with the named "
        "entities that occur in the given sentence.",
        prompt_template="Sentence: {text}\nNamed entities: {label}",
        instruction_prompt="Sentence: {text}\n\nIdentify the named entities in the "
        "sentence. You should output this as a JSON dictionary with the keys being "
        "{labels_str}. The values should be lists of the named entities of that "
        "type, exactly as they appear in the sentence.",
    ),
    ES: PromptConfig(
        prompt_label_mapping=dict(
            per="persona", loc="lugar", org="organización", misc="misceláneo"
        ),
        prompt_prefix="Lo siguiente son oraciones y diccionarios JSON con las "
        "entidades nombradas que aparecen en la oración dada.",
        prompt_template="Oración: {text}\nEntidades nombradas: {label}",
        instruction_prompt="Oración: {text}\n\nIdentifica las entidades nombradas "
        "en la oración. Debes producir esto como un diccionario JSON con las "
        "claves {labels_str}. Los valores deben ser listas de las "
        "entidades nombradas de ese tipo, exactamente como aparecen en la oración.",
    ),
    FO: PromptConfig(
        prompt_label_mapping=dict(
            per="persónur", loc="staður", org="felagsskapur", misc="ymiskt"
        ),
        prompt_prefix="Her eru nakrir setningar og nakrar JSON orðabøkur við "
        "nevndar eindir, sum eru í setningunum.",
        prompt_template="Setningur: {text}\nNevndar eindir: {label}",
        instruction_prompt="Setningur: {text}\n\nGreinið nevndu einingarnar í "
        "setningunni. Þú ættir að skila þessu sem JSON orðabók með lyklunum "
        "{labels_str}. Gildin ættu að vera listi yfir nevndu einingarnar af "
        "þeirri gerð, nákvæmlega eins og þær koma fram í setningunni.",
    ),
    FR: PromptConfig(
        prompt_label_mapping=dict(
            per="personne", loc="lieu", org="organisation", misc="divers"
        ),
        prompt_prefix="Vous trouverez ci-dessous des phrases et des "
        "dictionnaires JSON avec les entités nommées qui apparaissent dans la "
        "phrase donnée.",
        prompt_template="Sentence: {text}\nEntités nommées: {label}",
        instruction_prompt="Sentence: {text}\n\nIdentifiez les entités nommées "
        "dans la phrase. Vous devez produire ceci sous forme de dictionnaire JSON "
        "avec les clés {labels_str}. Les valeurs doivent être des listes des "
        "entités nommées de ce type, exactement comme elles apparaissent dans "
        "la phrase.",
    ),
    IS: PromptConfig(
        prompt_label_mapping=dict(
            per="einstaklingur", loc="staðsetning", org="stofnun", misc="ýmislegt"
        ),
        prompt_prefix="Eftirfarandi eru setningar ásamt JSON lyklum með nefndum "
        "einingum sem koma fyrir í setningunum.",
        prompt_template="Setning: {text}\nNefndar einingar: {label}",
        instruction_prompt="Setning: {text}\n\nGreinið nefndu einingarnar í "
        "setningunni. Þú ættir að skila þessu sem JSON orðabók með lyklunum "
        "{labels_str}. Gildin ættu að vera listi yfir nefndu "
        "einingarnar af þeirri gerð, nákvæmlega eins og þær koma fram í "
        "setningunni.",
    ),
    IT: PromptConfig(
        prompt_label_mapping=dict(
            per="persona", loc="posizione", org="organizzazione", misc="varie"
        ),
        prompt_prefix="Di seguito sono riportate le frasi e i dizionari JSON "
        "con le entità denominate presenti nella frase data.",
        prompt_template="Frase: {text}\nEntità denominate: {label}",
        instruction_prompt="Frase: {text}\n\nIdentificare le entità nominate nella "
        "frase. Il risultato dovrebbe essere un dizionario JSON con le chiavi "
        "{labels_str}. I valori devono essere elenchi di entità "
        "nominate di quel tipo, esattamente come appaiono nella frase.",
    ),
    NB: PromptConfig(
        prompt_label_mapping=dict(
            per="person", loc="sted", org="organisasjon", misc="diverse"
        ),
        prompt_prefix="Følgende er fraser og JSON-ordbøker med de navngitte "
        "enhetene som forekommer i den gitte frasen.",
        prompt_template="Frase: {text}\nNavngitte enheter: {label}",
        instruction_prompt="Frase: {text}\n\nIdentifiser de navngitte enhetene i "
        "frasen. Du bør outputte dette som en JSON-ordbok med nøklene {labels_str}."
        "Verdiene skal være lister over de navngitte enhetene "
        "av den typen, akkurat som de vises i frasen.",
    ),
    NL: PromptConfig(
        prompt_label_mapping=dict(
            per="persoon", loc="locatie", org="organisatie", misc="diversen"
        ),
        prompt_prefix="Hieronder staan zinnen en JSON woordenboeken met de "
        "genoemde entiteiten die voorkomen in de gegeven zin.",
        prompt_template="Zin: {text}\nGenoemde entiteiten: {label}",
        instruction_prompt="Zin: {text}\n\nIdentificeer de genoemde entiteiten "
        "in de zin. Je moet dit uitvoeren als een JSON-woordenboek met de sleutels "
        "{labels_str}. De waarden moeten lijsten zijn van de "
        "genoemde entiteiten van dat type, precies zoals ze voorkomen in de zin.",
    ),
    NN: PromptConfig(
        prompt_label_mapping=dict(
            per="person", loc="sted", org="organisasjon", misc="diverse"
        ),
        prompt_prefix="Følgende er fraser og JSON-ordbøker med de navngitte "
        "enhetene som forekommer i den gitte frasen.",
        prompt_template="Frase: {text}\nNavngitte enheter: {label}",
        instruction_prompt="Frase: {text}\n\nIdentifiser de navngitte enhetene i "
        "frasen. Du bør outputte dette som en JSON-ordbok med nøklene {labels_str}."
        "Verdiene skal være lister over de navngitte enhetene "
        "av den typen, akkurat som de vises i frasen.",
    ),
    NO: PromptConfig(
        prompt_label_mapping=dict(
            per="person", loc="sted", org="organisasjon", misc="diverse"
        ),
        prompt_prefix="Følgende er fraser og JSON-ordbøker med de navngitte "
        "enhetene som forekommer i den gitte frasen.",
        prompt_template="Frase: {text}\nNavngitte enheter: {label}",
        instruction_prompt="Frase: {text}\n\nIdentifiser de navngitte enhetene i "
        "frasen. Du bør outputte dette som en JSON-ordbok med nøklene {labels_str}."
        "Verdiene skal være lister over de navngitte enhetene "
        "av den typen, akkurat som de vises i frasen.",
    ),
    SV: PromptConfig(
        prompt_label_mapping=dict(
            per="person", loc="plats", org="organisation", misc="diverse"
        ),
        prompt_prefix="Följande är meningar och JSON-ordböcker med de namngivna "
        "enheter som förekommer i den givna meningen.",
        prompt_template="Mening: {text}\nNamngivna entiteter: {label}",
        instruction_prompt="Mening: {text}\n\nIdentifiera de namngivna enheterna i "
        "meningen. Du ska outputta detta som en JSON-ordbok med nycklarna "
        "{labels_str}. Värdena ska vara listor över de namngivna enheter av den "
        "typen, precis som de förekommer i meningen.",
    ),
}
