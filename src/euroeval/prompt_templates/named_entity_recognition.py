"""Templates for the Named Entity Recognition task."""

from ..data_models import PromptConfig
from ..languages import DA, DE, EN, ES, FI, FO, FR, IS, IT, NB, NL, NN, NO, SV

NER_TEMPLATES = {
    DA: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "person",
            "i-per": "person",
            "b-loc": "sted",
            "i-loc": "sted",
            "b-org": "organisation",
            "i-org": "organisation",
            "b-misc": "diverse",
            "i-misc": "diverse",
        },
        default_prompt_prefix="Følgende er sætninger og JSON-ordbøger med de navngivne "
        "enheder, som forekommer i den givne sætning.",
        default_prompt_template="Sætning: {text}\nNavngivne enheder: {label}",
        default_instruction_prompt="Sætning: {text}\n\nIdentificér de navngivne "
        "enheder i sætningen. Du skal outputte dette som en JSON-ordbog med nøglerne "
        "{labels_str}. Værdierne skal være lister over de navngivne enheder af den "
        "type, præcis som de forekommer i sætningen.",
    ),
    DE: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "person",
            "i-per": "person",
            "b-loc": "ort",
            "i-loc": "ort",
            "b-org": "organisation",
            "i-org": "organisation",
            "b-misc": "verschiedenes",
            "i-misc": "verschiedenes",
        },
        default_prompt_prefix="Es folgen Sätze und JSON-Wörterbücher mit den benannten "
        "Entitäten, die in der angegebenen Phrase vorkommen.",
        default_prompt_template="Satz: {text}\nBenannte Entitäten: {label}",
        default_instruction_prompt="Satz: {text}\n\nIdentifizieren Sie die benannten "
        "Entitäten im Satz. Sie sollten dies als JSON-Wörterbuch mit den "
        "Schlüsseln {labels_str} ausgeben. Die Werte sollten Listen der "
        "benannten Entitäten dieses Typs sein, genau wie sie im Satz erscheinen.",
    ),
    EN: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "person",
            "i-per": "person",
            "b-loc": "location",
            "i-loc": "location",
            "b-org": "organization",
            "i-org": "organization",
            "b-misc": "miscellaneous",
            "i-misc": "miscellaneous",
        },
        default_prompt_prefix="Below are sentences and JSON dictionaries with the "
        "named entities that occur in the given sentence.",
        default_prompt_template="Sentence: {text}\nNamed entities: {label}",
        default_instruction_prompt="Sentence: {text}\n\nIdentify the named entities in "
        "the sentence. You should output this as a JSON dictionary with the keys being "
        "{labels_str}. The values should be lists of the named entities of that "
        "type, exactly as they appear in the sentence.",
    ),
    ES: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "persona",
            "i-per": "persona",
            "b-loc": "lugar",
            "i-loc": "lugar",
            "b-org": "organización",
            "i-org": "organización",
            "b-misc": "misceláneo",
            "i-misc": "misceláneo",
        },
        default_prompt_prefix="Lo siguiente son oraciones y diccionarios JSON con las "
        "entidades nombradas que aparecen en la oración dada.",
        default_prompt_template="Oración: {text}\nEntidades nombradas: {label}",
        default_instruction_prompt="Oración: {text}\n\nIdentifica las entidades "
        "nombradas en la oración. Debes producir esto como un diccionario JSON con las "
        "claves {labels_str}. Los valores deben ser listas de las "
        "entidades nombradas de ese tipo, exactamente como aparecen en la oración.",
    ),
    FI: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "henkilö",
            "i-per": "henkilö",
            "b-loc": "paikka",
            "i-loc": "paikka",
            "b-org": "organisaatio",
            "i-org": "organisaatio",
            "b-misc": "muut",
            "i-misc": "muut",
        },
        default_prompt_prefix="Seuraavassa on lauseita ja JSON-sanakirjoja, jotka "
        "sisältävät annetussa lauseessa esiintyvät nimetyt entiteetit.",
        default_prompt_template="Lause: {text}\nNimetyt entiteetit: {label}",
        default_instruction_prompt="Lause: {text}\n\nTunnista lauseessa olevat "
        "entiteetit. Tulosta ne JSON-sanakirjana, jonka avaimet ovat {labels_str}. "
        "Arvojen tulee olla listoja kyseisen tyypin nimetyistä entiteeteistä "
        "täsmälleen siinä muodossa kuin ne esiintyvät lauseessa.",
    ),
    FO: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "persónur",
            "i-per": "persónur",
            "b-loc": "staður",
            "i-loc": "staður",
            "b-org": "felagsskapur",
            "i-org": "felagsskapur",
            "b-misc": "ymiskt",
            "i-misc": "ymiskt",
        },
        default_prompt_prefix="Her eru nakrir setningar og nakrar JSON orðabøkur við "
        "nevndar eindir, sum eru í setningunum.",
        default_prompt_template="Setningur: {text}\nNevndar eindir: {label}",
        default_instruction_prompt="Setningur: {text}\n\nGreinið nevndu einingarnar í "
        "setningunni. Þú ættir að skila þessu sem JSON orðabók með lyklunum "
        "{labels_str}. Gildin ættu að vera listi yfir nevndu einingarnar af "
        "þeirri gerð, nákvæmlega eins og þær koma fram í setningunni.",
    ),
    FR: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "personne",
            "i-per": "personne",
            "b-loc": "lieu",
            "i-loc": "lieu",
            "b-org": "organisation",
            "i-org": "organisation",
            "b-misc": "divers",
            "i-misc": "divers",
        },
        default_prompt_prefix="Vous trouverez ci-dessous des phrases et des "
        "dictionnaires JSON avec les entités nommées qui apparaissent dans la "
        "phrase donnée.",
        default_prompt_template="Sentence: {text}\nEntités nommées: {label}",
        default_instruction_prompt="Sentence: {text}\n\nIdentifiez les entités nommées "
        "dans la phrase. Vous devez produire ceci sous forme de dictionnaire JSON "
        "avec les clés {labels_str}. Les valeurs doivent être des listes des "
        "entités nommées de ce type, exactement comme elles apparaissent dans "
        "la phrase.",
    ),
    IS: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "einstaklingur",
            "i-per": "einstaklingur",
            "b-loc": "staðsetning",
            "i-loc": "staðsetning",
            "b-org": "stofnun",
            "i-org": "stofnun",
            "b-misc": "ýmislegt",
            "i-misc": "ýmislegt",
        },
        default_prompt_prefix="Eftirfarandi eru setningar ásamt JSON lyklum með "
        "nefndum einingum sem koma fyrir í setningunum.",
        default_prompt_template="Setning: {text}\nNefndar einingar: {label}",
        default_instruction_prompt="Setning: {text}\n\nGreinið nefndu einingarnar í "
        "setningunni. Þú ættir að skila þessu sem JSON orðabók með lyklunum "
        "{labels_str}. Gildin ættu að vera listi yfir nefndu "
        "einingarnar af þeirri gerð, nákvæmlega eins og þær koma fram í "
        "setningunni.",
    ),
    IT: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "persona",
            "i-per": "persona",
            "b-loc": "posizione",
            "i-loc": "posizione",
            "b-org": "organizzazione",
            "i-org": "organizzazione",
            "b-misc": "varie",
            "i-misc": "varie",
        },
        default_prompt_prefix="Di seguito sono riportate le frasi e i dizionari JSON "
        "con le entità denominate presenti nella frase data.",
        default_prompt_template="Frase: {text}\nEntità denominate: {label}",
        default_instruction_prompt="Frase: {text}\n\nIdentificare le entità nominate "
        "nella frase. Il risultato dovrebbe essere un dizionario JSON con le chiavi "
        "{labels_str}. I valori devono essere elenchi di entità "
        "nominate di quel tipo, esattamente come appaiono nella frase.",
    ),
    NB: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "person",
            "i-per": "person",
            "b-loc": "sted",
            "i-loc": "sted",
            "b-org": "organisasjon",
            "i-org": "organisasjon",
            "b-misc": "diverse",
            "i-misc": "diverse",
        },
        default_prompt_prefix="Følgende er fraser og JSON-ordbøker med de navngitte "
        "enhetene som forekommer i den gitte frasen.",
        default_prompt_template="Frase: {text}\nNavngitte enheter: {label}",
        default_instruction_prompt="Frase: {text}\n\nIdentifiser de navngitte "
        "enhetene i frasen. Du bør outputte dette som en JSON-ordbok med nøklene "
        "{labels_str}. Verdiene skal være lister over de navngitte enhetene av den "
        "typen, akkurat som de vises i frasen.",
    ),
    NL: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "persoon",
            "i-per": "persoon",
            "b-loc": "locatie",
            "i-loc": "locatie",
            "b-org": "organisatie",
            "i-org": "organisatie",
            "b-misc": "diversen",
            "i-misc": "diversen",
        },
        default_prompt_prefix="Hieronder staan zinnen en JSON woordenboeken met de "
        "genoemde entiteiten die voorkomen in de gegeven zin.",
        default_prompt_template="Zin: {text}\nGenoemde entiteiten: {label}",
        default_instruction_prompt="Zin: {text}\n\nIdentificeer de genoemde entiteiten "
        "in de zin. Je moet dit uitvoeren als een JSON-woordenboek met de sleutels "
        "{labels_str}. De waarden moeten lijsten zijn van de "
        "genoemde entiteiten van dat type, precies zoals ze voorkomen in de zin.",
    ),
    NN: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "person",
            "i-per": "person",
            "b-loc": "sted",
            "i-loc": "sted",
            "b-org": "organisasjon",
            "i-org": "organisasjon",
            "b-misc": "diverse",
            "i-misc": "diverse",
        },
        default_prompt_prefix="Følgende er fraser og JSON-ordbøker med de navngitte "
        "enhetene som forekommer i den gitte frasen.",
        default_prompt_template="Frase: {text}\nNavngitte enheter: {label}",
        default_instruction_prompt="Frase: {text}\n\nIdentifiser de navngitte enhetene "
        "i frasen. Du bør outputte dette som en JSON-ordbok med nøklene {labels_str}."
        "Verdiene skal være lister over de navngitte enhetene "
        "av den typen, akkurat som de vises i frasen.",
    ),
    NO: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "person",
            "i-per": "person",
            "b-loc": "sted",
            "i-loc": "sted",
            "b-org": "organisasjon",
            "i-org": "organisasjon",
            "b-misc": "diverse",
            "i-misc": "diverse",
        },
        default_prompt_prefix="Følgende er fraser og JSON-ordbøker med de navngitte "
        "enhetene som forekommer i den gitte frasen.",
        default_prompt_template="Frase: {text}\nNavngitte enheter: {label}",
        default_instruction_prompt="Frase: {text}\n\nIdentifiser de navngitte enhetene "
        "i frasen. Du bør outputte dette som en JSON-ordbok med nøklene {labels_str}."
        "Verdiene skal være lister over de navngitte enhetene "
        "av den typen, akkurat som de vises i frasen.",
    ),
    SV: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "person",
            "i-per": "person",
            "b-loc": "plats",
            "i-loc": "plats",
            "b-org": "organisation",
            "i-org": "organisation",
            "b-misc": "diverse",
            "i-misc": "diverse",
        },
        default_prompt_prefix="Följande är meningar och JSON-ordböcker med de "
        "namngivna enheter som förekommer i den givna meningen.",
        default_prompt_template="Mening: {text}\nNamngivna entiteter: {label}",
        default_instruction_prompt="Mening: {text}\n\nIdentifiera de namngivna "
        "enheterna i meningen. Du ska outputta detta som en JSON-ordbok med nycklarna "
        "{labels_str}. Värdena ska vara listor över de namngivna enheter av den "
        "typen, precis som de förekommer i meningen.",
    ),
}
