"""Templates for the Named Entity Recognition task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import (
    BULGARIAN,
    CROATIAN,
    CZECH,
    DANISH,
    DUTCH,
    ENGLISH,
    ESTONIAN,
    FAROESE,
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
    SLOVAK,
    SLOVENIAN,
    SPANISH,
    SWEDISH,
    UKRAINIAN,
)

if t.TYPE_CHECKING:
    from ..languages import Language


NER_TEMPLATES: dict["Language", PromptConfig] = {
    BULGARIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "лице",
            "i-per": "лице",
            "b-loc": "място",
            "i-loc": "място",
            "b-org": "организация",
            "i-org": "организация",
            "b-misc": "разни",
            "i-misc": "разни",
        },
        default_prompt_prefix="По-долу са изречения и JSON речници с именуваните "
        "обекти, които се срещат в дадените изречения.",
        default_prompt_template="Изречение: {text}\nИменувани обекти: {label}",
        default_instruction_prompt="Изречение: {text}\n\nИдентифицирайте именуваните "
        "обекти в изречението. Трябва да изведете това като JSON речник с ключовете "
        "{labels_str}. Стойностите трябва да бъдат списъци на именуваните обекти от "
        "този тип, точно както се появяват в изречението.",
    ),
    CROATIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "osoba",
            "i-per": "osoba",
            "b-loc": "mjesto",
            "i-loc": "mjesto",
            "b-org": "organizacija",
            "i-org": "organizacija",
            "b-misc": "razno",
            "i-misc": "razno",
        },
        default_prompt_prefix=(
            "Sljedeće su rečenice i JSON rječnici s imenicama koje se pojavljuju u "
            "rečenicama."
        ),
        default_prompt_template=("Rečenica: {text}\nImenovane entiteti: {label}"),
        default_instruction_prompt=(
            "Rečenica: {text}\n\n"
            "Identificirajte imenovane entitete u rečenici. Prikažite ih kao JSON "
            "rječnik s ključevima {labels_str}. Vrijednosti trebaju biti popisi "
            "imenovanih entiteta navedenog tipa, točno kako se pojavljuju u rečenici."
        ),
    ),
    CZECH: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "osoba",
            "i-per": "osoba",
            "b-loc": "místo",
            "i-loc": "místo",
            "b-org": "organizace",
            "i-org": "organizace",
            "b-misc": "různé",
            "i-misc": "různé",
        },
        default_prompt_prefix="Následující jsou věty a JSON slovníky s pojmenovanými "
        "entitami, které se v dané větě vyskytují.",
        default_prompt_template="Věta: {text}\nPojmenované entity: {label}",
        default_instruction_prompt="Věta: {text}\n\nIdentifikujte pojmenované entity "
        "ve větě. Měli byste to vypsat jako JSON slovník s klíči {labels_str}. "
        "Hodnoty by měly být seznamy pojmenovaných entit tohoto typu, přesně tak, "
        "jak se objevují ve větě.",
    ),
    DANISH: PromptConfig(
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
    GERMAN: PromptConfig(
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
    GREEK: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "πρόσωπο",
            "i-per": "πρόσωπο",
            "b-loc": "τοποθεσία",
            "i-loc": "τοποθεσία",
            "b-org": "οργανισμός",
            "i-org": "οργανισμός",
            "b-misc": "διάφορα",
            "i-misc": "διάφορα",
        },
        default_prompt_prefix="Ακολουθούν προτάσεις και λεξικά JSON με τις "
        "ονομαστικές οντότητες που εμφανίζονται στην δεδομένη πρόταση.",
        default_prompt_template="Πρόταση: {text}\nΟνομαστικές οντότητες: {label}",
        default_instruction_prompt="Πρόταση: {text}\n\nΑναγνωρίστε τις ονομαστικές "
        "οντότητες στην πρόταση. Θα πρέπει να παράγετε αυτό ως λεξικό JSON με "
        "κλειδιά {labels_str}. Οι τιμές πρέπει να είναι λίστες των ονομαστικών "
        "οντοτήτων αυτού του τύπου, ακριβώς όπως εμφανίζονται στην πρόταση.",
    ),
    ENGLISH: PromptConfig(
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
    SPANISH: PromptConfig(
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
    ESTONIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "inimene",
            "i-per": "inimene",
            "b-loc": "asukoht",
            "i-loc": "asukoht",
            "b-org": "organisatsioon",
            "i-org": "organisatsioon",
            "b-misc": "muu",
            "i-misc": "muu",
        },
        default_prompt_prefix="Allpool on laused ja JSON-sõnastikud, mis sisaldavad "
        "antud lauses esinevaid nimetatud üksuseid.",
        default_prompt_template="Lause: {text}\nNimetatud üksused: {label}",
        default_instruction_prompt="Lause: {text}\n\nTuvasta lauses "
        "nimetatud üksused. Väljund peaks olema JSON-sõnastik, "
        "mille võtmed on {labels_str}. Väärtused peaksid olema kindlat tüüpi nimetatud "
        "üksuste loendid, täpselt nii nagu need lauses esinevad.",
    ),
    PORTUGUESE: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "pessoa",
            "i-per": "pessoa",
            "b-loc": "local",
            "i-loc": "local",
            "b-org": "organização",
            "i-org": "organização",
            "b-misc": "diverso",
            "i-misc": "diverso",
        },
        default_prompt_prefix="Seguem-se frases e dicionários JSON com as entidades "
        "mencionadas presentes na frase indicada.",
        default_prompt_template="Frase: {text}\nEntidades mencionadas: {label}",
        default_instruction_prompt="Frase: {text}\n\nIdentifica as entidades "
        "mencionadas na frase. Deves devolver um dicionário JSON com as chaves "
        "{labels_str}. Os valores devem ser listas contendo as entidades "
        "mencionadas desse tipo, tal como ocorrem na frase.",
    ),
    FINNISH: PromptConfig(
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
    FAROESE: PromptConfig(
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
        default_instruction_prompt="Setningur: {text}\n\nGreindu nevndu einingarnar í "
        "setningunni. Þú ættir að skila þessu sem JSON orðabók með lyklunum "
        "{labels_str}. Gildin ættu að vera listi yfir nevndu einingarnar af "
        "þeirri gerð, nákvæmlega eins og þær koma fram í setningunni.",
    ),
    FRENCH: PromptConfig(
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
    ICELANDIC: PromptConfig(
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
        default_prompt_template="Setning: {text}\nNafneiningar: {label}",
        default_instruction_prompt="Setning: {text}\n\nGreindu nefndu einingarnar í "
        "setningunni. Þú ættir að skila þessu sem JSON orðabók með lyklunum "
        "{labels_str}. Gildin ættu að vera listi yfir nefndu "
        "einingarnar af þeirri gerð, nákvæmlega eins og þær koma fram í "
        "setningunni.",
    ),
    ITALIAN: PromptConfig(
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
    LITHUANIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "asmuo",
            "i-per": "asmuo",
            "b-loc": "vieta",
            "i-loc": "vieta",
            "b-org": "organizacija",
            "i-org": "organizacija",
            "b-misc": "kita",
            "i-misc": "kita",
        },
        default_prompt_prefix="Toliau pateikti sakiniai ir JSON žodynai su vardiniais "
        "vienetais, kurie pateikiame sakinyje.",
        default_prompt_template="Sakinys: {text}\nVardiniai vienetai: {label}",
        default_instruction_prompt="Sakinys: {text}\n\nIdentifikuokite vardinius "
        "vienetus sakinyje. Turėtumėte pateikti tai kaip JSON žodyną su raktais "
        "{labels_str}. Reikšmės turi būti to tipo vardinių vienetų sąrašai, "
        "tiksliai taip, kaip jie rodomi sakinyje.",
    ),
    LATVIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "persona",
            "i-per": "persona",
            "b-loc": "vieta",
            "i-loc": "vieta",
            "b-org": "organizācija",
            "i-org": "organizācija",
            "b-misc": "dažādi",
            "i-misc": "dažādi",
        },
        default_prompt_prefix="Tālāk ir teikumi un JSON vārdnīcas ar nosauktajiem "
        "objektiem, kas parādās dotajā teikumā.",
        default_prompt_template="Teikums: {text}\nNosauktie objekti: {label}",
        default_instruction_prompt="Teikums: {text}\n\n"
        "Identificējiet nosauktos objektus "
        "teikumā. Jums jāizvada šī informācija kā JSON vārdnīcu ar atslēgām "
        "{labels_str}. Vērtībām jābūt šī tipa nosaukto objektu sarakstiem, "
        "tieši tā, kā tie parādās teikumā.",
    ),
    NORWEGIAN_BOKMÅL: PromptConfig(
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
    DUTCH: PromptConfig(
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
    NORWEGIAN_NYNORSK: PromptConfig(
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
    NORWEGIAN: PromptConfig(
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
    POLISH: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "osoba",
            "i-per": "osoba",
            "b-loc": "miejsce",
            "i-loc": "miejsce",
            "b-org": "organizacja",
            "i-org": "organizacja",
            "b-misc": "inne",
            "i-misc": "inne",
        },
        default_prompt_prefix="Poniżej znajdują się zdania i słowniki JSON "
        "z jednostkami nazewniczymi, które występują w danym zdaniu.",
        default_prompt_template="Zdanie: {text}\nJednostki nazewnicze: {label}",
        default_instruction_prompt="Zdanie: {text}\n\nZidentyfikuj jednostki "
        "nazewnicze w zdaniu. Wypisz je jako słownik JSON z kluczami "
        "{labels_str}. Wartości odpowiadające kluczom powinny być listami jednostek "
        "nazewniczych danego typu, dokładnie tak, jak pojawiają się w zdaniu.",
    ),
    SLOVAK: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "osoba",
            "i-per": "osoba",
            "b-loc": "miesto",
            "i-loc": "miesto",
            "b-org": "organizácia",
            "i-org": "organizácia",
            "b-misc": "rôzne",
            "i-misc": "rôzne",
        },
        default_prompt_prefix="Nasledujúce sú vety a JSON-objekty s pomenovanými "
        "entitami, ktoré sa nachádzajú v danej vete.",
        default_prompt_template="Veta: {text}\nPomenované entity: {label}",
        default_instruction_prompt="Veta: {text}\n\nIdentifikujte pomenované "
        "entity vo vete. Výstup by mal byť vo forme JSON-objektu s kľúčmi "
        "{labels_str}. Hodnoty by mali byť zoznamy pomenovaných entít danej "
        "kategórie, presne tak, ako sa vyskytujú vo vete.",
    ),
    SLOVENIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "oseba",
            "i-per": "oseba",
            "b-loc": "kraj",
            "i-loc": "kraj",
            "b-org": "organizacija",
            "i-org": "organizacija",
            "b-misc": "razno",
            "i-misc": "razno",
        },
        default_prompt_prefix=(
            "Naslednje so povedi in JSON slovarji z poimenovanimi "
            "entitetami, ki se pojavijo v dani povedi."
        ),
        default_prompt_template=("Poved: {text}\nPoimenovane entitete: {label}"),
        default_instruction_prompt=(
            "Poved: {text}\n\nIdentificirajte poimenovane entitete v povedi. "
            "To morate izpisati kot JSON slovar s ključi {labels_str}. "
            "Vrednosti morajo biti seznami poimenovanih entitet te kategorije, "
            "tako kot se pojavijo v povedi."
        ),
    ),
    SERBIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "osoba",
            "i-per": "osoba",
            "b-loc": "mesto",
            "i-loc": "mesto",
            "b-org": "organizacija",
            "i-org": "organizacija",
            "b-misc": "razno",
            "i-misc": "razno",
        },
        default_prompt_prefix="Sledeće su rečenice i JSON rečnici sa imenovanim "
        "entitetima koji se pojavljuju u datoj rečenici.",
        default_prompt_template="Rečenica: {text}\nImenovani entiteti: {label}",
        default_instruction_prompt="Rečenica: {text}\n\nIdentifikujte imenovane "
        "entitete u rečenici. Trebalo bi da ovo ispišete kao JSON rečnik sa ključevima "
        "{labels_str}. Vrednosti treba da budu liste imenovanih entiteta te "
        "kategorije, tačno onako kako se pojavljuju u rečenici.",
    ),
    SWEDISH: PromptConfig(
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
        "{labels_str}. Värdena ska vara listor över de namngivna enheterna av den "
        "typen, precis som de förekommer i meningen.",
    ),
    UKRAINIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-per": "особа",
            "i-per": "особа",
            "b-loc": "місце",
            "i-loc": "місце",
            "b-org": "організація",
            "i-org": "організація",
            "b-misc": "різне",
            "i-misc": "різне",
        },
        default_prompt_prefix="Нижче наведені речення та JSON-словники з іменованими "
        "сутностями, які присутні у даному реченні.",
        default_prompt_template="Речення: {text}\nІменовані сутності: {label}",
        default_instruction_prompt="Речення: {text}\n\n"
        "Ідентифікуйте іменовані сутності у "
        "реченні. Ви повинні вивести це як JSON-словник з ключами {labels_str}. "
        "Значення мають бути списками іменованих сутностей цього типу, точно "
        "такими, як вони з'являються у реченні.",
    ),
}
