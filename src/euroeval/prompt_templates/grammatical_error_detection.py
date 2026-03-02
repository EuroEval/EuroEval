"""Templates for the Grammatical Error Detection task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import (
    ALBANIAN,
    BELARUSIAN,
    BOSNIAN,
    BULGARIAN,
    CATALAN,
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
    HUNGARIAN,
    ICELANDIC,
    ITALIAN,
    LATVIAN,
    LITHUANIAN,
    NORWEGIAN,
    NORWEGIAN_BOKMÅL,
    NORWEGIAN_NYNORSK,
    POLISH,
    PORTUGUESE,
    ROMANIAN,
    SERBIAN,
    SLOVAK,
    SLOVENE,
    SPANISH,
    SWEDISH,
    UKRAINIAN,
)

if t.TYPE_CHECKING:
    from ..languages import Language


GED_TEMPLATES: dict["Language", PromptConfig] = {
    ALBANIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "gabim",
            "i-err": "gabim",
        },
        default_prompt_prefix="Më poshtë janë fjali dhe fjalorë JSON me gabimet "
        "gramatikore që shfaqen në fjalinë e dhënë.",
        default_prompt_template="Fjali: {text}\nGabime gramatikore: {label}",
        default_instruction_prompt="Fjali: {text}\n\nIdentifikoni gabimet gramatikore "
        "në fjali. Duhet t'i jepni ato si një fjalor JSON me çelësin 'gabim'. Vlerat "
        "duhet të jenë lista të fjalëve të vendosura gabimisht, saktësisht ashtu siç "
        "shfaqen në fjali.",
    ),
    BELARUSIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "памылка",
            "i-err": "памылка",
        },
        default_prompt_prefix="Ніжэй прыведзены сказы і JSON-слоўнікі з граматычнымі "
        "памылкамі, якія прысутнічаюць у дадзеным сказе.",
        default_prompt_template="Сказ: {text}\nГраматычныя памылкі: {label}",
        default_instruction_prompt="Сказ: {text}\n\nІдэнтыфікуйце граматычныя памылкі "
        "ў сказе. Вы павінны вывесці гэта як JSON-слоўнік з ключом 'памылка'. "
        "Значэнні павінны быць спісамі няправільна размешчаных слоў, дакладна такімі, "
        "як яны з'яўляюцца ў сказе.",
    ),
    BOSNIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "greška",
            "i-err": "greška",
        },
        default_prompt_prefix="Slijede rečenice i JSON riječnici s gramatičkim "
        "greškama koje se pojavljuju u danoj rečenici.",
        default_prompt_template="Rečenica: {text}\nGramatičke greške: {label}",
        default_instruction_prompt="Rečenica: {text}\n\nIdentificirajte gramatičke "
        "greške u rečenici. Prikažite ih kao JSON riječnik s ključem 'greška'. "
        "Vrijednosti trebaju biti popisi pogrešno postavljenih riječi, točno kako se "
        "pojavljuju u rečenici.",
    ),
    BULGARIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "грешка",
            "i-err": "грешка",
        },
        default_prompt_prefix="По-долу са изречения и JSON речници с граматическите "
        "грешки, които се срещат в дадените изречения.",
        default_prompt_template="Изречение: {text}\nГраматически грешки: {label}",
        default_instruction_prompt="Изречение: {text}\n\nИдентифицирайте граматическите "
        "грешки в изречението. Трябва да изведете това като JSON речник с ключа "
        "'грешка'. Стойностите трябва да бъдат списъци на неправилно поставените думи, "
        "точно както се появяват в изречението.",
    ),
    CATALAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "error",
            "i-err": "error",
        },
        default_prompt_prefix="Aquestes són frases i diccionaris JSON amb els errors "
        "gramaticals que apareixen en la frase donada.",
        default_prompt_template="Frase: {text}\nErrors gramaticals: {label}",
        default_instruction_prompt="Frase: {text}\n\nIdentifiqueu els errors gramaticals "
        "en la frase. Mostreu-los com a diccionari JSON amb la clau 'error'. Els valors "
        "han de ser les llistes de les paraules mal col·locades, tal com apareixen en "
        "la frase.",
    ),
    CROATIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "greška",
            "i-err": "greška",
        },
        default_prompt_prefix="Sljedeće su rečenice i JSON rječnici s gramatičkim "
        "greškama koje se pojavljuju u danoj rečenici.",
        default_prompt_template="Rečenica: {text}\nGramatičke greške: {label}",
        default_instruction_prompt="Rečenica: {text}\n\nIdentificirajte gramatičke "
        "greške u rečenici. Prikažite ih kao JSON rječnik s ključem 'greška'. "
        "Vrijednosti trebaju biti popisi pogrešno postavljenih riječi, točno kako se "
        "pojavljuju u rečenici.",
    ),
    CZECH: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "chyba",
            "i-err": "chyba",
        },
        default_prompt_prefix="Následující jsou věty a JSON slovníky s gramatickými "
        "chybami, které se v dané větě vyskytují.",
        default_prompt_template="Věta: {text}\nGramatické chyby: {label}",
        default_instruction_prompt="Věta: {text}\n\nIdentifikujte gramatické chyby ve "
        "větě. Měli byste to vypsat jako JSON slovník s klíčem 'chyba'. Hodnoty by "
        "měly být seznamy nesprávně umístěných slov, přesně tak, jak se objevují ve "
        "větě.",
    ),
    DANISH: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "fejl",
            "i-err": "fejl",
        },
        default_prompt_prefix="Nedenstående er sætninger og JSON-ordbøger med de "
        "grammatiske fejl, der forekommer i den givne sætning.",
        default_prompt_template="Sætning: {text}\nGrammatiske fejl: {label}",
        default_instruction_prompt="Sætning: {text}\n\nIdentificér de grammatiske "
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
    ENGLISH: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "error",
            "i-err": "error",
        },
        default_prompt_prefix="Below are sentences and JSON dictionaries with the "
        "grammatical errors that occur in the given sentence.",
        default_prompt_template="Sentence: {text}\nGrammatical errors: {label}",
        default_instruction_prompt="Sentence: {text}\n\nIdentify the grammatical "
        "errors in the sentence. You should output this as a JSON dictionary with the "
        "key 'error'. The value should be a list of the misplaced words, exactly as "
        "they appear in the sentence.",
    ),
    ESTONIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "viga",
            "i-err": "viga",
        },
        default_prompt_prefix="Allpool on laused ja JSON-sõnastikud, mis sisaldavad "
        "antud lauses esinevaid grammatilisi vigu.",
        default_prompt_template="Lause: {text}\nGrammatilised vead: {label}",
        default_instruction_prompt="Lause: {text}\n\nTuvasta lauses esinevad "
        "grammatilised vead. Väljund peaks olema JSON-sõnastik, mille võti on 'viga'. "
        "Väärtus peaks olema valesti paigutatud sõnade loend, täpselt nii nagu need "
        "lauses esinevad.",
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
    FINNISH: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "virhe",
            "i-err": "virhe",
        },
        default_prompt_prefix="Seuraavassa on lauseita ja JSON-sanakirjoja, jotka "
        "sisältävät annetussa lauseessa esiintyvät kieliopilliset virheet.",
        default_prompt_template="Lause: {text}\nKieliopilliset virheet: {label}",
        default_instruction_prompt="Lause: {text}\n\nTunnista lauseessa olevat "
        "kieliopilliset virheet. Tulosta ne JSON-sanakirjana, jonka avain on 'virhe'. "
        "Arvon tulee olla lista väärin sijoitetuista sanoista täsmälleen siinä "
        "muodossa kuin ne esiintyvät lauseessa.",
    ),
    FRENCH: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "erreur",
            "i-err": "erreur",
        },
        default_prompt_prefix="Vous trouverez ci-dessous des phrases et des "
        "dictionnaires JSON avec les erreurs grammaticales qui apparaissent dans "
        "la phrase donnée.",
        default_prompt_template="Sentence: {text}\nErreurs grammaticales: {label}",
        default_instruction_prompt="Sentence: {text}\n\nIdentifiez les erreurs "
        "grammaticales dans la phrase. Vous devez produire ceci sous forme de "
        "dictionnaire JSON avec la clé 'erreur'. La valeur doit être une liste des "
        "mots mal placés, exactement comme ils apparaissent dans la phrase.",
    ),
    GERMAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "fehler",
            "i-err": "fehler",
        },
        default_prompt_prefix="Unten sind Sätze und JSON-Wörterbücher mit den "
        "grammatischen Fehlern, die im jeweiligen Satz vorkommen.",
        default_prompt_template="Satz: {text}\nGrammatische Fehler: {label}",
        default_instruction_prompt="Satz: {text}\n\nIdentifizieren Sie die grammatischen "
        "Fehler im Satz. Sie sollten dies als JSON-Wörterbuch mit dem Schlüssel "
        "'fehler' ausgeben. Der Wert soll eine Liste der falsch platzierten Wörter "
        "sein, genau so, wie sie im Satz erscheinen.",
    ),
    GREEK: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "σφάλμα",
            "i-err": "σφάλμα",
        },
        default_prompt_prefix="Ακολουθούν προτάσεις και λεξικά JSON με τα γραμματικά "
        "σφάλματα που εμφανίζονται στην δεδομένη πρόταση.",
        default_prompt_template="Πρόταση: {text}\nΓραμματικά σφάλματα: {label}",
        default_instruction_prompt="Πρόταση: {text}\n\nΑναγνωρίστε τα γραμματικά "
        "σφάλματα στην πρόταση. Θα πρέπει να παράγετε αυτό ως λεξικό JSON με το "
        "κλειδί 'σφάλμα'. Η τιμή πρέπει να είναι λίστα με τις λέξεις που είναι "
        "λανθασμένα τοποθετημένες, ακριβώς όπως εμφανίζονται στην πρόταση.",
    ),
    HUNGARIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "hiba",
            "i-err": "hiba",
        },
        default_prompt_prefix="Az alábbiakban mondatok és JSON szótárak találhatók "
        "az adott mondatokban előforduló nyelvtani hibákkal.",
        default_prompt_template="Mondat: {text}\nNyelvtani hibák: {label}",
        default_instruction_prompt="Mondat: {text}\n\nAzonosítsa a mondatban lévő "
        "nyelvtani hibákat. Adja meg ezeket JSON szótárként a 'hiba' kulccsal. Az "
        "érték legyen a rosszul elhelyezett szavak listája, pontosan úgy, ahogyan "
        "megjelennek a mondatban.",
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
    ITALIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "errore",
            "i-err": "errore",
        },
        default_prompt_prefix="Di seguito sono riportate le frasi e i dizionari JSON "
        "con gli errori grammaticali presenti nella frase data.",
        default_prompt_template="Frase: {text}\nErrori grammaticali: {label}",
        default_instruction_prompt="Frase: {text}\n\nIdentificare gli errori "
        "grammaticali nella frase. Il risultato dovrebbe essere un dizionario JSON con "
        "la chiave 'errore'. Il valore deve essere un elenco delle parole mal "
        "posizionate, esattamente come appaiono nella frase.",
    ),
    LATVIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "kļūda",
            "i-err": "kļūda",
        },
        default_prompt_prefix="Tālāk ir teikumi un JSON vārdnīcas ar gramatiskajām "
        "kļūdām, kas parādās dotajā teikumā.",
        default_prompt_template="Teikums: {text}\nGramatiskās kļūdas: {label}",
        default_instruction_prompt="Teikums: {text}\n\nIdentificējiet gramatiskās "
        "kļūdas teikumā. Jums jāizvada šī informācija kā JSON vārdnīca ar atslēgu "
        "'kļūda'. Vērtībai jābūt nepareizi novietoto vārdu sarakstam, tieši tā, kā "
        "tie parādās teikumā.",
    ),
    LITHUANIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "klaida",
            "i-err": "klaida",
        },
        default_prompt_prefix="Toliau pateikti sakiniai ir JSON žodynai su "
        "gramatinėmis klaidomis, kurios pateikiame sakinyje.",
        default_prompt_template="Sakinys: {text}\nGramatinės klaidos: {label}",
        default_instruction_prompt="Sakinys: {text}\n\nIdentifikuokite gramatines "
        "klaidas sakinyje. Turėtumėte pateikti tai kaip JSON žodyną su raktu "
        "'klaida'. Reikšmė turi būti neteisingai išdėstytų žodžių sąrašas, tiksliai "
        "taip, kaip jie rodomi sakinyje.",
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
    POLISH: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "błąd",
            "i-err": "błąd",
        },
        default_prompt_prefix="Poniżej znajdują się zdania i słowniki JSON z błędami "
        "gramatycznymi, które występują w danym zdaniu.",
        default_prompt_template="Zdanie: {text}\nBłędy gramatyczne: {label}",
        default_instruction_prompt="Zdanie: {text}\n\nZidentyfikuj błędy gramatyczne "
        "w zdaniu. Wypisz je jako słownik JSON z kluczem 'błąd'. Wartość powinna być "
        "listą nieprawidłowo umieszczonych słów, dokładnie tak, jak pojawiają się w "
        "zdaniu.",
    ),
    PORTUGUESE: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "erro",
            "i-err": "erro",
        },
        default_prompt_prefix="Seguem-se frases e dicionários JSON com os erros "
        "gramaticais presentes na frase indicada.",
        default_prompt_template="Frase: {text}\nErros gramaticais: {label}",
        default_instruction_prompt="Frase: {text}\n\nIdentifica os erros gramaticais "
        "na frase. Deves devolver um dicionário JSON com a chave 'erro'. O valor deve "
        "ser uma lista das palavras mal colocadas, tal como ocorrem na frase.",
    ),
    ROMANIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "eroare",
            "i-err": "eroare",
        },
        default_prompt_prefix="Mai jos sunt propoziții și dicționare JSON cu erorile "
        "gramaticale care apar în propoziția dată.",
        default_prompt_template="Propoziție: {text}\nErori gramaticale: {label}",
        default_instruction_prompt="Propoziție: {text}\n\nIdentifică erorile "
        "gramaticale din propoziție. Ar trebui să le enumeri ca un dicționar JSON cu "
        "cheia 'eroare'. Valoarea trebuie să fie o listă de cuvinte plasate greșit, "
        "exact cum apar în propoziție.",
    ),
    SERBIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "greška",
            "i-err": "greška",
        },
        default_prompt_prefix="Sledeće su rečenice i JSON rečnici sa gramatičkim "
        "greškama koji se pojavljuju u datoj rečenici.",
        default_prompt_template="Rečenica: {text}\nGramatičke greške: {label}",
        default_instruction_prompt="Rečenica: {text}\n\nIdentifikujte gramatičke "
        "greške u rečenici. Trebalo bi da ovo ispišete kao JSON rečnik sa ključem "
        "'greška'. Vrednost treba da bude lista reči pogrešno postavljenih, tačno "
        "onako kako se pojavljuju u rečenici.",
    ),
    SLOVAK: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "chyba",
            "i-err": "chyba",
        },
        default_prompt_prefix="Nasledujúce sú vety a JSON-objekty s gramatickými "
        "chybami, ktoré sa nachádzajú v danej vete.",
        default_prompt_template="Veta: {text}\nGramatické chyby: {label}",
        default_instruction_prompt="Veta: {text}\n\nIdentifikujte gramatické chyby vo "
        "vete. Výstup by mal byť vo forme JSON-objektu s kľúčom 'chyba'. Hodnota by "
        "mala byť zoznamom nesprávne umiestnených slov, presne tak, ako sa vyskytujú "
        "vo vete.",
    ),
    SLOVENE: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "napaka",
            "i-err": "napaka",
        },
        default_prompt_prefix="Naslednje so povedi in JSON slovarji z gramatičnimi "
        "napakami, ki se pojavijo v dani povedi.",
        default_prompt_template="Poved: {text}\nGramatične napake: {label}",
        default_instruction_prompt="Poved: {text}\n\nIdentificirajte gramatične napake "
        "v povedi. To morate izpisati kot JSON slovar s ključem 'napaka'. Vrednost "
        "mora biti seznam napačno postavljenih besed, tako kot se pojavijo v povedi.",
    ),
    SPANISH: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "error",
            "i-err": "error",
        },
        default_prompt_prefix="Lo siguiente son oraciones y diccionarios JSON con los "
        "errores gramaticales que aparecen en la oración dada.",
        default_prompt_template="Oración: {text}\nErrores gramaticales: {label}",
        default_instruction_prompt="Oración: {text}\n\nIdentifica los errores "
        "gramaticales en la oración. Debes producir esto como un diccionario JSON con "
        "la clave 'error'. El valor debe ser una lista de las palabras mal colocadas, "
        "exactamente como aparecen en la oración.",
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
    UKRAINIAN: PromptConfig(
        default_prompt_label_mapping={
            "b-err": "помилка",
            "i-err": "помилка",
        },
        default_prompt_prefix="Нижче наведені речення та JSON-словники з "
        "граматичними помилками, які присутні у даному реченні.",
        default_prompt_template="Речення: {text}\nГраматичні помилки: {label}",
        default_instruction_prompt="Речення: {text}\n\nІдентифікуйте граматичні "
        "помилки у реченні. Ви повинні вивести це як JSON-словник з ключем "
        "'помилка'. Значення має бути списком неправильно розміщених слів, точно "
        "такими, як вони з'являються у реченні.",
    ),
}
