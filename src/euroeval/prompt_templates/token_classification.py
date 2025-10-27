"""Templates for the token classification task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import (
    BULGARIAN,
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
    SLOVAK,
    SPANISH,
    SWEDISH,
    UKRAINIAN,
)

if t.TYPE_CHECKING:
    from ..languages import Language


TOKEN_CLASSIFICATION_TEMPLATES: dict["Language", PromptConfig] = {
    ENGLISH: PromptConfig(
        default_prompt_prefix="Below are texts and JSON dictionaries with the "
        "categories that occur in the given text.",
        default_prompt_template="Text: {text}\nCategories: {label}",
        default_instruction_prompt="Text: {text}\n\nIdentify the categories in "
        "the text. You should output this as a JSON dictionary with the keys being "
        "{labels_str}. The values should be lists of the spans of that category, "
        "exactly as they appear in the text.",
        default_prompt_label_mapping="auto",
    ),
    BULGARIAN: PromptConfig(
        default_prompt_prefix="По-долу са текстове и JSON речници с категориите, "
        "които се срещат в дадения текст.",
        default_prompt_template="Текст: {text}\nКатегории: {label}",
        default_instruction_prompt="Текст: {text}\n\nИдентифицирайте категориите "
        "в текста. Трябва да изведете това като JSON речник, като ключовете са "
        "{labels_str}. Стойностите трябва да бъдат списъци с откъсите от тази "
        "категория, точно както се появяват в текста.",
        default_prompt_label_mapping="auto",
    ),
    CZECH: PromptConfig(
        default_prompt_prefix="Níže jsou texty a JSON slovníky s kategoriemi, "
        "které se v daném textu vyskytují.",
        default_prompt_template="Text: {text}\nKategorie: {label}",
        default_instruction_prompt="Text: {text}\n\nIdentifikujte kategorie v "
        "textu. Měli byste to vypsat jako JSON slovník, kde klíče jsou "
        "{labels_str}. Hodnoty by měly být seznamy úseků dané kategorie, přesně "
        "tak, jak se objevují v textu.",
        default_prompt_label_mapping="auto",
    ),
    DANISH: PromptConfig(
        default_prompt_prefix="Nedenfor er tekster og JSON-ordbøger med de "
        "kategorier, der forekommer i den givne tekst.",
        default_prompt_template="Tekst: {text}\nKategorier: {label}",
        default_instruction_prompt="Tekst: {text}\n\nIdentificer kategorierne i "
        "teksten. Du skal udskrive dette som en JSON-ordbog, hvor nøglerne er "
        "{labels_str}. Værdierne skal være lister over uddrag af den kategori, "
        "præcis som de optræder i teksten.",
        default_prompt_label_mapping="auto",
    ),
    GERMAN: PromptConfig(
        default_prompt_prefix="Unten sind Texte und JSON-Wörterbücher mit den "
        "Kategorien, die im jeweiligen Text vorkommen.",
        default_prompt_template="Text: {text}\nKategorien: {label}",
        default_instruction_prompt="Text: {text}\n\nIdentifiziere die Kategorien "
        "im Text. Du solltest dies als ein JSON-Wörterbuch ausgeben, wobei die "
        "Schlüssel {labels_str} sind. Die Werte sollten Listen der Ausschnitte "
        "dieser Kategorie sein, genau so, wie sie im Text erscheinen.",
        default_prompt_label_mapping="auto",
    ),
    GREEK: PromptConfig(
        default_prompt_prefix="Παρακάτω είναι κείμενα και JSON λεξικά με τις "
        "κατηγορίες που εμφανίζονται στο δεδομένο κείμενο.",
        default_prompt_template="Κείμενο: {text}\nΚατηγορίες: {label}",
        default_instruction_prompt="Κείμενο: {text}\n\nΑναγνωρίστε τις "
        "κατηγορίες στο κείμενο. Θα πρέπει να το εκτυπώσετε ως ένα JSON "
        "λεξικό με τα κλειδιά να είναι {labels_str}. Οι τιμές θα πρέπει να "
        "είναι λίστες με τα αποσπάσματα αυτής της κατηγορίας, ακριβώς όπως "
        "εμφανίζονται στο κείμενο.",
        default_prompt_label_mapping="auto",
    ),
    SPANISH: PromptConfig(
        default_prompt_prefix="A continuación se presentan textos y diccionarios "
        "JSON con las categorías que aparecen en el texto dado.",
        default_prompt_template="Texto: {text}\nCategorías: {label}",
        default_instruction_prompt="Texto: {text}\n\nIdentifica las categorías "
        "en el texto. Debes imprimir esto como un diccionario JSON con las "
        "claves siendo {labels_str}. Los valores deben ser listas de los "
        "fragmentos de esa categoría, tal como aparecen en el texto.",
        default_prompt_label_mapping="auto",
    ),
    ESTONIAN: PromptConfig(
        default_prompt_prefix="Allpool on tekstid ja JSON-sõnastikud kategooriatega, "
        "mis esinevad antud tekstis.",
        default_prompt_template="Tekst: {text}\nKategooriad: {label}",
        default_instruction_prompt="Tekst: {text}\n\nTuvastage tekstis "
        "kategooriad. Te peate selle väljatrükkima JSON-sõnastikuna, kus "
        "võtmed on {labels_str}. Väärtused peaksid olema selle kategooria "
        "lõikude loendid, täpselt nii, nagu need tekstis esinevad.",
        default_prompt_label_mapping="auto",
    ),
    FINNISH: PromptConfig(
        default_prompt_prefix="Alla on tekstejä ja JSON-sanakirjoja, joissa on "
        "kategorioita, jotka esiintyvät annetussa tekstissä.",
        default_prompt_template="Teksti: {text}\nKategoriat: {label}",
        default_instruction_prompt="Teksti: {text}\n\nTunnista tekstin "
        "kategoriat. Sinun tulee tulostaa tämä JSON-sanakirjana, jossa "
        "avaimet ovat {labels_str}. Arvojen tulee olla kyseisen kategorian "
        "pätkien listoja, täsmälleen niin kuin ne esiintyvät tekstissä.",
        default_prompt_label_mapping="auto",
    ),
    FAROESE: PromptConfig(
        default_prompt_prefix="Niðanfyri eru tekstir og JSON orðabøkur við "
        "bólkum, ið eru í givna tekstinum.",
        default_prompt_template="Tekstur: {text}\nBólkar: {label}",
        default_instruction_prompt="Tekstur: {text}\n\nKenn aftur bólkarnar "
        "í tekstinum. Tú skalt prenta hetta sum ein JSON orðabók, har "
        "lyklarnir eru {labels_str}. Virðini skulu vera listar yvir "
        "brotini av tí bólkinum, beint sum tey síggjast í tekstinum.",
        default_prompt_label_mapping="auto",
    ),
    FRENCH: PromptConfig(
        default_prompt_prefix="Ci-dessous se trouvent des textes et des dictionnaires "
        "JSON avec les catégories qui apparaissent dans le texte donné.",
        default_prompt_template="Texte : {text}\nCatégories : {label}",
        default_instruction_prompt="Texte : {text}\n\nIdentifiez les catégories "
        "dans le texte. Vous devez l'imprimer sous la forme d'un dictionnaire JSON "
        "avec pour clés {labels_str}. Les valeurs doivent être des listes des "
        "extraits de cette catégorie, exactement comme ils apparaissent dans le texte.",
        default_prompt_label_mapping="auto",
    ),
    ICELANDIC: PromptConfig(
        default_prompt_prefix="Hér fyrir neðan eru textar og JSON orðabækur með "
        "flokkum sem koma fyrir í tilteknum texta.",
        default_prompt_template="Texti: {text}\nFlokkar: {label}",
        default_instruction_prompt="Texti: {text}\n\nFinndu flokkana í "
        "textanum. Þú átt að prenta þetta sem JSON orðabók þar sem lyklar "
        "eru {labels_str}. Gildin eiga að vera listar yfir brot af þeim "
        "flokki, nákvæmlega eins og þau koma fyrir í textanum.",
        default_prompt_label_mapping="auto",
    ),
    ITALIAN: PromptConfig(
        default_prompt_prefix="Di seguito sono riportati testi e dizionari JSON "
        "con le categorie che compaiono nel testo dato.",
        default_prompt_template="Testo: {text}\nCategorie: {label}",
        default_instruction_prompt="Testo: {text}\n\nIdentifica le categorie "
        "nel testo. Devi stampare questo come un dizionario JSON con le chiavi "
        "che sono {labels_str}. I valori devono essere liste dei brani di "
        "quella categoria, esattamente come appaiono nel testo.",
        default_prompt_label_mapping="auto",
    ),
    LITHUANIAN: PromptConfig(
        default_prompt_prefix="Žemiau pateikti tekstai ir JSON žodynai su "
        "kategorijomis, kurios pasitaiko nurodytame tekste.",
        default_prompt_template="Tekstas: {text}\nKategorijos: {label}",
        default_instruction_prompt="Tekstas: {text}\n\nNustatykite kategorijas "
        "tekste. Turite tai atspausdinti kaip JSON žodyną, kur raktai yra "
        "{labels_str}. Reikšmės turėtų būti tos kategorijos ištraukų sąrašai, "
        "tiksliai taip, kaip jos pateikiamos tekste.",
        default_prompt_label_mapping="auto",
    ),
    LATVIAN: PromptConfig(
        default_prompt_prefix="Zemāk ir teksti un JSON vārdnīcas ar kategorijām, "
        "kas parādās dotajā tekstā.",
        default_prompt_template="Teksts: {text}\nKategorijas: {label}",
        default_instruction_prompt="Teksts: {text}\n\nIdentificējiet "
        "kategorijas tekstā. Jums tas jāizdrukā kā JSON vārdnīca, kur "
        "atslēgas ir {labels_str}. Vērtībām jābūt šo kategoriju "
        "izvilkumu sarakstiem, tieši tā, kā tās parādās tekstā.",
        default_prompt_label_mapping="auto",
    ),
    NORWEGIAN_BOKMÅL: PromptConfig(
        default_prompt_prefix="Nedenfor er tekster og JSON-ordbøker med de "
        "kategoriene som forekommer i den gitte teksten.",
        default_prompt_template="Tekst: {text}\nKategorier: {label}",
        default_instruction_prompt="Tekst: {text}\n\nIdentifiser kategoriene "
        "i teksten. Du skal skrive dette ut som en JSON-ordbok med nøklene "
        "som er {labels_str}. Verdiene skal være lister over utdragene av "
        "den kategorien, akkurat som de vises i teksten.",
        default_prompt_label_mapping="auto",
    ),
    DUTCH: PromptConfig(
        default_prompt_prefix="Hieronder volgen teksten en JSON-woordenboeken "
        "met de categorieën die in de gegeven tekst voorkomen.",
        default_prompt_template="Tekst: {text}\nCategorieën: {label}",
        default_instruction_prompt="Tekst: {text}\n\nIdentificeer de "
        "categorieën in de tekst. Je moet dit afdrukken als een JSON-woordenboek "
        "met de sleutels zijnde {labels_str}. De waarden moeten lijsten zijn van "
        "de fragmenten van die categorie, precies zoals ze in de tekst voorkomen.",
        default_prompt_label_mapping="auto",
    ),
    NORWEGIAN_NYNORSK: PromptConfig(
        default_prompt_prefix="Nedanfor er tekstar og JSON-ordbøker med dei "
        "kategoriane som førekjem i den gitte teksten.",
        default_prompt_template="Tekst: {text}\nKategoriar: {label}",
        default_instruction_prompt="Tekst: {text}\n\nIdentifiser kategoriane "
        "i teksten. Du skal skrive dette ut som ein JSON-ordbok med nøklane "
        "som er {labels_str}. Verdiane skal vere lister over utdraga av den "
        "kategorien, akkurat som dei viser seg i teksten.",
        default_prompt_label_mapping="auto",
    ),
    NORWEGIAN: PromptConfig(
        default_prompt_prefix="Nedenfor er tekster og JSON-ordbøker med de "
        "kategoriene som forekommer i den gitte teksten.",
        default_prompt_template="Tekst: {text}\nKategorier: {label}",
        default_instruction_prompt="Tekst: {text}\n\nIdentifiser kategoriene "
        "i teksten. Du skal skrive dette ut som en JSON-ordbok med nøklene "
        "som er {labels_str}. Verdiene skal være lister over utdragene av "
        "den kategorien, akkurat som de vises i teksten.",
        default_prompt_label_mapping="auto",
    ),
    POLISH: PromptConfig(
        default_prompt_prefix="Poniżej znajdują się teksty i słowniki JSON z "
        "kategoriami występującymi w danym tekście.",
        default_prompt_template="Tekst: {text}\nKategorie: {label}",
        default_instruction_prompt="Tekst: {text}\n\nZidentyfikuj kategorie "
        "w tekście. Należy to wydrukować jako słownik JSON, w którym kluczami "
        "są {labels_str}. Wartości powinny być listami fragmentów danej "
        "kategorii, dokładnie tak, jak pojawiają się w tekście.",
        default_prompt_label_mapping="auto",
    ),
    PORTUGUESE: PromptConfig(
        default_prompt_prefix="A seguir estão textos e dicionários JSON com as "
        "categorias que aparecem no texto dado.",
        default_prompt_template="Texto: {text}\nCategorias: {label}",
        default_instruction_prompt="Texto: {text}\n\nIdentifique as categorias "
        "no texto. Você deve imprimir isso como um dicionário JSON com as "
        "chaves sendo {labels_str}. Os valores devem ser listas dos trechos "
        "dessa categoria, exatamente como aparecem no texto.",
        default_prompt_label_mapping="auto",
    ),
    SLOVAK: PromptConfig(
        default_prompt_prefix="Nižšie sú texty a JSON slovníky s kategóriami, "
        "ktoré sa v danom texte vyskytujú.",
        default_prompt_template="Text: {text}\nKategórie: {label}",
        default_instruction_prompt="Text: {text}\n\nIdentifikujte kategórie v "
        "texte. Mali by ste to vypísať ako JSON slovník, kde kľúče sú "
        "{labels_str}. Hodnoty by mali byť zoznamy úsekov danej kategórie, "
        "presne tak, ako sa objavujú v texte.",
        default_prompt_label_mapping="auto",
    ),
    SWEDISH: PromptConfig(
        default_prompt_prefix="Nedan är texter och JSON-ordböcker med de "
        "kategorier som förekommer i den givna texten.",
        default_prompt_template="Text: {text}\nKategorier: {label}",
        default_instruction_prompt="Text: {text}\n\nIdentifiera kategorierna "
        "i texten. Du ska skriva ut detta som en JSON-ordbok med nycklarna "
        "som är {labels_str}. Värdena ska vara listor över utdragen av den "
        "kategorin, precis som de visas i texten.",
        default_prompt_label_mapping="auto",
    ),
    UKRAINIAN: PromptConfig(
        default_prompt_prefix="Нижче наведено тексти та JSON-словники з "
        "категоріями, які зустрічаються в наведеному тексті.",
        default_prompt_template="Текст: {text}\nКатегорії: {label}",
        default_instruction_prompt="Текст: {text}\n\nВизначте категорії в "
        "тексті. Ви повинні надрукувати це як JSON-словник, де ключі - це "
        "{labels_str}. Значення повинні бути списками уривків цієї категорії, "
        "саме так, як вони з'являються в тексті.",
        default_prompt_label_mapping="auto",
    ),
}
