"""Templates for the Word in Context task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import (
    ALBANIAN,
    BELARUSIAN,
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

WIC_TEMPLATES: dict["Language", PromptConfig] = {
    ALBANIAN: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="po", different_sense="jo"),
        default_prompt_prefix="Më poshtë janë shembuj të fjalëve të përdorura në dy "
        "kontekste dhe nëse kanë të njëjtën kuptim.",
        default_prompt_template="{text}\nE njëjta kuptim: {label}",
        default_instruction_prompt="{text}\n\nA ka fjala të njëjtën kuptim në të dy "
        "kontekstet? Përgjigjuni me {labels_str}, dhe asgjë tjetër.",
    ),
    BELARUSIAN: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="так", different_sense="не"),
        default_prompt_prefix="Ніжэй прыведзены прыклады слоў, якія выкарыстоўваюцца "
        "ў двух кантэкстах, і ці маюць яны аднолькавае значэнне.",
        default_prompt_template="{text}\nАднолькавае значэнне: {label}",
        default_instruction_prompt="{text}\n\nЦі мае слова аднолькавае значэнне ў "
        "абодвух кантэкстах? Адкажыце толькі {labels_str}, і нічога іншага.",
    ),
    BULGARIAN: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="да", different_sense="не"),
        default_prompt_prefix="Следват примери за думи, използвани в два контекста, и "
        "дали имат едно и също значение.",
        default_prompt_template="{text}\nЕдно и също значение: {label}",
        default_instruction_prompt="{text}\n\nИма ли думата едно и също значение в "
        "двата контекста? Отговорете с {labels_str}, и нищо друго.",
    ),
    CATALAN: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="sí", different_sense="no"),
        default_prompt_prefix="A continuació es mostren exemples de paraules usades en "
        "dos contextos i si tenen el mateix significat.",
        default_prompt_template="{text}\nMateix significat: {label}",
        default_instruction_prompt="{text}\n\nTé la paraula el mateix significat en "
        "els dos contextos? Respon amb {labels_str}, i res més.",
    ),
    CROATIAN: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="da", different_sense="ne"),
        default_prompt_prefix="Sljedeći su primjeri riječi korištenih u dva konteksta "
        "i imaju li isto značenje.",
        default_prompt_template="{text}\nIsto značenje: {label}",
        default_instruction_prompt="{text}\n\nIma li riječ isto značenje u oba "
        "konteksta? Odgovorite s {labels_str}, i ništa drugo.",
    ),
    CZECH: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="ano", different_sense="ne"),
        default_prompt_prefix="Následující jsou příklady slov použitých ve dvou "
        "kontextech a zda mají stejný význam.",
        default_prompt_template="{text}\nStejný význam: {label}",
        default_instruction_prompt="{text}\n\nMá slovo ve dvou kontextech stejný "
        "význam? Odpovězte {labels_str}, a nic jiné.",
    ),
    DANISH: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="ja", different_sense="nej"),
        default_prompt_prefix="Følgende er eksempler på ord brugt i to kontekster og "
        "om de har samme betydning.",
        default_prompt_template="{text}\nSamme betydning: {label}",
        default_instruction_prompt="{text}\n\nHar ordet den samme betydning i de to "
        "kontekster? Svar kun med {labels_str}, og intet andet.",
    ),
    DUTCH: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="ja", different_sense="nee"),
        default_prompt_prefix="Hieronder staan voorbeelden van woorden in twee "
        "contexten en of ze dezelfde betekenis hebben.",
        default_prompt_template="{text}\nZelfde betekenis: {label}",
        default_instruction_prompt="{text}\n\nHeeft het woord dezelfde betekenis in "
        "beide contexten? Antwoord met {labels_str}, en verder niets.",
    ),
    ENGLISH: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="yes", different_sense="no"),
        default_prompt_prefix="The following are examples of words used in two "
        "contexts and whether they have the same meaning.",
        default_prompt_template="{text}\nSame meaning: {label}",
        default_instruction_prompt="{text}\n\nDoes the word have the same meaning in "
        "both contexts? Answer with {labels_str}, and nothing else.",
    ),
    ESTONIAN: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="jah", different_sense="ei"),
        default_prompt_prefix="Järgnevad on näited sõnadest, mida kasutatakse kahes "
        "kontekstis, ja kas neil on sama tähendus.",
        default_prompt_template="{text}\nSama tähendus: {label}",
        default_instruction_prompt="{text}\n\nKas sõnal on mõlemas kontekstis sama "
        "tähendus? Vasta {labels_str}, ja mitte midagi muud.",
    ),
    FAROESE: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="ja", different_sense="nei"),
        default_prompt_prefix="Hér eru dæmi um orð brúkt í tveimum samanhangum og um "
        "tey hava somu týdningina.",
        default_prompt_template="{text}\nSama týdningur: {label}",
        default_instruction_prompt="{text}\n\nHevur orðið somu týdningina í báðum "
        "samanhingunum? Svara við {labels_str}, og einki annað.",
    ),
    FINNISH: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="kyllä", different_sense="ei"),
        default_prompt_prefix="Seuraavat ovat esimerkkejä sanoista kahdessa "
        "asiayhteydessä ja onko niillä sama merkitys.",
        default_prompt_template="{text}\nSama merkitys: {label}",
        default_instruction_prompt="{text}\n\nOnko sanalla sama merkitys molemmissa "
        "asiayhteyksissä? Vastaa {labels_str}, ja ei mitään muuta.",
    ),
    FRENCH: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="oui", different_sense="non"),
        default_prompt_prefix="Les exemples suivants montrent des mots utilisés dans "
        "deux contextes et s'ils ont le même sens.",
        default_prompt_template="{text}\nMême sens : {label}",
        default_instruction_prompt="{text}\n\nLe mot a-t-il le même sens dans les deux "
        "contextes ? Répondez par {labels_str}, et rien d'autre.",
    ),
    GERMAN: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="ja", different_sense="nein"),
        default_prompt_prefix="Im Folgenden sind Beispiele für Wörter in zwei "
        "Kontexten und ob sie die gleiche Bedeutung haben.",
        default_prompt_template="{text}\nGleiche Bedeutung: {label}",
        default_instruction_prompt="{text}\n\nHat das Wort in beiden Kontexten die "
        "gleiche Bedeutung? Antworten Sie mit {labels_str}, und nichts anderes.",
    ),
    GREEK: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="ναι", different_sense="όχι"),
        default_prompt_prefix="Τα παρακάτω είναι παραδείγματα λέξεων που "
        "χρησιμοποιούνται σε δύο περιβάλλοντα και αν έχουν την ίδια σημασία.",
        default_prompt_template="{text}\nΊδια σημασία: {label}",
        default_instruction_prompt="{text}\n\nΈχει η λέξη την ίδια σημασία και στα δύο "
        "περιβάλλοντα; Απαντήστε με {labels_str}, και τίποτα άλλο.",
    ),
    HUNGARIAN: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="igen", different_sense="nem"),
        default_prompt_prefix="A következők szavak két kontextusban való használatára "
        "és azonos jelentésükre mutatnak példát.",
        default_prompt_template="{text}\nAzonos jelentés: {label}",
        default_instruction_prompt="{text}\n\nAzonos jelentéssel bír-e a szó mindkét "
        "kontextusban? Csak {labels_str}-val válaszoljon, és semmi mással.",
    ),
    ICELANDIC: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="já", different_sense="nei"),
        default_prompt_prefix="Hér fyrir neðan eru dæmi um orð notuð í tveimur "
        "samhengi og hvort þau hafi sömu merkingu.",
        default_prompt_template="{text}\nSama merking: {label}",
        default_instruction_prompt="{text}\n\nHefur orðið sömu merkingu í báðum "
        "samhengjum? Svaraðu með {labels_str}, og ekkert annað.",
    ),
    ITALIAN: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="sì", different_sense="no"),
        default_prompt_prefix="Di seguito sono riportati esempi di parole usate in due "
        "contesti e se hanno lo stesso significato.",
        default_prompt_template="{text}\nStesso significato: {label}",
        default_instruction_prompt="{text}\n\nLa parola ha lo stesso significato in "
        "entrambi i contesti? Rispondere con {labels_str}, e nient'altro.",
    ),
    LATVIAN: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="jā", different_sense="nē"),
        default_prompt_prefix="Turpmāk ir piemēri vārdiem, kas izmantoti divos "
        "kontekstos, un vai tiem ir vienāda nozīme.",
        default_prompt_template="{text}\nVienāda nozīme: {label}",
        default_instruction_prompt="{text}\n\nVai vārdam ir vienāda nozīme abos "
        "kontekstos? Atbildiet ar {labels_str}, un neko citu.",
    ),
    LITHUANIAN: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="taip", different_sense="ne"),
        default_prompt_prefix="Toliau pateikti pavyzdžiai, kaip žodžiai vartojami "
        "dviejuose kontekstuose, ir ar jie turi tą pačią reikšmę.",
        default_prompt_template="{text}\nTa pati reikšmė: {label}",
        default_instruction_prompt="{text}\n\nAr žodis turi tą pačią reikšmę "
        "abiejuose kontekstuose? Atsakykite su {labels_str}, ir nieko kito.",
    ),
    NORWEGIAN_BOKMÅL: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="ja", different_sense="nei"),
        default_prompt_prefix="Følgende er eksempler på ord brukt i to kontekster og "
        "om de har samme betydning.",
        default_prompt_template="{text}\nSamme betydning: {label}",
        default_instruction_prompt="{text}\n\nHar ordet samme betydning i begge "
        "kontekstene? Svar med {labels_str}, og ikke noe annet.",
    ),
    NORWEGIAN_NYNORSK: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="ja", different_sense="nei"),
        default_prompt_prefix="Følgjande er eksempel på ord brukte i to kontekstar og "
        "om dei har same tyding.",
        default_prompt_template="{text}\nSame tyding: {label}",
        default_instruction_prompt="{text}\n\nHar ordet same tyding i begge "
        "kontekstane? Svar med {labels_str}, og ikkje noko anna.",
    ),
    NORWEGIAN: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="ja", different_sense="nei"),
        default_prompt_prefix="Følgende er eksempler på ord brukt i to kontekster og "
        "om de har samme betydning.",
        default_prompt_template="{text}\nSamme betydning: {label}",
        default_instruction_prompt="{text}\n\nHar ordet samme betydning i begge "
        "kontekstene? Svar med {labels_str}, og ikke noe annet.",
    ),
    POLISH: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="tak", different_sense="nie"),
        default_prompt_prefix="Poniżej znajdują się przykłady słów użytych w dwóch "
        "kontekstach i czy mają to samo znaczenie.",
        default_prompt_template="{text}\nTo samo znaczenie: {label}",
        default_instruction_prompt="{text}\n\nCzy słowo ma to samo znaczenie w obu "
        "kontekstach? Odpowiedz używając wyłącznie {labels_str}.",
    ),
    PORTUGUESE: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="sim", different_sense="não"),
        default_prompt_prefix="A seguir estão exemplos de palavras usadas em dois "
        "contextos e se têm o mesmo significado.",
        default_prompt_template="{text}\nMesmo significado: {label}",
        default_instruction_prompt="{text}\n\nA palavra tem o mesmo significado em "
        "ambos os contextos? Responde com {labels_str}, e nada mais.",
    ),
    ROMANIAN: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="da", different_sense="nu"),
        default_prompt_prefix="Următoarele sunt exemple de cuvinte folosite în două "
        "contexte și dacă au același sens.",
        default_prompt_template="{text}\nAcelași sens: {label}",
        default_instruction_prompt="{text}\n\nAre cuvântul același sens în ambele "
        "contexte? Răspundeți cu {labels_str}, și nimic altceva.",
    ),
    SERBIAN: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="da", different_sense="ne"),
        default_prompt_prefix="U nastavku su primjeri reči korišćenih u dva konteksta "
        "i da li imaju isto značenje.",
        default_prompt_template="{text}\nIsto značenje: {label}",
        default_instruction_prompt="{text}\n\nIma li reč isto značenje u oba "
        "konteksta? Odgovorite sa {labels_str}, i ništa drugo.",
    ),
    SLOVAK: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="áno", different_sense="nie"),
        default_prompt_prefix="Nasledujú príklady slov použitých v dvoch kontextoch "
        "a či majú rovnaký význam.",
        default_prompt_template="{text}\nRovnaký význam: {label}",
        default_instruction_prompt="{text}\n\nMá slovo v oboch kontextoch rovnaký "
        "význam? Odpovedzte so {labels_str}, a nič iné.",
    ),
    SLOVENE: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="da", different_sense="ne"),
        default_prompt_prefix="Sledeči so primeri besed, ki se uporabljajo v dveh "
        "kontekstih, in ali imajo enak pomen.",
        default_prompt_template="{text}\nEnak pomen: {label}",
        default_instruction_prompt="{text}\n\nAli ima beseda enak pomen v obeh "
        "kontekstih? Odgovorite z {labels_str}, in nič drugega.",
    ),
    SPANISH: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="sí", different_sense="no"),
        default_prompt_prefix="A continuación se presentan ejemplos de palabras usadas "
        "en dos contextos y si tienen el mismo significado.",
        default_prompt_template="{text}\nMismo significado: {label}",
        default_instruction_prompt="{text}\n\n¿Tiene la palabra el mismo significado "
        "en ambos contextos? Responde con {labels_str}, y nada más.",
    ),
    SWEDISH: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="ja", different_sense="nej"),
        default_prompt_prefix="Följande är exempel på ord använda i två sammanhang och "
        "om de har samma betydelse.",
        default_prompt_template="{text}\nSamma betydelse: {label}",
        default_instruction_prompt="{text}\n\nHar ordet samma betydelse i båda "
        "sammanhangen? Svara med {labels_str}, och inget annat.",
    ),
    UKRAINIAN: PromptConfig(
        default_prompt_label_mapping=dict(same_sense="так", different_sense="ні"),
        default_prompt_prefix="Нижче наведені приклади слів, що використовуються в "
        "двох контекстах, і чи мають вони однакове значення.",
        default_prompt_template="{text}\nОднакове значення: {label}",
        default_instruction_prompt="{text}\n\nЧи має слово однакове значення в обох "
        "контекстах? Відповідайте {labels_str}, і нічого більше.",
    ),
}
