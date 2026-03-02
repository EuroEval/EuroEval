"""Templates for the Natural Language Inference task."""

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

NLI_TEMPLATES: dict["Language", PromptConfig] = {
    ALBANIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="e vërtetë", neutral="neutrale", contradiction="e rreme"
        ),
        default_prompt_prefix="Më poshtë janë çifte pohimesh dhe lidhja e tyre "
        "logjike, e cila mund të jetë {labels_str}.",
        default_prompt_template="{text}\nImplikimi: {label}",
        default_instruction_prompt="{text}\n\nPërcaktoni nëse pohimi i dytë rrjedh "
        "nga i pari, e kundërshton atë, apo nuk ka lidhje logjike me të. Përgjigjuni "
        "vetëm me {labels_str}, dhe asgjë tjetër.",
    ),
    BELARUSIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="праўда", neutral="нейтральна", contradiction="хлусня"
        ),
        default_prompt_prefix="Ніжэй прыведзены пары сцвярджэнняў і іх лагічная "
        "сувязь, якая можа быць {labels_str}.",
        default_prompt_template="{text}\nІмплікацыя: {label}",
        default_instruction_prompt="{text}\n\nВызначце, ці другое сцвярджэнне "
        "вынікае з першага, супярэчыць яму ці не мае з ім лагічнай сувязі. "
        "Адкажыце толькі {labels_str}, і нічога іншага.",
    ),
    BOSNIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="tačno", neutral="neutralno", contradiction="netačno"
        ),
        default_prompt_prefix="Slijede parovi tvrdnji i njihova logička veza, koja "
        "može biti {labels_str}.",
        default_prompt_template="{text}\nImplikacija: {label}",
        default_instruction_prompt="{text}\n\nOdredite slijedi li druga tvrdnja iz "
        "prve, proturječi joj ili s njom nema logičke veze. Odgovorite samo s "
        "{labels_str}, i ništa drugo.",
    ),
    BULGARIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="вярно", neutral="неутрално", contradiction="невярно"
        ),
        default_prompt_prefix="Следват двойки твърдения и логическата им връзка, "
        "която може да бъде {labels_str}.",
        default_prompt_template="{text}\nИмпликация: {label}",
        default_instruction_prompt="{text}\n\nОпределете дали второто твърдение "
        "следва от първото, противоречи му или няма логическа връзка с него. "
        "Отговорете с {labels_str}, и нищо друго.",
    ),
    CATALAN: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="cert", neutral="neutral", contradiction="fals"
        ),
        default_prompt_prefix="A continuació hi ha parells d'afirmacions i la seva "
        "relació lògica, que pot ser {labels_str}.",
        default_prompt_template="{text}\nImplicació: {label}",
        default_instruction_prompt="{text}\n\nDetermineu si la segona afirmació es "
        "dedueix de la primera, la contradiu o no té relació lògica amb ella. "
        "Contesteu només amb {labels_str}, i res més.",
    ),
    DANISH: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="sand", neutral="neutral", contradiction="falsk"
        ),
        default_prompt_prefix="Følgende er par af udsagn og om det andet udsagn "
        "følger af det første, hvilket kan være {labels_str}.",
        default_prompt_template="{text}\nEntailment: {label}",
        default_instruction_prompt="{text}\n\nBestem om det andet udsagn følger "
        "af det første udsagn. Svar kun med {labels_str}, og intet andet.",
    ),
    CROATIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="točno", neutral="neutralno", contradiction="netočno"
        ),
        default_prompt_prefix="Slijede parovi tvrdnji i njihova logička veza, koja "
        "može biti {labels_str}.",
        default_prompt_template="{text}\nImplikacija: {label}",
        default_instruction_prompt="{text}\n\nOdredite slijedi li druga tvrdnja iz "
        "prve, proturječi joj ili s njom nema logičke veze. Odgovorite samo s "
        "{labels_str}, i ništa drugo.",
    ),
    CZECH: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="pravda", neutral="neutrální", contradiction="nepravda"
        ),
        default_prompt_prefix="Následují páry tvrzení a jejich logická souvislost, "
        "která může být {labels_str}.",
        default_prompt_template="{text}\nImplikace: {label}",
        default_instruction_prompt="{text}\n\nUrčete, zda druhé tvrzení vyplývá z "
        "prvního, je s ním v rozporu nebo nemá logickou vazbu. Odpovězte pouze "
        "{labels_str}, a nic jiného.",
    ),
    DUTCH: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="waar", neutral="neutraal", contradiction="onwaar"
        ),
        default_prompt_prefix="Hieronder volgen paren van uitspraken en hun logische "
        "relatie, die {labels_str} kan zijn.",
        default_prompt_template="{text}\nImplicatie: {label}",
        default_instruction_prompt="{text}\n\nBepaal of de tweede uitspraak volgt "
        "uit de eerste, ermee in tegenspraak is of er geen logisch verband mee heeft. "
        "Antwoord met {labels_str}, en verder niets.",
    ),
    ENGLISH: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="entailment", neutral="neutral", contradiction="contradiction"
        ),
        default_prompt_prefix="The following are pairs of statements and their "
        "logical relationship, which can be {labels_str}.",
        default_prompt_template="{text}\nEntailment: {label}",
        default_instruction_prompt="{text}\n\nDetermine if the second statement "
        "follows from the first, contradicts it, or has no logical connection to it. "
        "Answer with {labels_str}, and nothing else.",
    ),
    ESTONIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="tõene", neutral="neutraalne", contradiction="väär"
        ),
        default_prompt_prefix="Järgmised on väidete paarid ja nende loogiline seos, "
        "mis võib olla {labels_str}.",
        default_prompt_template="{text}\nImplikatsioon: {label}",
        default_instruction_prompt="{text}\n\nOtsusta, kas teine väide tuleneb "
        "esimesest, on sellega vastuolus või puudub loogiline seos. Vasta "
        "{labels_str}, ja mitte midagi muud.",
    ),
    FAROESE: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="satt", neutral="neutralt", contradiction="ósatt"
        ),
        default_prompt_prefix="Hetta eru par av frágreiðingum og teirra logiska "
        "samband, sum kann vera {labels_str}.",
        default_prompt_template="{text}\nImplikasjon: {label}",
        default_instruction_prompt="{text}\n\nGreindu, um onnur frágreiðingin fylgir "
        "fyrru, stríðir móti henni, ella hevur einki logiskt samband við hana. Svara "
        "við {labels_str}, og einki annað.",
    ),
    FINNISH: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="tosi", neutral="neutraali", contradiction="epätosi"
        ),
        default_prompt_prefix="Seuraavat ovat väitepareja ja niiden looginen suhde, "
        "joka voi olla {labels_str}.",
        default_prompt_template="{text}\nImplikaatio: {label}",
        default_instruction_prompt="{text}\n\nMääritä, seuraako toinen väite "
        "ensimmäisestä, onko se ristiriidassa sen kanssa vai onko niiden välillä "
        "ei ole loogista yhteyttä. Vastaa vain {labels_str}, ei muuta.",
    ),
    FRENCH: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="vrai", neutral="neutre", contradiction="faux"
        ),
        default_prompt_prefix="Les paires suivantes sont des énoncés et leur relation "
        "logique, qui peut être {labels_str}.",
        default_prompt_template="{text}\nImplication: {label}",
        default_instruction_prompt="{text}\n\nDéterminez si le second énoncé découle "
        "du premier, le contredit ou n'a aucun lien logique avec lui. Répondez par "
        "{labels_str}, et rien d'autre.",
    ),
    GERMAN: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="wahr", neutral="neutral", contradiction="falsch"
        ),
        default_prompt_prefix="Die folgenden Paare von Aussagen und ihre logische "
        "Beziehung, die {labels_str} sein kann.",
        default_prompt_template="{text}\nImplikation: {label}",
        default_instruction_prompt="{text}\n\nBestimmen Sie, ob die zweite Aussage "
        "aus der ersten folgt, ihr widerspricht oder keine logische Verbindung zu ihr "
        "hat. Antworten Sie mit {labels_str}, und nichts anderes.",
    ),
    GREEK: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="αληθές", neutral="ουδέτερο", contradiction="ψευδές"
        ),
        default_prompt_prefix="Τα ακόλουθα είναι ζεύγη προτάσεων και η λογική τους "
        "σχέση, η οποία μπορεί να είναι {labels_str}.",
        default_prompt_template="{text}\nΣυμπέρασμα: {label}",
        default_instruction_prompt="{text}\n\nΠροσδιορίστε αν η δεύτερη πρόταση "
        "απορρέει από την πρώτη, την αντικρούει ή δεν έχει λογική σύνδεση με αυτήν. "
        "Απαντήστε με {labels_str}, και τίποτα άλλο.",
    ),
    HUNGARIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="igaz", neutral="semleges", contradiction="hamis"
        ),
        default_prompt_prefix="Az alábbiakban állítások párjai és logikai "
        "összefüggésük látható, amely {labels_str} lehet.",
        default_prompt_template="{text}\nImplikáció: {label}",
        default_instruction_prompt="{text}\n\nHatározza meg, hogy a második állítás "
        "következik-e az elsőből, ellentmond-e annak, vagy nincs logikai összefüggés "
        "köztük. Válaszoljon {labels_str}, és semmi mással.",
    ),
    ICELANDIC: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="satt", neutral="hlutlægt", contradiction="rangt"
        ),
        default_prompt_prefix="Hér fyrir neðan eru par af fullyrðingum og rökleg "
        "tengsl þeirra, sem geta verið {labels_str}.",
        default_prompt_template="{text}\nÁlyktun: {label}",
        default_instruction_prompt="{text}\n\nGreindu hvort önnur fullyrðingin "
        "leiðir af þeirri fyrri, gengur þvert gegn henni eða hefur engin rökleg "
        "tengsl við hana. Svaraðu með {labels_str}, og ekkert annað.",
    ),
    ITALIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="vero", neutral="neutro", contradiction="falso"
        ),
        default_prompt_prefix="Di seguito sono riportate coppie di affermazioni e la "
        "loro relazione logica, che può essere {labels_str}.",
        default_prompt_template="{text}\nImplicazione: {label}",
        default_instruction_prompt="{text}\n\nDeterminate se la seconda affermazione "
        "deriva dalla prima, la contraddice o non ha alcuna connessione logica con "
        "essa. Rispondere con {labels_str}, e nient'altro.",
    ),
    LATVIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="patiess", neutral="neitrāls", contradiction="nepatiess"
        ),
        default_prompt_prefix="Turpmāk ir apgalvojumu pāri un to loģiskā saistība, "
        "kas var būt {labels_str}.",
        default_prompt_template="{text}\nImplikācija: {label}",
        default_instruction_prompt="{text}\n\nNoteiciet, vai otrais apgalvojums "
        "izriet no pirmā, ir tam pretrunā vai nav loģiskas saistības ar to. "
        "Atbildiet ar {labels_str}, un neko citu.",
    ),
    LITHUANIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="tiesa", neutral="neutralu", contradiction="netiesa"
        ),
        default_prompt_prefix="Toliau pateikiamos teiginių poros ir jų loginis "
        "ryšys, kuris gali būti {labels_str}.",
        default_prompt_template="{text}\nImplikacija: {label}",
        default_instruction_prompt="{text}\n\nNustatykite, ar antrasis teiginys "
        "išplaukia iš pirmojo, jam prieštarauja ar neturi loginio ryšio su juo. "
        "Atsakykite su {labels_str}, ir nieko kito.",
    ),
    NORWEGIAN_BOKMÅL: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="sant", neutral="nøytral", contradiction="usant"
        ),
        default_prompt_prefix="Følgende er par av påstander og deres logiske "
        "sammenheng, som kan være {labels_str}.",
        default_prompt_template="{text}\nImplikasjon: {label}",
        default_instruction_prompt="{text}\n\nBestem om den andre påstanden følger "
        "av den første, motsier den eller ikke har logisk sammenheng med den. Svar "
        "med {labels_str}, og ikke noe annet.",
    ),
    NORWEGIAN_NYNORSK: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="sant", neutral="nøytral", contradiction="usant"
        ),
        default_prompt_prefix="Følgjande er par av påstandar og den logiske "
        "samanhengen deira, som kan vere {labels_str}.",
        default_prompt_template="{text}\nImplikasjon: {label}",
        default_instruction_prompt="{text}\n\nAvgjer om den andre påstanden følgjer "
        "av den første, er i strid med han eller ikkje har logisk samanheng med han. "
        "Svar med {labels_str}, og ikkje noko anna.",
    ),
    NORWEGIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="sant", neutral="nøytral", contradiction="usant"
        ),
        default_prompt_prefix="Følgende er par av påstander og deres logiske "
        "sammenheng, som kan være {labels_str}.",
        default_prompt_template="{text}\nImplikasjon: {label}",
        default_instruction_prompt="{text}\n\nBestem om den andre påstanden følger "
        "av den første, motsier den eller ikke har logisk sammenheng med den. Svar "
        "med {labels_str}, og ikke noe annet.",
    ),
    POLISH: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="prawda", neutral="neutralny", contradiction="fałsz"
        ),
        default_prompt_prefix="Poniżej znajdują się pary twierdzeń i ich logiczny "
        "związek, który może być {labels_str}.",
        default_prompt_template="{text}\nImplikacja: {label}",
        default_instruction_prompt="{text}\n\nOkreśl, czy drugie twierdzenie wynika "
        "z pierwszego, jest z nim sprzeczne lub nie ma logicznego związku. "
        "Odpowiedz jedynie {labels_str}, i nic więcej.",
    ),
    PORTUGUESE: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="verdadeiro", neutral="neutro", contradiction="falso"
        ),
        default_prompt_prefix="Abaixo encontras pares de afirmações e a sua relação "
        "lógica, que pode ser {labels_str}.",
        default_prompt_template="{text}\nImplicação: {label}",
        default_instruction_prompt="{text}\n\nDetermina se a segunda afirmação "
        "decorre da primeira, a contradiz ou não tem nenhuma ligação lógica com ela. "
        "Responde apenas com {labels_str}, e nada mais.",
    ),
    ROMANIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="adevărat", neutral="neutru", contradiction="fals"
        ),
        default_prompt_prefix="Urmează perechi de afirmații și relația lor logică, "
        "care poate fi {labels_str}.",
        default_prompt_template="{text}\nImplicație: {label}",
        default_instruction_prompt="{text}\n\nStabiliți dacă a doua afirmație "
        "decurge din prima, o contrazice sau nu are nicio legătură logică cu ea. "
        "Răspundeți cu {labels_str}, și nimic altceva.",
    ),
    SERBIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="tačno", neutral="neutralno", contradiction="netačno"
        ),
        default_prompt_prefix="U nastavku su parovi tvrdnji i njihova logička veza, "
        "koja može biti {labels_str}.",
        default_prompt_template="{text}\nImplikacija: {label}",
        default_instruction_prompt="{text}\n\nOdredite da li druga tvrdnja sledi iz "
        "prve, protivreči joj ili nema logičku vezu s njom. Odgovorite sa "
        "{labels_str}, i ništa drugo.",
    ),
    SLOVAK: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="pravda", neutral="neutrálny", contradiction="nepravda"
        ),
        default_prompt_prefix="Nasledujú páry tvrdení a ich logická súvislosť, ktorá "
        "môže byť {labels_str}.",
        default_prompt_template="{text}\nImplikácia: {label}",
        default_instruction_prompt="{text}\n\nUrčite, či druhé tvrdenie vyplýva z "
        "prvého, je s ním v rozpore alebo nemá logickú väzbu. Odpovedzte so "
        "{labels_str}, a nič iné.",
    ),
    SLOVENE: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="res", neutral="nevtralno", contradiction="napačno"
        ),
        default_prompt_prefix="Spodaj so pari trditev in njihova logična povezava, "
        "ki je lahko {labels_str}.",
        default_prompt_template="{text}\nImplikacija: {label}",
        default_instruction_prompt="{text}\n\nUgotovite, ali druga trditev sledi iz "
        "prve, ji nasprotuje ali nima logične povezave z njo. Odgovorite z "
        "{labels_str}, in nič drugega.",
    ),
    SPANISH: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="verdadero", neutral="neutral", contradiction="falso"
        ),
        default_prompt_prefix="A continuación se muestran pares de afirmaciones y su "
        "relación lógica, que puede ser {labels_str}.",
        default_prompt_template="{text}\nImplicación: {label}",
        default_instruction_prompt="{text}\n\nDetermina si la segunda afirmación se "
        "deduce de la primera, la contradice o no tiene ninguna conexión lógica con "
        "ella. Responde con {labels_str}, y nada más.",
    ),
    SWEDISH: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="sant", neutral="neutralt", contradiction="falskt"
        ),
        default_prompt_prefix="Nedan följer par av påståenden och deras logiska "
        "samband, som kan vara {labels_str}.",
        default_prompt_template="{text}\nImplikation: {label}",
        default_instruction_prompt="{text}\n\nBestäm om det andra påståendet följer "
        "av det första, motsäger det eller saknar logiskt samband med det. Svara med "
        "{labels_str}, och inget annat.",
    ),
    UKRAINIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            entailment="правда", neutral="нейтрально", contradiction="неправда"
        ),
        default_prompt_prefix="Нижче наведені пари тверджень і їх логічний зв'язок, "
        "який може бути {labels_str}.",
        default_prompt_template="{text}\nІмплікація: {label}",
        default_instruction_prompt="{text}\n\nВизначте, чи друге твердження "
        "випливає з першого, суперечить йому або не має логічного зв'язку з ним. "
        "Відповідайте {labels_str}, і нічого більше.",
    ),
}
