"""Templates for the Sentiment Analysis task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import (
    ALBANIAN,
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

SENT_TEMPLATES: dict["Language", PromptConfig] = {
    ALBANIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="pozitive", neutral="neutrale", negative="negative"
        ),
        default_prompt_prefix="Më poshtë janë dokumentet dhe ndjenjat e tyre, të cilat "
        "mund të jenë {labels_str}.",
        default_prompt_template="Dokument: {text}\nNdjenja: {label}",
        default_instruction_prompt="Dokument: {text}\n\nKlasifikoni ndjenjën në "
        "dokument. Përgjigjuni vetëm me {labels_str}, dhe asgjë tjetër.",
    ),
    BOSNIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="pozitivno", neutral="neutralno", negative="negativno"
        ),
        default_prompt_prefix="Slijede dokumenti i njihova osjetila, koja mogu biti "
        "{labels_str}.",
        default_prompt_template="Dokument: {text}\nOsjetilo: {label}",
        default_instruction_prompt="Dokument: {text}\n\nKlasificirajte osjećaj u "
        "dokumentu. Odgovorite samo s {labels_str}, i ništa drugo.",
    ),
    BULGARIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="позитивен", neutral="неутрален", negative="негативен"
        ),
        default_prompt_prefix="Следват документи и техният сентимент, който може да "
        "бъде{labels_str}.",
        default_prompt_template="Документ: {text}\nСентимент: {label}",
        default_instruction_prompt="Документ: {text}\n\nКласифицирайте сентимента в "
        "документа. Отговорете с {labels_str}, и нищо друго.",
    ),
    CATALAN: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="positiu", neutral="neutral", negative="negatiu"
        ),
        default_prompt_prefix="Els documents següents i el seu sentiment, que pot "
        "ser {labels_str}.",
        default_prompt_template="Document: {text}\nSentiment: {label}",
        default_instruction_prompt="Document: {text}\n\nClassifiqueu el sentiment en "
        "el document. Contesteu només amb {labels_str}, i res més.",
    ),
    DANISH: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="positiv", neutral="neutral", negative="negativ"
        ),
        default_prompt_prefix="Følgende er dokumenter og deres sentiment, som kan være "
        "{labels_str}.",
        default_prompt_template="Dokument: {text}\nSentiment: {label}",
        default_instruction_prompt="Dokument: {text}\n\nKlassificer sentimentet i "
        "dokumentet. Svar kun med {labels_str}, og intet andet.",
    ),
    CROATIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="pozitivno", neutral="neutralno", negative="negativno"
        ),
        default_prompt_prefix="Slijede dokumenti i njihova osjetila, koja mogu biti "
        "{labels_str}.",
        default_prompt_template="Dokument: {text}\nOsjetilo: {label}",
        default_instruction_prompt="Dokument: {text}\n\nKlasificirajte osjećaj u "
        "dokumentu. Odgovorite samo s {labels_str}, i ništa drugo.",
    ),
    CZECH: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="pozitivní", neutral="neutrální", negative="negativní"
        ),
        default_prompt_prefix="Následují dokumenty a jejich sentiment, který může být "
        "{labels_str}.",
        default_prompt_template="Dokument: {text}\nSentiment: {label}",
        default_instruction_prompt="Dokument: {text}\n\nKlasifikujte sentiment v "
        "dokumentu. Odpovězte pouze s {labels_str}, a nic jiného.",
    ),
    GERMAN: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="positiv", neutral="neutral", negative="negativ"
        ),
        default_prompt_prefix="Nachfolgend finden Sie Dokumente und ihre Bewertung, "
        "die {labels_str} sein kann.",
        default_prompt_template="Dokument: {text}\nStimmung: {label}",
        default_instruction_prompt="Dokument: {text}\n\nKlassifizieren Sie die "
        "Stimmung im Dokument. Antworten Sie mit {labels_str}, und nichts anderes.",
    ),
    GREEK: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="θετικό", neutral="ουδέτερο", negative="αρνητικό"
        ),
        default_prompt_prefix="Τα ακόλουθα είναι έγγραφα και το συναίσθημά τους, "
        "το οποίο μπορεί να είναι {labels_str}.",
        default_prompt_template="Έγγραφο: {text}\nΣυναίσθημα: {label}",
        default_instruction_prompt="Έγγραφο: {text}\n\nΤαξινομήστε το συναίσθημα "
        "στο έγγραφο. Απαντήστε με {labels_str}, και τίποτα άλλο.",
    ),
    HUNGARIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="pozitív", neutral="semleges", negative="negatív"
        ),
        default_prompt_prefix="Az alábbiak dokumentumok és érzelmük, ami lehet "
        "{labels_str}.",
        default_prompt_template="Dokumentum: {text}\nÉrzelem: {label}",
        default_instruction_prompt="Dokumentum: {text}\n\nOsztályozza az érzelmet a "
        "dokumentumban. Válaszoljon {labels_str}, és semmi mással.",
    ),
    ENGLISH: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="positive", neutral="neutral", negative="negative"
        ),
        default_prompt_prefix="The following are documents and their sentiment, which "
        "can be {labels_str}.",
        default_prompt_template="Document: {text}\nSentiment: {label}",
        default_instruction_prompt="Document: {text}\n\nClassify the sentiment in the "
        "document. Answer with {labels_str}, and nothing else.",
    ),
    SPANISH: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="positivo", neutral="neutral", negative="negativo"
        ),
        default_prompt_prefix="A continuación se muestran los documentos y su "
        "sentimiento, que puede ser {labels_str}.",
        default_prompt_template="Documento: {text}\nSentimiento: {label}",
        default_instruction_prompt="Documento: {text}\n\nClasifica el sentimiento del "
        "documento. Responde con {labels_str}, y nada más.",
    ),
    ESTONIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="positiivne", neutral="neutraalne", negative="negatiivne"
        ),
        default_prompt_prefix="Järgmised on dokumendid ja nende meelestatus, "
        "mis võib olla {labels_str}.",
        default_prompt_template="Dokument: {text}\nMeelestatus: {label}",
        default_instruction_prompt="Dokument: {text}\n\nKlassifitseeri dokument "
        "meelestatuse järgi. Võimalikud vastused: {labels_str}. Muud vastused "
        "ei ole lubatud.",
    ),
    POLISH: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="pozytywny", neutral="neutralny", negative="negatywny"
        ),
        default_prompt_prefix="Poniżej znajdują się dokumenty i ich sentyment, który "
        "może być {labels_str}.",
        default_prompt_template="Dokument: {text}\nSentyment: {label}",
        default_instruction_prompt="Dokument: {text}\n\nKlasyfikuj sentyment w "
        "dokumencie. Odpowiedz jednym słowem: {labels_str}.",
    ),
    PORTUGUESE: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="positivo", neutral="neutro", negative="negativo"
        ),
        default_prompt_prefix="Abaixo encontras documentos e os seus "
        "sentimentos correspondentes, que podem ser {labels_str}.",
        default_prompt_template="Documento: {text}\nSentimento: {label}",
        default_instruction_prompt="Documento: {text}\n\nClassifica o "
        "sentimento do documento. Responde apenas com {labels_str}.",
    ),
    FINNISH: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="positiivinen", neutral="neutrali", negative="negatiivinen"
        ),
        default_prompt_prefix="Seuraavassa on arvosteluja ja niiden tunnesävy, joka "
        "voi olla {labels_str}.",
        default_prompt_template="Teksti: {text}\nTunnesävy: {label}",
        default_instruction_prompt="Teksti: {text}\n\nLuokittele arvostelun tunnesävy. "
        "Vastaa vain {labels_str}, ei muuta.",
    ),
    FAROESE: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="positivt", neutral="neutralt", negative="negativt"
        ),
        default_prompt_prefix="Niðanfyri eru skjøl og teirra kenslur, sum kunnu vera "
        "{labels_str}.",
        default_prompt_template="Skjal: {text}\nKensla: {label}",
        default_instruction_prompt="Skjal: {text}\n\nFlokka kensluna í skjalinum. "
        "Svara við {labels_str}, og einki annað.",
    ),
    FRENCH: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="positif", neutral="neutre", negative="négatif"
        ),
        default_prompt_prefix="Les documents suivants sont accompagnés de leur "
        "sentiment, qui peut être {labels_str}.",
        default_prompt_template="Document: {text}\nSentiment: {label}",
        default_instruction_prompt="Document: {text}\n\nClassez le sentiment dans le "
        "document. Répondez par {labels_str}, et rien d'autre.",
    ),
    ICELANDIC: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="jákvætt", neutral="hlutlaust", negative="neikvætt"
        ),
        default_prompt_prefix="Hér fyrir neðan eru textabrot ásamt lyndisgildi þeirra "
        "sem getur verið 'jákvætt', 'hlutlaust' eða 'neikvætt'.",
        default_prompt_template="Textabrot: {text}\nViðhorf: {label}",
        default_instruction_prompt="Textabrot: {text}\n\nGreindu lyndið í "
        "textabrotinu. Svaraðu með {labels_str}, og ekkert annað.",
    ),
    ITALIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="positivo", neutral="neutro", negative="negativo"
        ),
        default_prompt_prefix="Di seguito sono riportati i documenti e il loro "
        "sentiment, che può essere {labels_str}.",
        default_prompt_template="Documento: {text}\nSentimento: {label}",
        default_instruction_prompt="Documento: {text}\n\nClassificare il sentiment del "
        "documento. Rispondere con {labels_str}, e nient'altro.",
    ),
    LITHUANIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="teigiamas", neutral="neutralus", negative="neigiamas"
        ),
        default_prompt_prefix="Toliau pateikti dokumentai ir jų nuotaika, kuri "
        "gali būti {labels_str}.",
        default_prompt_template="Dokumentas: {text}\nNuotaika: {label}",
        default_instruction_prompt="Dokumentas: {text}\n\nKlasifikuokite nuotaiką "
        "dokumente. Atsakykite su {labels_str}, ir nieko kito.",
    ),
    LATVIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="pozitīvs", neutral="neitrāls", negative="negatīvs"
        ),
        default_prompt_prefix="Tālāk ir dokumenti un to noskaņojums, kas var būt "
        "{labels_str}.",
        default_prompt_template="Dokuments: {text}\nNoskaņojums: {label}",
        default_instruction_prompt="Dokuments: {text}\n\nKlasificējiet noskaņojumu "
        "dokumentā. Atbildiet ar {labels_str}, un neko citu.",
    ),
    NORWEGIAN_BOKMÅL: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="positiv", neutral="nøytral", negative="negativ"
        ),
        default_prompt_prefix="Her følger dokumenter og deres sentiment, som kan være "
        "{labels_str}",
        default_prompt_template="Dokument: {text}\nSentiment: {label}",
        default_instruction_prompt="Dokument: {text}\n\nKlassifiser følelsen i "
        "teksten. Svar med {labels_str}, og ikke noe annet.",
    ),
    DUTCH: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="positief", neutral="neutraal", negative="negatief"
        ),
        default_prompt_prefix="Hieronder volgen documenten en hun sentiment, dat "
        "{labels_str} kan zijn.",
        default_prompt_template="Document: {text}\nSentiment: {label}",
        default_instruction_prompt="Document: {text}\n\nClassificeer het sentiment in "
        "het document. Antwoord met {labels_str}, en verder niets.",
    ),
    NORWEGIAN_NYNORSK: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="positiv", neutral="nøytral", negative="negativ"
        ),
        default_prompt_prefix="Her følger dokumenter og deres sentiment, som kan være "
        "{labels_str}",
        default_prompt_template="Dokument: {text}\nSentiment: {label}",
        default_instruction_prompt="Dokument: {text}\n\nKlassifiser følelsen i "
        "teksten. Svar med {labels_str}, og ikke noe annet.",
    ),
    NORWEGIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="positiv", neutral="nøytral", negative="negativ"
        ),
        default_prompt_prefix="Her følger dokumenter og deres sentiment, som kan være "
        "{labels_str}",
        default_prompt_template="Dokument: {text}\nSentiment: {label}",
        default_instruction_prompt="Dokument: {text}\n\nKlassifiser følelsen i "
        "teksten. Svar med {labels_str}, og ikke noe annet.",
    ),
    ROMANIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="pozitiv", neutral="neutru", negative="negativ"
        ),
        default_prompt_prefix="Urmează documentele și sentimentul acestora, care "
        "poate fi {labels_str}.",
        default_prompt_template="Document: {text}\nSentiment: {label}",
        default_instruction_prompt="Document: {text}\n\nClasificați sentimentul "
        "documentului. Răspundeți cu {labels_str}, și nimic altceva.",
    ),
    SLOVAK: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="pozitívne", neutral="neutrálne", negative="negatívne"
        ),
        default_prompt_prefix="Nižšie sú dokumenty a ich sentiment, ktorý môže byť "
        "{labels_str}.",
        default_prompt_template="Dokument: {text}\nSentiment: {label}",
        default_instruction_prompt="Dokument: {text}\n\nKlasifikujte pocit v "
        "dokumente. Odpovedzte so {labels_str}, a nič iné.",
    ),
    SLOVENE: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="pozitivno", neutral="nevtralno", negative="negativno"
        ),
        default_prompt_prefix="Spodaj so dokumenti in njihov sentiment, ki je lahko "
        "{labels_str}.",
        default_prompt_template="Dokument: {text}\nSentiment: {label}",
        default_instruction_prompt="Dokument: {text}\n\nKlasificirajte sentiment v "
        "dokumentu. Odgovorite z {labels_str}, in nič drugega.",
    ),
    SERBIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="pozitivan", neutral="neutralan", negative="negativan"
        ),
        default_prompt_prefix="U nastavku su dokumenti i njihov sentiment, koji može "
        "biti {labels_str}.",
        default_prompt_template="Dokument: {text}\nSentiment: {label}",
        default_instruction_prompt="Dokument: {text}\n\nKlasifikujte sentiment u "
        "dokumentu. Odgovorite sa {labels_str}, i ništa drugo.",
    ),
    SWEDISH: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="positiv", neutral="neutral", negative="negativ"
        ),
        default_prompt_prefix="Nedan följer dokument och deras sentiment, som kan vara "
        "{labels_str}.",
        default_prompt_template="Dokument: {text}\nSentiment: {label}",
        default_instruction_prompt="Dokument: {text}\n\nKlassificera känslan i "
        "dokumentet. Svara med {labels_str}, och inget annat.",
    ),
    UKRAINIAN: PromptConfig(
        default_prompt_label_mapping=dict(
            positive="позитивний", neutral="нейтральний", negative="негативний"
        ),
        default_prompt_prefix="Нижче наведені документи і їх настрій, який може "
        "бути {labels_str}.",
        default_prompt_template="Документ: {text}\nНастрій: {label}",
        default_instruction_prompt="Документ: {text}\n\nКласифікуйте настрій у "
        "документі. Відповідайте {labels_str}, і нічого більше.",
    ),
}
