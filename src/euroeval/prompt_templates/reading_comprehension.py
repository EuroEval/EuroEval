"""Templates for the Reading Comprehension task."""

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

RC_TEMPLATES: dict["Language", PromptConfig] = {
    BULGARIAN: PromptConfig(
        default_prompt_prefix="Следват текстове със съответни въпроси и отговори.",
        default_prompt_template="Текст: {text}\nВъпрос: {question}\nОтговор с максимум "
        "3 думи: {label}",
        default_instruction_prompt="Текст: {text}\n\nОтговорете на следния въпрос "
        "относно текста по-горе с максимум 3 думи.\n\nВъпрос: {question}",
        default_prompt_label_mapping=dict(),
    ),
    CROATIAN: PromptConfig(
        default_prompt_prefix=("Sljedeći tekstovi sadrže pitanja i odgovore."),
        default_prompt_template=(
            "Tekst: {text}\nPitanje: {question}\nOdgovor s najviše 3 riječi: {label}"
        ),
        default_instruction_prompt=(
            "Tekst: {text}\n\n"
            "Odgovorite na sljedeće pitanje o gornjem tekstu s najviše 3 riječi.\n\n"
            "Pitanje: {question}"
        ),
        default_prompt_label_mapping=dict(),
    ),
    CZECH: PromptConfig(
        default_prompt_prefix="Následující texty obsahují otázky a odpovědi.",
        default_prompt_template=(
            "Text: {text}\nOtázka: {question}\nOdpověď maximálně 3 slovy: {label}"
        ),
        default_instruction_prompt=(
            "Text: {text}\n\n"
            "Odpovězte na následující otázku k výše uvedenému textu "
            "maximálně 3 slovy.\n\n"
            "Otázka: {question}"
        ),
        default_prompt_label_mapping=dict(),
    ),
    DANISH: PromptConfig(
        default_prompt_prefix="Følgende er tekster med tilhørende spørgsmål og svar.",
        default_prompt_template="Tekst: {text}\nSpørgsmål: {question}\nSvar med maks. "
        "3 ord: {label}",
        default_instruction_prompt="Tekst: {text}\n\nBesvar følgende spørgsmål om "
        "teksten ovenfor med maks. 3 ord.\n\nSpørgsmål: {question}",
        default_prompt_label_mapping=dict(),
    ),
    GERMAN: PromptConfig(
        default_prompt_prefix="Im Folgenden finden Sie Texte mit den dazugehörigen "
        "Fragen und Antworten.",
        default_prompt_template="Text: {text}\nFragen: {question}\nFragen Antwort in "
        "maximal 3 Wörtern: {label}",
        default_instruction_prompt="Text: {text}\n\nBeantworten Sie die folgende Frage "
        "zum obigen Text in höchstens 3 Wörtern.\n\nFrage: {question}",
        default_prompt_label_mapping=dict(),
    ),
    GREEK: PromptConfig(
        default_prompt_prefix="Ακολουθούν κείμενα με τις αντίστοιχες ερωτήσεις και "
        "απαντήσεις.",
        default_prompt_template="Κείμενο: {text}\nΕρώτηση: {question}\nΑπάντηση σε το "
        "πολύ 3 λέξεις: {label}",
        default_instruction_prompt="Κείμενο: {text}\n\nΑπαντήστε την παρακάτω ερώτηση "
        "σχετικά με το παραπάνω κείμενο σε το πολύ 3 λέξεις.\n\nΕρώτηση: {question}",
        default_prompt_label_mapping=dict(),
    ),
    ENGLISH: PromptConfig(
        default_prompt_prefix="The following are texts with accompanying questions and "
        "answers.",
        default_prompt_template="Text: {text}\nQuestion: {question}\nAnswer in max "
        "3 words: {label}",
        default_instruction_prompt="Text: {text}\n\nAnswer the following question "
        "about the above text in at most 3 words.\n\nQuestion: {question}",
        default_prompt_label_mapping=dict(),
    ),
    SPANISH: PromptConfig(
        default_prompt_prefix="A continuación se presentan textos con sus preguntas y "
        "respuestas correspondientes.",
        default_prompt_template="Texto: {text}\nPregunta: {question}\nRespuesta en "
        "máximo 3 palabras: {label}",
        default_instruction_prompt="Texto: {text}\n\nResponda la siguiente pregunta "
        "sobre el texto anterior en máximo 3 palabras.\n\nPregunta: {question}",
        default_prompt_label_mapping=dict(),
    ),
    ESTONIAN: PromptConfig(
        default_prompt_prefix="Järgnevad on tekstid koos küsimuste ja vastustega.",
        default_prompt_template="Tekst: {text}\nKüsimus: {question}\nVasta "
        "maksimaalselt 3 sõnaga: {label}",
        default_instruction_prompt="Tekst: {text}\n\nVasta järgmisele küsimusele "
        "ülevaltoodud teksti kohta maksimaalselt 3 sõnaga.\n\nKüsimus: {question}",
        default_prompt_label_mapping=dict(),
    ),
    FINNISH: PromptConfig(
        default_prompt_prefix="Seuraavassa on tekstejä ja niihin liittyviä kysymyksiä "
        "ja vastauksia.",
        default_prompt_template="Teksti: {text}\nKysymys: {question} "
        "\nVastaa enintään 3 sanalla: {label}",
        default_instruction_prompt="Teksti: {text}\n\nVastaa seuraavaan "
        "kysymykseen yllä olevasta tekstistä enintään 3 sanalla.\n\n"
        "Kysymys: {question}",
        default_prompt_label_mapping=dict(),
    ),
    FAROESE: PromptConfig(
        default_prompt_prefix="Hetta eru tekstir saman við spurningum og svar.",
        default_prompt_template="Tekstur: {text}\nSpurningur: {question}\nSvara við í "
        "mesta lagi trimum orðum: {label}",
        default_instruction_prompt="Tekstur: {text}\n\nSvara hesum spurninginum um "
        "tekstin uppiyvir við í mesta lagi trimum orðum.\n\nSpurningur: {question}",
        default_prompt_label_mapping=dict(),
    ),
    FRENCH: PromptConfig(
        default_prompt_prefix="Les textes suivants sont accompagnés de questions et de "
        "réponses.",
        default_prompt_template="Texte: {text}\nQuestion: {question}\nRéponse en 3 "
        "mots maximum: {label}",
        default_instruction_prompt="Texte: {text}\n\nRépondez à la question suivante "
        "sur le texte ci-dessus en 3 mots maximum.\n\nQuestion: {question}",
        default_prompt_label_mapping=dict(),
    ),
    ICELANDIC: PromptConfig(
        default_prompt_prefix="Eftirfarandi eru textar með tilheyrandi spurningum og "
        "svörum.",
        default_prompt_template="Texti: {text}\nSpurning: {question}\nSvaraðu með að "
        "hámarki 3 orðum: {label}",
        default_instruction_prompt="Texti: {text}\n\nSvaraðu eftirfarandi spurningu um "
        "textann að hámarki í 3 orðum.\n\nSpurning: {question}",
        default_prompt_label_mapping=dict(),
    ),
    ITALIAN: PromptConfig(
        default_prompt_prefix="I testi che seguono sono accompagnati da domande e "
        "risposte.",
        default_prompt_template="Testo: {text}\nDomanda: {question}\nRispondere in "
        "massimo 3 parole: {label}",
        default_instruction_prompt="Testo: {text}\n\nRispondi alla seguente domanda "
        "sul in un massimo di 3 parole.\n\nDomanda: {question}",
        default_prompt_label_mapping=dict(),
    ),
    LITHUANIAN: PromptConfig(
        default_prompt_prefix="Toliau pateikti tekstai su atitinkamais klausimais ir "
        "atsakymais.",
        default_prompt_template="Tekstas: {text}\nKlausimas: {question}\nAtsakykite ne "
        "daugiau kaip 3 žodžiais: {label}",
        default_instruction_prompt="Tekstas: {text}\n\nAtsakykite į šį klausimą apie "
        "aukščiau pateiktą tekstą ne daugiau kaip 3 žodžiais.\n\nKlausimas: {question}",
        default_prompt_label_mapping=dict(),
    ),
    LATVIAN: PromptConfig(
        default_prompt_prefix="Turpmāk seko teksti ar atbilstošiem jautājumiem un "
        "atbildēm.",
        default_prompt_template="Teksts: {text}\nJautājums: {question}\nAtbildēt ar "
        "maksimāli 3 vārdiem: {label}",
        default_instruction_prompt="Teksts: {text}\n\nAtbildiet uz šo jautājumu par "
        "iepriekš minēto tekstu ar maksimāli 3 vārdiem.\n\nJautājums: {question}",
        default_prompt_label_mapping=dict(),
    ),
    NORWEGIAN_BOKMÅL: PromptConfig(
        default_prompt_prefix="Her følger tekster med tilhørende spørsmål og svar.",
        default_prompt_template="Tekst: {text}\nSpørsmål: {question}\nSvar på maks 3 "
        "ord: {label}",
        default_instruction_prompt="Tekst: {text}\n\nBesvar følgende spørsmål om "
        "teksten ovenfor med maks 3 ord.\n\nSpørsmål: {question}",
        default_prompt_label_mapping=dict(),
    ),
    DUTCH: PromptConfig(
        default_prompt_prefix="Hieronder volgen teksten met bijbehorende vragen en "
        "antwoorden.",
        default_prompt_template="Tekst: {text}\nVraag: {question}\nAntwoord in max "
        "3 woorden: {label}",
        default_instruction_prompt="Tekst: {text}\n\nBeantwoord de volgende vraag "
        "over de bovenstaande tekst in maximaal 3 woorden.\n\nVraag: {question}",
        default_prompt_label_mapping=dict(),
    ),
    NORWEGIAN_NYNORSK: PromptConfig(
        default_prompt_prefix="Her følger tekster med tilhørende spørsmål og svar.",
        default_prompt_template="Tekst: {text}\nSpørsmål: {question}\nSvar på maks 3 "
        "ord: {label}",
        default_instruction_prompt="Tekst: {text}\n\nBesvar følgende spørsmål om "
        "teksten ovenfor med maks 3 ord.\n\nSpørsmål: {question}",
        default_prompt_label_mapping=dict(),
    ),
    NORWEGIAN: PromptConfig(
        default_prompt_prefix="Her følger tekster med tilhørende spørsmål og svar.",
        default_prompt_template="Tekst: {text}\nSpørsmål: {question}\nSvar på maks 3 "
        "ord: {label}",
        default_instruction_prompt="Tekst: {text}\n\nBesvar følgende spørsmål om "
        "teksten ovenfor med maks 3 ord.\n\nSpørsmål: {question}",
        default_prompt_label_mapping=dict(),
    ),
    POLISH: PromptConfig(
        default_prompt_prefix=(
            "Poniżej znajdują się teksty z towarzyszącymi pytaniami i odpowiedziami."
        ),
        default_prompt_template="Tekst: {text}\nPytanie: {question}\nOdpowiedź z "
        "użyciem maksymalnie 3 słów: {label}",
        default_instruction_prompt="Tekst: {text}\n\nOdpowiedz na następujące pytanie "
        "dotyczące powyższego tekstu, używając maksymalnie 3 słów.\n\nPytanie: "
        "{question}",
        default_prompt_label_mapping=dict(),
    ),
    PORTUGUESE: PromptConfig(
        default_prompt_prefix="Os textos que se seguem são acompanhados de perguntas "
        "e respostas.",
        default_prompt_template="Texto: {text}\nPergunta: {question}\nResposta com "
        "um máximo de 3 palavras: {label}",
        default_instruction_prompt="Texto: {text}\n\nResponde à seguinte pergunta "
        "sobre o texto acima num máximo de 3 palavras.\n\nPergunta: {question}",
        default_prompt_label_mapping=dict(),
    ),
    SLOVENIAN: PromptConfig(
        default_prompt_prefix=(
            "Spodaj so besedila z ustreznimi vprašanji in odgovori."
        ),
        default_prompt_template=(
            "Besedilo: {text}\n"
            "Vprašanje: {question}\n"
            "Odgovor v največ 3 besedah: {label}"
        ),
        default_instruction_prompt=(
            "Besedilo: {text}\n\n"
            "Odgovorite na naslednje vprašanje o zgornjem besedilu "
            "v največ 3 besedah.\n\n"
            "Vprašanje: {question}"
        ),
        default_prompt_label_mapping=dict(),
    ),
    SLOVAK: PromptConfig(
        default_prompt_prefix=("Nasledujú texty s pridruženými otázkami a odpoveďami."),
        default_prompt_template=(
            "Text: {text}\nOtázka: {question}\nOdpoveď na maximálne 3 slová: {label}"
        ),
        default_instruction_prompt=(
            "Text: {text}\n\n"
            "Odpovedzte na nasledujúcu otázku týkajúcu sa textu uvedeného vyššie "
            "maximálne 3 slovami.\n\nOtázka: {question}"
        ),
        default_prompt_label_mapping=dict(),
    ),
    SERBIAN: PromptConfig(
        default_prompt_prefix=(
            "Следе текстови са одговарајућим питањима и одговорима."
        ),
        default_prompt_template=(
            "Текст: {text}\nПитање: {question}\nОдговор у максимум 3 речи: {label}"
        ),
        default_instruction_prompt=(
            "Текст: {text}\n\n"
            "Одговорите на следеће питање о горњем тексту у максимум 3 речи.\n\n"
            "Питање: {question}"
        ),
        default_prompt_label_mapping=dict(),
    ),
    SWEDISH: PromptConfig(
        default_prompt_prefix="Nedan följer texter med tillhörande frågor och svar.",
        default_prompt_template="Text: {text}\nFråga: {question}\nSvar på max 3 ord: "
        "{label}",
        default_instruction_prompt="Text: {text}\n\nBesvara följande fråga om texten "
        "ovan med högst 3 ord.\n\nFråga: {question}",
        default_prompt_label_mapping=dict(),
    ),
    UKRAINIAN: PromptConfig(
        default_prompt_prefix=(
            "Нижче наведені тексти з відповідними питаннями та відповідями."
        ),
        default_prompt_template=(
            "Текст: {text}\nПитання: {question}\nВідповідь максимум 3 словами: {label}"
        ),
        default_instruction_prompt=(
            "Текст: {text}\n\n"
            "Відповідь на наступне питання про вищезазначений текст максимум 3 "
            "словами.\n\nПитання: {question}"
        ),
        default_prompt_label_mapping=dict(),
    ),
}
