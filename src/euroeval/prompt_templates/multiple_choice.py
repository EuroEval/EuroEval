"""Templates for all multiple choice tasks."""

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

# TODO: Missing Faroese
MULTIPLE_CHOICE_TEMPLATES: dict["Language", PromptConfig] = {
    BULGARIAN: PromptConfig(
        default_prompt_prefix="Следват въпроси с множествен избор (с отговори).",
        default_prompt_template="Въпрос: {text}\nОтговор: {label}",
        default_instruction_prompt="Въпрос: {text}\n\nОтговорете на горния въпрос "
        "като отговорите с {labels_str}, и нищо друго.",
        default_prompt_label_mapping="auto",
    ),
    CROATIAN: PromptConfig(
        default_prompt_prefix=(
            "Sljedeća su pitanja s višestrukim izborom (s odgovorima)."
        ),
        default_prompt_template=("Pitanje: {text}\nOdgovor: {label}"),
        default_instruction_prompt=(
            "Pitanje: {text}\n\n"
            "Odgovorite na gornje pitanje koristeći {labels_str}, i ništa drugo."
        ),
        default_prompt_label_mapping="auto",
    ),
    CZECH: PromptConfig(
        default_prompt_prefix=(
            "Následující jsou otázky s výběrem z více možností (s odpověďmi)."
        ),
        default_prompt_template="Otázka: {text}\nOdpověď: {label}",
        default_instruction_prompt=(
            "Otázka: {text}\n\nOdpovězte na výše uvedenou otázku "
            "pomocí {labels_str}, a nic jiného."
        ),
        default_prompt_label_mapping="auto",
    ),
    DANISH: PromptConfig(
        default_prompt_prefix="Følgende er multiple choice spørgsmål (med svar).",
        default_prompt_template="Spørgsmål: {text}\nSvar: {label}",
        default_instruction_prompt="Spørgsmål: {text}\n\nBesvar ovenstående spørgsmål "
        "ved at svare med {labels_str}, og intet andet.",
        default_prompt_label_mapping="auto",
    ),
    GERMAN: PromptConfig(
        default_prompt_prefix="Die folgenden Fragen sind Multiple-Choice-Fragen "
        "(mit Antworten).",
        default_prompt_template="Frage: {text}\nAntwort: {label}",
        default_instruction_prompt="Frage: {text}\n\nBeantworten Sie die obige Frage "
        "mit {labels_str}, und nichts anderes.",
        default_prompt_label_mapping="auto",
    ),
    GREEK: PromptConfig(
        default_prompt_prefix="Ακολουθούν ερωτήσεις πολλαπλών επιλογών "
        "(με απαντήσεις).",
        default_prompt_template="Ερώτηση: {text}\nΑπάντηση: {label}",
        default_instruction_prompt="Ερώτηση: {text}\n\nΑπαντήστε στην παραπάνω ερώτηση "
        "χρησιμοποιώντας {labels_str}, και τίποτα άλλο.",
        default_prompt_label_mapping="auto",
    ),
    ENGLISH: PromptConfig(
        default_prompt_prefix="The following are multiple choice questions (with "
        "answers).",
        default_prompt_template="Question: {text}\nAnswer: {label}",
        default_instruction_prompt="Question: {text}\n\nAnswer the above question by "
        "replying with {labels_str}, and nothing else.",
        default_prompt_label_mapping="auto",
    ),
    SPANISH: PromptConfig(
        default_prompt_prefix="Las siguientes son preguntas de opción múltiple "
        "(con respuestas).",
        default_prompt_template="Pregunta: {text}\nRespuesta: {label}",
        default_instruction_prompt="Pregunta: {text}\n\nResponda la pregunta anterior "
        "usando solo {labels_str}, y nada más.",
        default_prompt_label_mapping="auto",
    ),
    ESTONIAN: PromptConfig(
        default_prompt_prefix="Järgnevad on vastusevariantidega küsimused (koos "
        "vastustega).",
        default_prompt_template="Küsimus: {text}\nVastus: {label}",
        default_instruction_prompt="Küsimus: {text}\n\nVasta ülaltoodud küsimusele "
        "ainult {labels_str}, ja mitte millegi muuga.",
        default_prompt_label_mapping="auto",
    ),
    PORTUGUESE: PromptConfig(
        default_prompt_prefix="As seguintes são perguntas de escolha múltipla "
        "(com respostas).",
        default_prompt_template="Pergunta: {text}\nResposta: {label}",
        default_instruction_prompt="Pergunta: {text}\n\nResponde à pergunta "
        "acima usando só {labels_str}, e nada mais.",
        default_prompt_label_mapping="auto",
    ),
    FINNISH: PromptConfig(
        default_prompt_prefix="Seuraavat ovat monivalintakysymyksiä (vastauksineen).",
        default_prompt_template="Kysymys: {text}\nVastaus: {label}",
        default_instruction_prompt="Kysymys: {text}\n\nVastaa yllä olevaan kysymykseen "
        "käyttämällä {labels_str}, äläkä mitään muuta.",
        default_prompt_label_mapping="auto",
    ),
    FRENCH: PromptConfig(
        default_prompt_prefix="Les questions suivantes sont des questions à choix "
        "multiples (avec réponses).",
        default_prompt_template="Question: {text}\nRéponse: {label}",
        default_instruction_prompt="Question: {text}\n\nRépondez à la question "
        "ci-dessus par {labels_str}, et rien d'autre.",
        default_prompt_label_mapping="auto",
    ),
    ICELANDIC: PromptConfig(
        default_prompt_prefix="Eftirfarandi eru fjölvalsspurningar (með svörum).",
        default_prompt_template="Spurningar: {text}\nSvara: {label}",
        default_instruction_prompt="Spurningar: {text}\n\nSvaraðu eftirfarandi "
        "spurningum með {labels_str}, og engu öðru.",
        default_prompt_label_mapping="auto",
    ),
    ITALIAN: PromptConfig(
        default_prompt_prefix="Le seguenti sono domande a scelta multipla "
        "(con relative risposte).",
        default_prompt_template="Domanda: {text}\nRisposta: {label}",
        default_instruction_prompt="Domanda: {text}\n\nRispondete alla domanda "
        "precedente con {labels_str}, e nient'altro.",
        default_prompt_label_mapping="auto",
    ),
    LITHUANIAN: PromptConfig(
        default_prompt_prefix="Toliau pateikti daugiavariančiai klausimai "
        "(su atsakymais).",
        default_prompt_template="Klausimas: {text}\nAtsakymas: {label}",
        default_instruction_prompt="Klausimas: {text}\n\nAtsakykite į aukščiau "
        "pateiktą klausimą atsakydami {labels_str}, ir nieko daugiau.",
        default_prompt_label_mapping="auto",
    ),
    LATVIAN: PromptConfig(
        default_prompt_prefix="Tālāk seko jautājumi ar vairākām atbilžu izvēlēm "
        "(ar atbildēm).",
        default_prompt_template="Jautājums: {text}\nAtbilde: {label}",
        default_instruction_prompt="Jautājums: {text}\n\nAtbildiet uz iepriekšējo "
        "jautājumu, atbildot ar {labels_str}, un nekas cits.",
        default_prompt_label_mapping="auto",
    ),
    NORWEGIAN_BOKMÅL: PromptConfig(
        default_prompt_prefix="Følgende er flervalgsspørsmål (med svar).",
        default_prompt_template="Spørsmål: {text}\nSvar: {label}",
        default_instruction_prompt="Spørsmål: {text}\n\nBesvar følgende spørsmål med "
        "{labels_str}, og ikke noe annet.",
        default_prompt_label_mapping="auto",
    ),
    DUTCH: PromptConfig(
        default_prompt_prefix="Hieronder staan meerkeuzevragen (met antwoorden).",
        default_prompt_template="Vraag: {text}\nAntwoord: {label}",
        default_instruction_prompt="Vraag: {text}\n\nBeantwoord de bovenstaande vraag "
        "met {labels_str}, en niets anders.",
        default_prompt_label_mapping="auto",
    ),
    NORWEGIAN_NYNORSK: PromptConfig(
        default_prompt_prefix="Følgende er flervalgsspørsmål (med svar).",
        default_prompt_template="Spørsmål: {text}\nSvar: {label}",
        default_instruction_prompt="Spørsmål: {text}\n\nBesvar følgende spørsmål med "
        "{labels_str}, og ikke noe annet.",
        default_prompt_label_mapping="auto",
    ),
    NORWEGIAN: PromptConfig(
        default_prompt_prefix="Følgende er flervalgsspørsmål (med svar).",
        default_prompt_template="Spørsmål: {text}\nSvar: {label}",
        default_instruction_prompt="Spørsmål: {text}\n\nBesvar følgende spørsmål med "
        "{labels_str}, og ikke noe annet.",
        default_prompt_label_mapping="auto",
    ),
    POLISH: PromptConfig(
        default_prompt_prefix="Poniżej znajdują się pytania wielokrotnego wyboru "
        "(z odpowiedziami).",
        default_prompt_template="Pytanie: {text}\nOdpowiedź: {label}",
        default_instruction_prompt="Pytanie: {text}\n\nOdpowiedz na powyższe pytanie, "
        "używając {labels_str} i niczego więcej.",
        default_prompt_label_mapping="auto",
    ),
    SLOVAK: PromptConfig(
        default_prompt_prefix=(
            "Nasledujú otázky s viacerými možnosťami (s odpoveďami)."
        ),
        default_prompt_template="Otázka: {text}\nOdpoveď: {label}",
        default_instruction_prompt=(
            "Otázka: {text}\n\n"
            "Odpovedzte na nasledujúcu otázku použitím {labels_str}, a nič iné."
        ),
        default_prompt_label_mapping="auto",
    ),
    SLOVENIAN: PromptConfig(
        default_prompt_prefix="Naslednja so vprašanja z več možnostmi (z odgovori).",
        default_prompt_template="Vprašanje: {text}\nOdgovor: {label}",
        default_instruction_prompt=(
            "Vprašanje: {text}\n\n"
            "Odgovorite na navedeno vprašanje z uporabo {labels_str}, in nič drugega."
        ),
        default_prompt_label_mapping="auto",
    ),
    SERBIAN: PromptConfig(
        default_prompt_prefix=("Slede pitanja višestrukog izbora (sa odgovorima)."),
        default_prompt_template="Pitanje: {text}\nOdgovor: {label}",
        default_instruction_prompt=(
            "Pitanje: {text}\n\n"
            "Odgovorite na navedeno pitanje koristeći {labels_str}, i ništa drugo."
        ),
        default_prompt_label_mapping="auto",
    ),
    SWEDISH: PromptConfig(
        default_prompt_prefix="Följande är flervalsfrågor (med svar).",
        default_prompt_template="Fråga: {text}\nSvar: {label}",
        default_instruction_prompt="Fråga: {text}\n\nBesvara följande fråga med "
        "{labels_str}, och inget annat.",
        default_prompt_label_mapping="auto",
    ),
    UKRAINIAN: PromptConfig(
        default_prompt_prefix=(
            "Нижче наведено питання з кількома варіантами відповідей (з відповідями)."
        ),
        default_prompt_template=("Питання: {text}\nВідповідь: {label}"),
        default_instruction_prompt=(
            "Питання: {text}\n\n"
            "Дайте відповідь на наведене вище питання, використовуючи "
            "{labels_str}, і нічого іншого."
        ),
        default_prompt_label_mapping="auto",
    ),
}
