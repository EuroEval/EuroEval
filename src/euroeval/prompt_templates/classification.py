"""Templates for the classification task."""

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

CLASSIFICATION_TEMPLATES: dict["Language", PromptConfig] = {
    ENGLISH: PromptConfig(
        default_prompt_prefix="The following are texts and their labels.",
        default_prompt_template="Text: {text}\nLabel: {label}",
        default_instruction_prompt="Here is a text:\n'{text}'.\n\nClassify the text "
        "into the categories {labels_str}, and reply with only the label.",
        default_prompt_label_mapping="auto",
    ),
    BULGARIAN: PromptConfig(
        default_prompt_prefix="Следват текстове и техните етикети.",
        default_prompt_template="Текст: {text}\nЕтикет: {label}",
        default_instruction_prompt="Ето един текст:\n'{text}'.\n\nКласифицирайте "
        "текста в категориите {labels_str} и отговорете само с етикета.",
        default_prompt_label_mapping="auto",
    ),
    CZECH: PromptConfig(
        default_prompt_prefix="Následují texty a jejich štítky.",
        default_prompt_template="Text: {text}\nŠtítek: {label}",
        default_instruction_prompt="Zde je text:\n'{text}'.\n\nKlasifikujte text do "
        "kategorií {labels_str} a odpovězte pouze štítkem.",
        default_prompt_label_mapping="auto",
    ),
    DANISH: PromptConfig(
        default_prompt_prefix="Følgende er tekster og deres etiketter.",
        default_prompt_template="Tekst: {text}\nEtiket: {label}",
        default_instruction_prompt="Her er en tekst:\n'{text}'.\n\nKlassificer teksten "
        "i kategorierne {labels_str}, og svar kun med etiketten.",
        default_prompt_label_mapping="auto",
    ),
    GERMAN: PromptConfig(
        default_prompt_prefix="Im Folgenden sind Texte und ihre Labels aufgeführt.",
        default_prompt_template="Text: {text}\nLabel: {label}",
        default_instruction_prompt="Hier ist ein Text:\n'{text}'.\n\nKlassifiziere den "
        "Text in die Kategorien {labels_str} und antworte nur mit dem Label.",
        default_prompt_label_mapping="auto",
    ),
    GREEK: PromptConfig(
        default_prompt_prefix="Ακολουθούν κείμενα και οι ετικέτες τους.",
        default_prompt_template="Κείμενο: {text}\nΕτικέτα: {label}",
        default_instruction_prompt="Εδώ είναι ένα κείμενο:\n'{text}'.\n\n"
        "Κατηγοριοποιήστε το κείμενο στις κατηγορίες {labels_str} και απαντήστε μόνο "
        "με την ετικέτα.",
        default_prompt_label_mapping="auto",
    ),
    SPANISH: PromptConfig(
        default_prompt_prefix="A continuación se presentan textos y sus etiquetas.",
        default_prompt_template="Texto: {text}\nEtiqueta: {label}",
        default_instruction_prompt="Aquí hay un texto:\n'{text}'.\n\nClasifica el "
        "texto en las categorías {labels_str} y responde solo con la etiqueta.",
        default_prompt_label_mapping="auto",
    ),
    ESTONIAN: PromptConfig(
        default_prompt_prefix="Järgnevad on tekstid ja nende sildid.",
        default_prompt_template="Tekst: {text}\nSilt: {label}",
        default_instruction_prompt="Siin on tekst:\n'{text}'.\n\nKlassifitseeri tekst "
        "kategooriatesse {labels_str} ja vasta ainult sildiga.",
        default_prompt_label_mapping="auto",
    ),
    FINNISH: PromptConfig(
        default_prompt_prefix="Seuraavassa on tekstejä ja niiden tunnisteita.",
        default_prompt_template="Teksti: {text}\nTunniste: {label}",
        default_instruction_prompt="Tässä on teksti:\n'{text}'.\n\nLuokittele teksti "
        "kategorioihin {labels_str} ja vastaa vain tunnisteella.",
        default_prompt_label_mapping="auto",
    ),
    FAROESE: PromptConfig(
        default_prompt_prefix="Hér eru tekster og teirra etikettir.",
        default_prompt_template="Tekstur: {text}\nEtikettur: {label}",
        default_instruction_prompt="Her er ein tekstur:\n'{text}'.\n\nFlokka teksturin "
        "í bólkar {labels_str} og svara bert við etikettinum.",
        default_prompt_label_mapping="auto",
    ),
    FRENCH: PromptConfig(
        default_prompt_prefix="Voici des textes et leurs étiquettes.",
        default_prompt_template="Texte : {text}\nÉtiquette : {label}",
        default_instruction_prompt="Voici un texte :\n'{text}'.\n\nClassifiez le texte "
        "dans les catégories {labels_str} et répondez uniquement avec l'étiquette.",
        default_prompt_label_mapping="auto",
    ),
    ICELANDIC: PromptConfig(
        default_prompt_prefix="Hér fyrir neðan eru textar og merkingar þeirra.",
        default_prompt_template="Texti: {text}\nMerking: {label}",
        default_instruction_prompt="Hér er texti:\n'{text}'.\n\nFlokkaðu textann "
        "í flokkana {labels_str} og svaraðu aðeins með merkingenni.",
        default_prompt_label_mapping="auto",
    ),
    ITALIAN: PromptConfig(
        default_prompt_prefix="Di seguito sono riportati testi e le loro etichette.",
        default_prompt_template="Testo: {text}\nEtichetta: {label}",
        default_instruction_prompt="Ecco un testo:\n'{text}'.\n\nClassifica il testo "
        "nelle categorie {labels_str} e rispondi solo con l'etichetta.",
        default_prompt_label_mapping="auto",
    ),
    LITHUANIAN: PromptConfig(
        default_prompt_prefix="Toliau pateikiami tekstai ir jų etiketės.",
        default_prompt_template="Tekstas: {text}\nEtiketė: {label}",
        default_instruction_prompt="Štai tekstas:\n'{text}'.\n\nKlasifikuokite tekstą "
        "į kategorijas {labels_str} ir atsakykite tik etiketę.",
        default_prompt_label_mapping="auto",
    ),
    LATVIAN: PromptConfig(
        default_prompt_prefix="Turpmāk ir teksti un to etiķetes.",
        default_prompt_template="Teksts: {text}\nEtiķete: {label}",
        default_instruction_prompt="Šeit ir teksts:\n'{text}'.\n\nKlasificējiet tekstu "
        "kategorijās {labels_str} un atbildiet tikai ar etiķeti.",
        default_prompt_label_mapping="auto",
    ),
    NORWEGIAN_BOKMÅL: PromptConfig(
        default_prompt_prefix="Følgende er tekster og deres etiketter.",
        default_prompt_template="Tekst: {text}\nEtikett: {label}",
        default_instruction_prompt="Her er en tekst:\n'{text}'.\n\nKlassifiser teksten "
        "i kategoriene {labels_str}, og svar kun med etiketten.",
        default_prompt_label_mapping="auto",
    ),
    DUTCH: PromptConfig(
        default_prompt_prefix="Hieronder volgen teksten en hun labels.",
        default_prompt_template="Tekst: {text}\nLabel: {label}",
        default_instruction_prompt="Hier is een tekst:\n'{text}'.\n\nClassificeer de "
        "tekst in de categorieën {labels_str} en antwoord alleen met het label.",
        default_prompt_label_mapping="auto",
    ),
    NORWEGIAN_NYNORSK: PromptConfig(
        default_prompt_prefix="Følgjande er tekstar og deira etikettar.",
        default_prompt_template="Tekst: {text}\nEtikett: {label}",
        default_instruction_prompt="Her er ein tekst:\n'{text}'.\n\nKlassifiser "
        "teksten i kategoriane {labels_str}, og svar berre med etiketten.",
        default_prompt_label_mapping="auto",
    ),
    NORWEGIAN: PromptConfig(
        default_prompt_prefix="Følgende er tekster og deres etiketter.",
        default_prompt_template="Tekst: {text}\nEtikett: {label}",
        default_instruction_prompt="Her er en tekst:\n'{text}'.\n\nKlassifiser teksten "
        "i kategoriene {labels_str}, og svar kun med etiketten.",
        default_prompt_label_mapping="auto",
    ),
    POLISH: PromptConfig(
        default_prompt_prefix="Poniżej znajdują się teksty i ich etykiety.",
        default_prompt_template="Tekst: {text}\nEtykieta: {label}",
        default_instruction_prompt="Oto tekst:\n'{text}'.\n\nSklasyfikuj tekst do "
        "kategorii {labels_str} i odpowiedz tylko etykietą.",
        default_prompt_label_mapping="auto",
    ),
    PORTUGUESE: PromptConfig(
        default_prompt_prefix="A seguir estão textos e seus rótulos.",
        default_prompt_template="Texto: {text}\nRótulo: {label}",
        default_instruction_prompt="Aqui está um texto:\n'{text}'.\n\nClassifique o "
        "texto nas categorias {labels_str} e responda apenas com o rótulo.",
        default_prompt_label_mapping="auto",
    ),
    SLOVAK: PromptConfig(
        default_prompt_prefix="Nasledujú texty a ich štítky.",
        default_prompt_template="Text: {text}\nŠtítok: {label}",
        default_instruction_prompt="Tu je text:\n'{text}'.\n\nKlasifikujte text do "
        "kategorií {labels_str} a odpovedzte iba štítkom.",
        default_prompt_label_mapping="auto",
    ),
    SWEDISH: PromptConfig(
        default_prompt_prefix="Följande är texter och deras etiketter.",
        default_prompt_template="Text: {text}\nEtikett: {label}",
        default_instruction_prompt="Här är en text:\n'{text}'.\n\nKlassificera texten "
        "i kategorierna {labels_str} och svara endast med etiketten.",
        default_prompt_label_mapping="auto",
    ),
    UKRAINIAN: PromptConfig(
        default_prompt_prefix="Нижче наведено тексти та їхні позначки.",
        default_prompt_template="Текст: {text}\nПозначка: {label}",
        default_instruction_prompt="Ось текст:\n'{text}'.\n\nКласифікуйте текст у "
        "категорії {labels_str} і відповідайте лише позначкою.",
        default_prompt_label_mapping="auto",
    ),
}
