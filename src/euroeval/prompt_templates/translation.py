"""Templates for the translation task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import (
    ALBANIAN,
    BOSNIAN,
    CATALAN,
    CZECH,
    DANISH,
    DUTCH,
    ENGLISH,
    ESTONIAN,
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
    POLISH,
    PORTUGUESE,
    ROMANIAN,
    SERBIAN,
    SPANISH,
    SWEDISH,
    UKRAINIAN,
    get_all_languages,
)

if t.TYPE_CHECKING:
    from ..data_models import Language


TRANSLATION_TEMPLATES: dict[tuple["Language", "Language"], PromptConfig] = {
    **{
        (ENGLISH, language): PromptConfig(
            default_prompt_prefix=(
                f"The following are English texts with corresponding {language.name} "
                "translations."
            ),
            default_prompt_template=(
                f"English text: {{text}}\n{language.name} translation: {{target_text}}"
            ),
            default_instruction_prompt=(
                "English text: {text}\n\n"
                f"Translate the above text into {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != ENGLISH
    },
    **{
        (DANISH, language): PromptConfig(
            default_prompt_prefix=(
                f"Nedenfor er danske tekster med tilhørende {language.name} "
                "oversættelser."
            ),
            default_prompt_template=(
                f"Dansk tekst: {{text}}\n{language.name} oversættelse: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Dansk tekst: {text}\n\n"
                f"Oversæt den ovenstående tekst til {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != DANISH
    },
    **{
        (ALBANIAN, language): PromptConfig(
            default_prompt_prefix=(
                "Më poshtë janë tekste në shqip me përkthimet përkatëse në "
                f"{language.name}."
            ),
            default_prompt_template=(
                f"Tekst shqip: {{text}}\nPërkthim në {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Tekst shqip: {text}\n\n"
                f"Përkthejeni tekstin e mësipërm në {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != ALBANIAN
    },
    **{
        (BOSNIAN, language): PromptConfig(
            default_prompt_prefix=(
                "Sljedeći tekstovi su na bosanskom jeziku sa odgovarajućim prevodima "
                f"na {language.name}."
            ),
            default_prompt_template=(
                "Tekst na bosanskom: {text}\n"
                f"Prevod na {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Tekst na bosanskom: {text}\n\n"
                f"Prevedite gornji tekst na {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != BOSNIAN
    },
    **{
        (CATALAN, language): PromptConfig(
            default_prompt_prefix=(
                "A continuació es mostren textos en català amb les traduccions "
                f"corresponents a {language.name}."
            ),
            default_prompt_template=(
                "Text en català: {text}\n"
                f"Traducció a {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Text en català: {text}\n\n"
                f"Traduïu el text anterior a {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != CATALAN
    },
    **{
        (CZECH, language): PromptConfig(
            default_prompt_prefix=(
                "Níže jsou uvedeny české texty s odpovídajícími překlady do "
                f"{language.name}."
            ),
            default_prompt_template=(
                f"Český text: {{text}}\nPřeklad do {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Český text: {text}\n\n"
                f"Přeložte výše uvedený text do {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != CZECH
    },
    **{
        (DUTCH, language): PromptConfig(
            default_prompt_prefix=(
                "Hieronder volgen Nederlandse teksten met bijbehorende vertalingen "
                f"in het {language.name}."
            ),
            default_prompt_template=(
                "Nederlandse tekst: {text}\n"
                f"Vertaling in het {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Nederlandse tekst: {text}\n\n"
                f"Vertaal de bovenstaande tekst naar het {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != DUTCH
    },
    **{
        (ESTONIAN, language): PromptConfig(
            default_prompt_prefix=(
                "Järgnevad on eestikeelsed tekstid koos vastavate tõlgetega "
                f"keelde {language.name}."
            ),
            default_prompt_template=(
                "Eestikeelne tekst: {text}\n"
                f"Tõlge keelde {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Eestikeelne tekst: {text}\n\n"
                f"Tõlkige ülaltoodud tekst keelde {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != ESTONIAN
    },
    **{
        (FINNISH, language): PromptConfig(
            default_prompt_prefix=(
                "Seuraavassa on suomenkielisiä tekstejä ja niiden käännökset "
                f"kielelle {language.name}."
            ),
            default_prompt_template=(
                "Suomenkielinen teksti: {text}\n"
                f"Käännös kielelle {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Suomenkielinen teksti: {text}\n\n"
                f"Käännä yllä oleva teksti kielelle {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != FINNISH
    },
    **{
        (FRENCH, language): PromptConfig(
            default_prompt_prefix=(
                "Ce qui suit sont des textes en français avec les traductions "
                f"correspondantes en {language.name}."
            ),
            default_prompt_template=(
                "Texte en français: {text}\n"
                f"Traduction en {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Texte en français : {text}\n\n"
                f"Traduisez le texte ci-dessus en {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != FRENCH
    },
    **{
        (GERMAN, language): PromptConfig(
            default_prompt_prefix=(
                "Im Folgenden finden Sie deutsche Texte mit entsprechenden "
                f"Übersetzungen ins {language.name}."
            ),
            default_prompt_template=(
                "Deutscher Text: {text}\n"
                f"Übersetzung ins {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Deutscher Text: {text}\n\n"
                f"Übersetzen Sie den oben stehenden Text ins {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != GERMAN
    },
    **{
        (GREEK, language): PromptConfig(
            default_prompt_prefix=(
                "Ακολουθούν ελληνικά κείμενα με αντίστοιχες μεταφράσεις "
                f"στα {language.name}."
            ),
            default_prompt_template=(
                "Ελληνικό κείμενο: {text}\n"
                f"Μετάφραση στα {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Ελληνικό κείμενο: {text}\n\n"
                f"Μεταφράστε το παραπάνω κείμενο στα {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != GREEK
    },
    **{
        (HUNGARIAN, language): PromptConfig(
            default_prompt_prefix=(
                f"Az alábbiak magyar szövegek a megfelelő {language.name} "
                "fordításokkal."
            ),
            default_prompt_template=(
                f"Magyar szöveg: {{text}}\n{language.name} fordítás: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Magyar szöveg: {text}\n\n"
                f"Fordítsa le a fenti szöveget {language.name} nyelvre."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != HUNGARIAN
    },
    **{
        (ICELANDIC, language): PromptConfig(
            default_prompt_prefix=(
                "Hér á eftir fara íslenskir textar með samsvarandi þýðingum "
                f"á {language.name}."
            ),
            default_prompt_template=(
                f"Íslenskur texti: {{text}}\nÞýðing á {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Íslenskur texti: {text}\n\n"
                f"Þýddu textann hér að ofan á {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != ICELANDIC
    },
    **{
        (ITALIAN, language): PromptConfig(
            default_prompt_prefix=(
                "Di seguito sono riportati testi in italiano con le corrispondenti "
                f"traduzioni in {language.name}."
            ),
            default_prompt_template=(
                "Testo in italiano: {text}\n"
                f"Traduzione in {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Testo in italiano: {text}\n\n"
                f"Traduci il testo precedente in {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != ITALIAN
    },
    **{
        (LATVIAN, language): PromptConfig(
            default_prompt_prefix=(
                "Tālāk ir doti teksti latviešu valodā ar atbilstošiem "
                f"tulkojumiem {language.name} valodā."
            ),
            default_prompt_template=(
                "Teksts latviešu valodā: {text}\n"
                f"Tulkojums {language.name} valodā: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Teksts latviešu valodā: {text}\n\n"
                f"Tulkojiet iepriekš minēto tekstu {language.name} valodā."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != LATVIAN
    },
    **{
        (LITHUANIAN, language): PromptConfig(
            default_prompt_prefix=(
                "Žemiau pateikiami tekstai lietuvių kalba su atitinkamais "
                f"vertimais į {language.name}."
            ),
            default_prompt_template=(
                "Tekstas lietuvių kalba: {text}\n"
                f"Vertimas į {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Tekstas lietuvių kalba: {text}\n\n"
                f"Išverskite aukščiau esantį tekstą į {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != LITHUANIAN
    },
    **{
        (NORWEGIAN, language): PromptConfig(
            default_prompt_prefix=(
                "Følgende er norske tekster med tilsvarende oversettelser "
                f"til {language.name}."
            ),
            default_prompt_template=(
                "Norsk tekst: {text}\n"
                f"Oversettelse til {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Norsk tekst: {text}\n\n"
                f"Oversett teksten ovenfor til {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != NORWEGIAN
    },
    **{
        (POLISH, language): PromptConfig(
            default_prompt_prefix=(
                "Poniżej znajdują się teksty w języku polskim wraz z odpowiednimi "
                f"tłumaczeniami na {language.name}."
            ),
            default_prompt_template=(
                "Tekst w języku polskim: {text}\n"
                f"Tłumaczenie na {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Tekst w języku polskim: {text}\n\n"
                f"Przetłumacz powyższy tekst na {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != POLISH
    },
    **{
        (PORTUGUESE, language): PromptConfig(
            default_prompt_prefix=(
                "Abaixo estão textos em português com as traduções "
                f"correspondentes em {language.name}."
            ),
            default_prompt_template=(
                "Texto em português: {text}\n"
                f"Tradução em {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Texto em português: {text}\n\n"
                f"Traduza o texto acima para {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != PORTUGUESE
    },
    **{
        (ROMANIAN, language): PromptConfig(
            default_prompt_prefix=(
                "Următoarele sunt texte în limba română cu traduceri "
                f"corespunzătoare în {language.name}."
            ),
            default_prompt_template=(
                "Text în română: {text}\n"
                f"Traducere în {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Text în română: {text}\n\n"
                f"Traduceți textul de mai sus în {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != ROMANIAN
    },
    **{
        (SERBIAN, language): PromptConfig(
            default_prompt_prefix=(
                "Slede tekstovi na srpskom sa odgovarajućim prevodima "
                f"na {language.name}."
            ),
            default_prompt_template=(
                "Tekst na srpskom: {text}\n"
                f"Prevod na {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Tekst na srpskom: {text}\n\n"
                f"Prevedite gornji tekst na {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != SERBIAN
    },
    **{
        (SPANISH, language): PromptConfig(
            default_prompt_prefix=(
                "A continuación se presentan textos en español con las "
                f"traducciones correspondientes a {language.name}."
            ),
            default_prompt_template=(
                "Texto en español: {text}\n"
                f"Traducción a {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Texto en español: {text}\n\n"
                f"Traduce el texto anterior a {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != SPANISH
    },
    **{
        (SWEDISH, language): PromptConfig(
            default_prompt_prefix=(
                "Följande är svenska texter med motsvarande översättningar "
                f"till {language.name}."
            ),
            default_prompt_template=(
                "Svensk text: {text}\n"
                f"Översättning till {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                f"Svensk text: {{text}}\n\nÖversätt texten ovan till {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != SWEDISH
    },
    **{
        (UKRAINIAN, language): PromptConfig(
            default_prompt_prefix=(
                "Нижче наведені тексти українською мовою з відповідними "
                f"перекладами на {language.name}."
            ),
            default_prompt_template=(
                "Текст українською: {text}\n"
                f"Переклад на {language.name}: {{target_text}}"
            ),
            default_instruction_prompt=(
                "Текст українською: {text}\n\n"
                f"Перекладіть наведений вище текст на {language.name}."
            ),
            default_prompt_label_mapping=dict(),
        )
        for language in get_all_languages().values()
        if language != UKRAINIAN
    },
}
