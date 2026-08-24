"""Transliteration of text between scripts.

Some languages are written in more than one script (e.g., Serbian, which is written in
both Cyrillic and Latin). When evaluating with character-based metrics such as ChrF, a
model may translate into a valid script that differs from the reference's script, which
would collapse the score even though the translation is correct. This module normalises
text to a single canonical script for such languages, keyed off the `multiple_scripts`
attribute of the `Language` object.
"""

import collections.abc as c

import cyrtranslit

from .exceptions import InvalidBenchmark
from .languages import BOSNIAN, SERBIAN, Language


def _serbo_croatian_to_latin(text: str) -> str:
    """Transliterate Serbo-Croatian text to Latin script.

    Serbian, Bosnian, Croatian and Montenegrin share the same Cyrillic-Latin
    correspondence, so the Serbian ('sr') mapping handles all of them.

    Args:
        text:
            The text to transliterate.

    Returns:
        The text with any Cyrillic characters transliterated to Latin.
    """
    return cyrtranslit.to_latin(text, "sr")


# Maps each language written in multiple scripts to the function that normalises its
# text to a single canonical script. Every language flagged with `multiple_scripts=True`
# that is actually transliterated must appear here, or `transliterate` raises.
TRANSLITERATION_FUNCTIONS: dict[Language, c.Callable[[str], str]] = {
    SERBIAN: _serbo_croatian_to_latin,
    BOSNIAN: _serbo_croatian_to_latin,
}


def transliterate(text: str, language: Language) -> str:
    """Normalise text to the canonical script of a multiple-script language.

    Args:
        text:
            The text to transliterate.
        language:
            The language of the text. This is expected to be a language written in
            multiple scripts (i.e., `language.multiple_scripts` is True).

    Returns:
        The text transliterated to the language's canonical script.

    Raises:
        InvalidBenchmark:
            If the language has no registered transliteration function.
    """
    if language not in TRANSLITERATION_FUNCTIONS:
        raise InvalidBenchmark(
            f"The language {language.name!r} is written in multiple scripts but has no "
            "registered transliteration function. Please add one to "
            "`TRANSLITERATION_FUNCTIONS` in the `transliteration` module."
        )
    return TRANSLITERATION_FUNCTIONS[language](text)
