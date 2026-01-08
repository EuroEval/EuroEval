"""Language pairs for machine translation."""

from typing import NamedTuple

from .languages import DANISH, UKRAINIAN, Language


class LanguagePair(NamedTuple):
    """A language pair for machine translation."""

    source: Language
    target: Language


DANISH_TO_UKRAINIAN = LanguagePair(source=DANISH, target=UKRAINIAN)
UKRAINIAN_TO_DANISH = LanguagePair(source=UKRAINIAN, target=DANISH)
