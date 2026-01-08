"""Language pairs for machine translation."""

from .languages import DANISH, UKRAINIAN, Language

DANISH_TO_UKRAINIAN: tuple[Language, Language] = (DANISH, UKRAINIAN)
UKRAINIAN_TO_DANISH: tuple[Language, Language] = (UKRAINIAN, DANISH)
