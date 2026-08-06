"""Tests for the `transliteration` module."""

import pytest

from euroeval.exceptions import InvalidBenchmark
from euroeval.languages import BOSNIAN, SERBIAN, Language
from euroeval.transliteration import transliterate


def test_bosnian_uses_the_serbo_croatian_mapping() -> None:
    """Bosnian shares the Serbo-Croatian mapping, so digraphs transliterate right."""
    assert transliterate(text="Њемачка и џеп", language=BOSNIAN) == "Njemačka i džep"


def test_serbian_cyrillic_is_transliterated_to_latin() -> None:
    """Serbian Cyrillic text should be transliterated to Latin script."""
    assert transliterate(text="Срећна вам субота.", language=SERBIAN) == (
        "Srećna vam subota."
    )


def test_serbian_latin_is_left_unchanged() -> None:
    """Serbian text already in Latin script should be returned unchanged."""
    assert transliterate(text="Srećna vam subota.", language=SERBIAN) == (
        "Srećna vam subota."
    )


def test_unregistered_language_raises() -> None:
    """A multiple-script language with no registered function should raise."""
    unregistered = Language(
        code="xx", name="Testish", _and_separator="and", _or_separator="or"
    )
    with pytest.raises(InvalidBenchmark):
        transliterate(text="whatever", language=unregistered)
