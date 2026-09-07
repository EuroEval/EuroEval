"""Tests for the `languages` module."""

from typing import Generator

import pytest

from euroeval.data_models import Language
from euroeval.languages import (
    ISO_639_3_ALIASES,
    get_all_languages,
    get_correct_language_codes,
    get_language,
    is_language_code,
)


class TestGetAllLanguages:
    """Tests for the `get_all_languages` function."""

    @pytest.fixture(scope="class")
    def languages(self) -> Generator[dict[str, Language], None, None]:
        """Yields all languages."""
        yield get_all_languages()

    def test_languages_are_objects(self, languages: dict[str, Language]) -> None:
        """Tests that the values of `languages` are `Language` objects."""
        for language in languages.values():
            assert isinstance(language, Language)

    def test_languages_contain_germanic_languages(
        self, languages: dict[str, Language]
    ) -> None:
        """Tests that `languages` contains the Germanic languages."""
        assert "sv" in languages
        assert "da" in languages
        assert "no" in languages
        assert "nb" in languages
        assert "nn" in languages
        assert "is" in languages
        assert "fo" in languages
        assert "de" in languages
        assert "nl" in languages
        assert "en" in languages

    def test_languages_is_dict(self, languages: dict[str, Language]) -> None:
        """Tests that `languages` is a dictionary."""
        assert isinstance(languages, dict)


@pytest.mark.parametrize(
    argnames=["input_language_codes", "expected_language_codes"],
    argvalues=[
        ("da", ["da"]),
        (["da"], ["da"]),
        (["da", "en"], ["da", "en"]),
        ("no", ["no", "nb", "nn"]),
        (["nb"], ["nb", "no"]),
        ("all", list(get_all_languages().keys())),
    ],
    ids=[
        "single language",
        "single language as list",
        "multiple languages",
        "no -> no + nb + nn",
        "nb -> nb + no",
        "all -> all languages",
    ],
)
def test_get_correct_language_codes(
    input_language_codes: str | list[str], expected_language_codes: list[str]
) -> None:
    """Test that the correct language codes are returned."""
    languages = get_correct_language_codes(language_codes=input_language_codes)
    assert set(languages) == set(expected_language_codes)


def test_is_language_code_tells_names_apart() -> None:
    """`zho` is a missing language; `default` was never a language at all."""
    assert is_language_code("zho")
    assert is_language_code("dan")
    assert is_language_code("DA")
    assert not is_language_code("default")
    assert not is_language_code("train")


def test_iso_639_3_aliases_resolve() -> None:
    """Every alias points at a language EuroEval actually has."""
    registry = get_all_languages()
    unresolvable = sorted(
        alias
        for alias, code in ISO_639_3_ALIASES.items()
        if code not in registry or get_language(alias) is not registry[code]
    )
    assert unresolvable == []
