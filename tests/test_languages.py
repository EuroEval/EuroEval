"""Tests for the `languages` module."""

from typing import Generator

import pytest

from euroeval.data_models import Language
from euroeval.languages import (
    CANTONESE,
    DANISH,
    PORTUGUESE,
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


def test_a_639_3_code_shared_by_regional_variants_is_not_guessed() -> None:
    """Portuguese answers to `por`, but two Chineses do and neither is chosen.

    A regional variant is not the language itself, so `zho` names no language until
    EuroEval registers an untagged Chinese.
    """
    assert get_language("por") is PORTUGUESE
    assert get_language("zho") is None


def test_legacy_keyword_language_construction() -> None:
    """The historical keyword constructor keeps code and name in their old roles."""
    language = Language(code="da", name="Danish")

    assert language.code == "da"
    assert language.name == "Danish"


def test_legacy_positional_language_construction() -> None:
    """The historical positional constructor keeps code before name."""
    language = Language("da", "Danish")

    assert language.code == "da"
    assert language.name == "Danish"


def test_a_language_carries_both_codes() -> None:
    """Danish is `da`, and `dan` is how a dataset configuration may name it."""
    assert DANISH.code == "da"
    assert DANISH.code_1 == "da"
    assert DANISH.code_3 == "dan"
    assert get_language("dan") is DANISH
    assert get_language("DAN") is DANISH


def test_a_language_without_a_639_1_code_is_keyed_by_its_639_3_code() -> None:
    """Cantonese has no 639-1 code, so its 639-3 code is its identity."""
    assert CANTONESE.code_1 is None
    assert CANTONESE.code == "yue" == CANTONESE.code_3
    assert "yue" in get_all_languages()


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


def test_iso_639_3_codes_resolve() -> None:
    """Every 639-3 code belonging to one language leads back to it.

    A code can belong to several registrations only when they are regional variants of
    the same language, in which case `get_language` declines to choose.
    """
    registry = get_all_languages()
    by_code_3: dict[str, list[Language]] = {}
    for language in registry.values():
        by_code_3.setdefault(language.code_3, []).append(language)

    unresolvable = sorted(
        code
        for code, languages in by_code_3.items()
        if len(languages) == 1
        and code not in registry
        and get_language(code) is not languages[0]
    )
    assert unresolvable == []

    shared = {
        code: languages for code, languages in by_code_3.items() if len(languages) > 1
    }
    assert sorted(shared) == ["por", "zho"]
    assert all(
        len({language.code.partition("-")[0] for language in languages}) == 1
        for languages in shared.values()
    ), "a shared 639-3 code must only be shared by regional variants"


def test_the_registry_is_keyed_by_the_preferred_code() -> None:
    """Only one code per language keys the registry.

    Dataset IDs, results and model filters are all keyed by these codes, so a language
    must not silently re-key itself to its 639-3 code.
    """
    registry = get_all_languages()
    assert registry["da"] is DANISH
    assert "dan" not in registry
    assert all(code == language.code for code, language in registry.items())
