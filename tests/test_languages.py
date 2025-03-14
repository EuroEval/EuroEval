"""Unit tests for the `languages` module."""

from typing import Generator

import pytest

from euroeval.data_models import Language
from euroeval.languages import get_all_languages


class TestGetAllLanguages:
    """Unit tests for the `get_all_languages` function."""

    @pytest.fixture(scope="class")
    def languages(self) -> Generator[dict[str, Language], None, None]:
        """Yields all languages."""
        yield get_all_languages()

    def test_languages_is_dict(self, languages: dict[str, Language]) -> None:
        """Tests that `languages` is a dictionary."""
        assert isinstance(languages, dict)

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
