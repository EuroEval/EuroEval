"""Tests for the language detection metric."""

from euroeval.data_models import DatasetConfig
from euroeval.languages import BULGARIAN, DANISH, ENGLISH
from euroeval.tasks import TRANSLATION


class TestDanishNorwegianLanguageDetection:
    """Tests for Danish/Norwegian language detection edge cases."""

    def test_danish_translation_target(self) -> None:
        """Test that Danish translation respects target language."""
        config = DatasetConfig(
            name="wmt24pp-en-da",
            pretty_name="WMT24++-da",
            source="EuroEval/wmt24pp-en-da",
            task=TRANSLATION,
            languages=DANISH,
            source_language=ENGLISH,
            target_language=DANISH,
        )

        assert config._target_language == DANISH
        main_lang = config.main_language
        assert isinstance(main_lang, tuple)
        assert main_lang == (ENGLISH, DANISH)


class TestLanguageDetectorTranslationTarget:
    """Tests for language detection with translation tasks."""

    def test_fallback_to_old_behavior_without_explicit_target(self) -> None:
        """Test fallback to old behavior when explicit target is not provided."""
        config = DatasetConfig(
            name="test-translation",
            pretty_name="Test Translation",
            source="test",
            task=TRANSLATION,
            languages=[ENGLISH, BULGARIAN],
        )

        # Check that main_language still works
        main_lang = config.main_language
        assert isinstance(main_lang, tuple)
        assert main_lang[0] == ENGLISH
        assert main_lang[1] == BULGARIAN

        # Check that _target_language is None (fallback mode)
        assert config._target_language is None

    def test_uses_explicit_target_for_reverse_translation(self) -> None:
        """Test that language detection uses explicit target for reverse translation."""
        config = DatasetConfig(
            name="wmt24pp-bg-en",
            pretty_name="WMT24++-bg-en",
            source="EuroEval/wmt24pp-bg-en",
            task=TRANSLATION,
            languages=BULGARIAN,
            source_language=BULGARIAN,
            target_language=ENGLISH,
        )

        # Check that target is English for reverse translation
        main_lang = config.main_language
        assert isinstance(main_lang, tuple)
        assert main_lang[0] == BULGARIAN  # source
        assert main_lang[1] == ENGLISH  # target

        # Check that _target_language is set
        assert config._target_language == ENGLISH

    def test_uses_explicit_target_language(self) -> None:
        """Test that language detection uses explicit target language."""
        config = DatasetConfig(
            name="wmt24pp-en-bg",
            pretty_name="WMT24++-bg",
            source="EuroEval/wmt24pp-en-bg",
            task=TRANSLATION,
            languages=BULGARIAN,
            source_language=ENGLISH,
            target_language=BULGARIAN,
        )

        # Note: detector needs to download the model, so we just test the config
        # setup is correct

        # Check that main_language returns the correct tuple
        main_lang = config.main_language
        assert isinstance(main_lang, tuple)
        assert main_lang[0] == ENGLISH  # source
        assert main_lang[1] == BULGARIAN  # target

        # Check that _target_language is set for language detection to use
        assert config._target_language == BULGARIAN
