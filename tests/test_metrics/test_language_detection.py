"""Tests for the language detection metric."""

from unittest.mock import patch

from lingua import Language as LinguaLanguage

from euroeval.data_models import DatasetConfig
from euroeval.languages import BULGARIAN, DANISH, ENGLISH, NORWEGIAN_BOKMÅL
from euroeval.metrics.language_detection import LanguageDetector
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

    def test_explicit_target_preserves_danish_norwegian_tolerance(self) -> None:
        """Test that explicit target Danish includes Norwegian variants."""
        # Test with Danish as explicit target
        config_da = DatasetConfig(
            name="wmt24pp-en-da",
            pretty_name="WMT24++-da",
            source="EuroEval/wmt24pp-en-da",
            task=TRANSLATION,
            languages=DANISH,
            source_language=ENGLISH,
            target_language=DANISH,
        )

        # Test with Norwegian Bokmål as explicit target
        config_nb = DatasetConfig(
            name="wmt24pp-en-nb",
            pretty_name="WMT24++-nb",
            source="EuroEval/wmt24pp-en-nb",
            task=TRANSLATION,
            languages=NORWEGIAN_BOKMÅL,
            source_language=ENGLISH,
            target_language=NORWEGIAN_BOKMÅL,
        )

        captured_languages_da: list[LinguaLanguage] = []
        captured_languages_nb: list[LinguaLanguage] = []

        def capture_detect_language_da(
            self: "LanguageDetector",
            predictions: tuple[str],
            detector_languages: tuple[LinguaLanguage],
        ) -> list[float]:
            captured_languages_da.extend(detector_languages)
            return [1.0] * len(predictions)

        def capture_detect_language_nb(
            self: "LanguageDetector",
            predictions: tuple[str],
            detector_languages: tuple[LinguaLanguage],
        ) -> list[float]:
            captured_languages_nb.extend(detector_languages)
            return [1.0] * len(predictions)

        detector = LanguageDetector()
        detector.model = MockLanguageDetector()  # ty: ignore[invalid-assignment]

        # Test Danish config
        with patch.object(
            LanguageDetector, "_detect_language", capture_detect_language_da
        ):
            detector(["test"], config_da)

        # Test Norwegian config
        with patch.object(
            LanguageDetector, "_detect_language", capture_detect_language_nb
        ):
            detector(["test"], config_nb)

        # Both should include all three: Danish, Norwegian Bokmål, Norwegian Nynorsk
        expected_codes = {"DA", "NB", "NN"}

        actual_codes_da = {
            (
                lang.iso_code_639_1.name
                if lang.iso_code_639_1
                else lang.iso_code_639_3.name
            )
            for lang in captured_languages_da
        }
        actual_codes_nb = {
            (
                lang.iso_code_639_1.name
                if lang.iso_code_639_1
                else lang.iso_code_639_3.name
            )
            for lang in captured_languages_nb
        }

        assert expected_codes.issubset(actual_codes_da), (
            f"Danish config: expected {expected_codes}, got {actual_codes_da}"
        )
        assert expected_codes.issubset(actual_codes_nb), (
            f"Norwegian config: expected {expected_codes}, got {actual_codes_nb}"
        )

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

        # Check that target_language is None (fallback mode)
        assert config.target_language is None

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

        # Check that target_language is set
        assert config.target_language == ENGLISH

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

        # Check that main_language returns the correct tuple
        main_lang = config.main_language
        assert isinstance(main_lang, tuple)
        assert main_lang[0] == ENGLISH  # source
        assert main_lang[1] == BULGARIAN  # target

        # Check that target_language is set for language detection to use
        assert config.target_language == BULGARIAN

    def test_wmt24pp_en_da_danish_norwegian_tolerance(self) -> None:
        """Test that wmt24pp-en-da includes Danish and Norwegian variants."""
        config = DatasetConfig(
            name="wmt24pp-en-da",
            pretty_name="WMT24++-da",
            source="EuroEval/wmt24pp-en-da",
            task=TRANSLATION,
            languages=DANISH,
            source_language=ENGLISH,
            target_language=DANISH,
        )

        # Capture the detector_languages passed to _detect_language
        captured_languages: list[LinguaLanguage] = []

        def mock_detect_language(
            self: "LanguageDetector",
            predictions: tuple[str],
            detector_languages: tuple[LinguaLanguage],
        ) -> list[float]:
            captured_languages.extend(detector_languages)
            # Return mock scores
            return [1.0] * len(predictions)

        detector = LanguageDetector()
        # Mock the model to avoid downloading Lingua
        detector.model = MockLanguageDetector()  # ty: ignore[invalid-assignment]

        with patch.object(LanguageDetector, "_detect_language", mock_detect_language):
            detector(["test prediction"], config)

        # Should have Danish and Norwegian codes (uppercase in Lingua)
        expected_codes = {"DA", "NB", "NN"}
        actual_codes = {
            (
                lang.iso_code_639_1.name
                if lang.iso_code_639_1
                else lang.iso_code_639_3.name
            )
            for lang in captured_languages
        }
        assert expected_codes.issubset(actual_codes), (
            f"Expected Danish and Norwegian codes {expected_codes}, got {actual_codes}"
        )


class MockLanguageDetector:
    """Mock language detector for testing."""

    def compute_language_confidence_values_in_parallel(
        self, texts: list[str]
    ) -> list[list[object]]:
        """Return mock confidence values."""
        # Return a list of empty confidence lists for each text
        return [[] for _ in texts]
