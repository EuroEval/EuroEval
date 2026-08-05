"""Tests for the language detection metric."""

from unittest.mock import patch

from lingua import Language as LinguaLanguage

from euroeval.data_models import DatasetConfig, TranslationDatasetConfig
from euroeval.languages import BULGARIAN, DANISH, ENGLISH
from euroeval.metrics.language_detection import LanguageDetector
from euroeval.tasks import TRANSLATION


class TestLanguageDetectorTranslationTarget:
    """Tests for language detection with translation tasks."""

    def test_danish_target_preserves_norwegian_tolerance(self) -> None:
        """Test Danish WMT24++ targets include Norwegian variants."""
        config = TranslationDatasetConfig(
            name="wmt24pp-en-da",
            pretty_name="WMT24++-en-da",
            source="EuroEval/wmt24pp-en-da",
            task=TRANSLATION,
            languages=[DANISH],
            source_language=ENGLISH,
            target_language=DANISH,
        )
        assert config.main_language == (ENGLISH, DANISH)
        assert {"DA", "NB", "NN"}.issubset(_detector_codes(dataset_config=config))

    def test_forward_translation_uses_non_english_target(self) -> None:
        """Test forward WMT24++ configs penalise against the target language."""
        config = TranslationDatasetConfig(
            name="wmt24pp-en-bg",
            pretty_name="WMT24++-en-bg",
            source="EuroEval/wmt24pp-en-bg",
            task=TRANSLATION,
            languages=[BULGARIAN],
            source_language=ENGLISH,
            target_language=BULGARIAN,
        )
        assert config.main_language == (ENGLISH, BULGARIAN)
        assert _detector_codes(dataset_config=config) == {"BG"}

    def test_reverse_translation_uses_english_target(self) -> None:
        """Test reverse WMT24++ configs penalise against English."""
        config = TranslationDatasetConfig(
            name="wmt24pp-bg-en",
            pretty_name="WMT24++-bg-en",
            source="EuroEval/wmt24pp-bg-en",
            task=TRANSLATION,
            languages=[BULGARIAN],
            source_language=BULGARIAN,
            target_language=ENGLISH,
        )
        assert config.main_language == (BULGARIAN, ENGLISH)
        assert _detector_codes(dataset_config=config) == {"EN"}


def _detector_codes(dataset_config: DatasetConfig) -> set[str]:
    """Return the Lingua detector language codes for a dataset config."""
    captured_languages: list[LinguaLanguage] = []

    def mock_detect_language(
        self: "LanguageDetector",
        predictions: tuple[str],
        detector_languages: tuple[LinguaLanguage],
    ) -> list[float]:
        captured_languages.extend(detector_languages)
        return [1.0] * len(predictions)

    detector = LanguageDetector()
    detector.model = MockLanguageDetector()  # ty: ignore[invalid-assignment]

    with patch.object(LanguageDetector, "_detect_language", mock_detect_language):
        detector(predictions=["test prediction"], dataset_config=dataset_config)

    return {
        lang.iso_code_639_1.name if lang.iso_code_639_1 else lang.iso_code_639_3.name
        for lang in captured_languages
    }


class MockLanguageDetector:
    """Mock language detector for testing."""

    def compute_language_confidence_values_in_parallel(
        self, texts: list[str]
    ) -> list[list[object]]:
        """Return mock confidence values."""
        return [[] for _ in texts]
