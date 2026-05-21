"""Language detection module."""

import collections.abc as c
import logging
import typing as t

from lingua import IsoCode639_1, IsoCode639_3, LanguageDetectorBuilder
from lingua import Language as LinguaLanguage

from ..caching_utils import cache_arguments
from ..constants import MIN_LANG_CONFIDENCE_SCORE
from ..languages import (
    DANISH,
    NORWEGIAN_BOKMÅL,
    NORWEGIAN_NYNORSK,
    get_correct_language_codes,
)
from ..logging_utils import log_once

if t.TYPE_CHECKING:
    from lingua import LanguageDetector as LinguaLanguageDetector

    from ..data_models import DatasetConfig


class LanguageDetector:
    """Language detector class."""

    def __init__(self) -> None:
        """Initialize the language detector."""
        self.model: "LinguaLanguageDetector | None" = None

    def download(self) -> None:
        """Download and initialize the language detection model."""
        if self.model is not None:
            return

        self.model = (
            LanguageDetectorBuilder.from_all_spoken_languages()
            .with_preloaded_language_models()
            .build()
        )

    def __call__(
        self, predictions: c.Sequence, dataset_config: "DatasetConfig"
    ) -> list[float]:
        """Classify if the predictions are in the dataset languages.

        Args:
            predictions:
                The predictions to detect the language of.
            dataset_config:
                The dataset configuration.

        Returns:
            List of binary scores (1.0 or 0.0) for each prediction, where 1.0
            indicates the prediction is in the correct language(s) and 0.0 indicates
            it is not.
        """
        # Danish and Norwegian variants are mutually included since they are
        # difficult to distinguish, even for humans, and some sentences can be
        # identical across them.
        danish_and_norwegian = {DANISH, NORWEGIAN_BOKMÅL, NORWEGIAN_NYNORSK}
        dataset_languages = set(dataset_config.languages)

        # extend with danish_and_norwegian if any are present
        if danish_and_norwegian & dataset_languages:
            dataset_languages |= danish_and_norwegian

        # Exclude the source language for translation tasks
        # as it is tuple for translation tasks, otherwise a single language is returned
        main_lang = dataset_config.main_language
        if isinstance(main_lang, tuple):
            source_lang, _ = main_lang
            dataset_languages.discard(source_lang)

        target_language_codes = get_correct_language_codes(
            language_codes=[lang.code for lang in dataset_languages]
        )

        detector_languages: list[LinguaLanguage] = list()
        for lang in target_language_codes:
            lang = lang.split("-")[0]
            match len(lang):
                case 2:
                    try:
                        iso_code = IsoCode639_1.from_str(string=lang)
                        lingua_language = LinguaLanguage.from_iso_code_639_1(
                            iso_code=iso_code
                        )
                    except ValueError:
                        # We ignore 'no', since that's covered by 'nb' and 'nn'
                        if lang != "no":
                            log_once(
                                f"The ISO 639-1 code {lang!r} is not supported by "
                                "Lingua, skipping",
                                level=logging.WARNING,
                            )
                        continue
                case 3:
                    try:
                        iso_code = IsoCode639_3.from_str(string=lang)
                        lingua_language = LinguaLanguage.from_iso_code_639_3(
                            iso_code=iso_code
                        )
                    except ValueError:
                        log_once(
                            f"The ISO 639-3 code {lang!r} is not supported by Lingua, "
                            "skipping",
                            level=logging.WARNING,
                        )
                        continue
                case _:
                    log_once(
                        f"Language {lang!r} is neither an ISO 639-1 nor an ISO 639-3 "
                        "code. Skipping.",
                        level=logging.WARNING,
                    )
                    continue
            detector_languages.append(lingua_language)

        return self._detect_language(
            predictions=tuple(predictions), detector_languages=tuple(detector_languages)
        )

    @cache_arguments()
    def _detect_language(
        self, predictions: tuple[str], detector_languages: tuple[LinguaLanguage]
    ) -> list[float]:
        """Internal method that performs language detection with caching.

        Args:
            predictions:
                The predictions to detect the language of.
            detector_languages:
                The detector languages to use for filtering.

        Returns:
            List of binary scores (1.0 or 0.0) for each prediction, where 1.0
            indicates the prediction is in the correct language(s) and 0.0 indicates
            it is not. Score is 1.0 if the sum of confidence values for target
            languages exceeds MIN_LANG_CONFIDENCE_SCORE
        """
        if self.model is None:
            self.download()
        assert self.model is not None, (
            "Model is not initialized, please call download() first"
        )

        conf_values = self.model.compute_language_confidence_values_in_parallel(
            list(predictions)
        )
        scores = []
        for confidence in conf_values:
            # Sum the confidence values so we get a single value for datasets with
            # 'multiple' languages, like Danish and Norwegian
            lang_confidence = sum(
                conf.value for conf in confidence if conf.language in detector_languages
            )
            scores.append(1.0 if lang_confidence > MIN_LANG_CONFIDENCE_SCORE else 0.0)
        return scores


language_detector = LanguageDetector()
