"""Tests for the `dataset_configs` module."""

import os
from collections import defaultdict
from pathlib import Path
from typing import Generator

import pytest

from euroeval import dataset_configs as dc_module
from euroeval.data_models import DatasetConfig, TranslationDatasetConfig
from euroeval.dataset_configs import get_all_dataset_configs
from euroeval.languages import BULGARIAN, ENGLISH, FAROESE
from euroeval.tasks import TRANSLATION

WMT24PP_LANGUAGE_CODES = (
    "bg",
    "ca",
    "cs",
    "da",
    "de",
    "el",
    "et",
    "fi",
    "fr",
    "hr",
    "hu",
    "is",
    "it",
    "lt",
    "lv",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sr",
    "sv",
    "uk",
)


class TestDatasetConfigLanguageLists:
    """Tests for DatasetConfig language lists."""

    def test_sequence_of_languages_is_preserved(self) -> None:
        """Test that a sequence of languages is preserved as a list."""
        config = DatasetConfig(
            name="test",
            pretty_name="Test",
            source="test",
            task=TRANSLATION,
            languages=[ENGLISH, BULGARIAN],
        )
        assert isinstance(config.languages, list)
        assert len(config.languages) == 2
        assert config.languages[0] == ENGLISH
        assert config.languages[1] == BULGARIAN


class TestGetAllDatasetConfigs:
    """Tests for the `get_all_dataset_configs` function."""

    @pytest.fixture(scope="class")
    def dataset_configs(self) -> Generator[dict[str, DatasetConfig], None, None]:
        """Yields all dataset configurations."""
        yield get_all_dataset_configs(
            custom_datasets_file=Path("custom_datasets.py"),
            dataset_ids=[],
            api_key=os.getenv("HF_TOKEN"),
            cache_dir=Path(".euroeval_cache"),
            trust_remote_code=True,
            run_with_cli=True,
        )

    def test_dataset_configs_are_objects(
        self, dataset_configs: dict[str, DatasetConfig]
    ) -> None:
        """Test that the dataset configs are `DatasetConfig` objects."""
        for dataset_config in dataset_configs.values():
            assert isinstance(dataset_config, DatasetConfig)

    def test_dataset_configs_is_dict(
        self, dataset_configs: dict[str, DatasetConfig]
    ) -> None:
        """Test that the dataset configs are a dict."""
        assert isinstance(dataset_configs, dict)


class TestTranslationDatasetConfig:
    """Tests for the `TranslationDatasetConfig` subclass."""

    def test_is_a_dataset_config(self) -> None:
        """Test that a TranslationDatasetConfig is also a DatasetConfig."""
        config = TranslationDatasetConfig(
            name="wmt24pp-en-bg",
            pretty_name="WMT24++-en-bg",
            source="EuroEval/wmt24pp-en-bg",
            task=TRANSLATION,
            languages=[BULGARIAN],
            source_language=ENGLISH,
            target_language=BULGARIAN,
        )
        assert isinstance(config, DatasetConfig)
        assert config.languages == [BULGARIAN]

    def test_main_language_returns_forward_direction(self) -> None:
        """Test that main_language returns the explicit forward direction."""
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

    def test_main_language_returns_reverse_direction(self) -> None:
        """Test that main_language returns the explicit reverse direction."""
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


def test_flores_gap_language_configs_are_official() -> None:
    """FLORES+ fills the WMT24++ gaps, so those configs are official."""
    en_fo = dc_module.FLORES_EN_FO_CONFIG
    assert en_fo.main_language == (ENGLISH, FAROESE)
    assert en_fo.languages == [FAROESE]
    assert en_fo.unofficial is False

    fo_en = dc_module.FLORES_FO_EN_CONFIG
    assert fo_en.main_language == (FAROESE, ENGLISH)
    assert fo_en.unofficial is False


def test_flores_wmt24pp_language_configs_are_unofficial() -> None:
    """For languages WMT24++ already covers, FLORES+ is added as unofficial."""
    en_de = dc_module.FLORES_EN_DE_CONFIG
    assert en_de.main_language[0] == ENGLISH
    assert en_de.unofficial is True
    assert dc_module.FLORES_DE_EN_CONFIG.unofficial is True


def test_include_sr_uses_cyrillic_prompt() -> None:
    """INCLUDE-sr content is Cyrillic, so it overrides the Latin KNOW template."""
    config = dc_module.INCLUDE_SR_CONFIG
    assert _has_cyrillic(config.prompt_prefix)
    assert _has_cyrillic(config.instruction_prompt)


def _has_cyrillic(text: str) -> bool:
    """Return whether the text contains any Cyrillic characters."""
    return any("Ѐ" <= ch <= "ӿ" for ch in text)


def test_no_duplicate_dataset_config_variable_names() -> None:
    """Test that there are no duplicate variable names for dataset configs."""
    submodules = [
        value
        for value in dc_module.__dict__.values()
        if isinstance(value, type(dc_module))
    ]
    language_to_dataset_vars: dict[str, list[str]] = {
        submodule.__name__.split(".")[-1]: [
            var_name
            for var_name, var_value in submodule.__dict__.items()
            if isinstance(var_value, DatasetConfig)
        ]
        for submodule in submodules
    }

    dataset_variable_name_counts: dict[str, int] = defaultdict(int)
    for var_names in language_to_dataset_vars.values():
        for var_name in var_names:
            dataset_variable_name_counts[var_name] += 1

    duplicate_variable_names = [
        name for name, count in dataset_variable_name_counts.items() if count > 1
    ]
    assert not duplicate_variable_names, (
        f"Duplicate dataset config variable names found: {duplicate_variable_names}. "
        "Please ensure that each dataset config variable has a unique name."
    )


def test_serbian_translation_prompt_is_cyrillic() -> None:
    """The Serbian source translation prompt matches the Cyrillic source text."""
    config = dc_module.WMT24PP_SR_EN_CONFIG
    assert _has_cyrillic(config.prompt_prefix)
    assert _has_cyrillic(config.instruction_prompt)


def test_translation_not_included_in_english_leaderboard() -> None:
    """Test that WMT24++ configs don't accidentally pollute English selection."""
    all_configs = [
        cfg for cfg in vars(dc_module).values() if isinstance(cfg, DatasetConfig)
    ]
    wmt24pp_configs = [
        cfg
        for cfg in all_configs
        if cfg.name.startswith("wmt24pp-") and cfg.task == TRANSLATION
    ]
    for config in wmt24pp_configs:
        assert ENGLISH not in config.languages
        assert len(config.languages) == 1


@pytest.mark.parametrize("language_code", WMT24PP_LANGUAGE_CODES)
def test_wmt24pp_configs_cover_all_documented_languages(language_code: str) -> None:
    """Test that both WMT24++ directions exist for every documented language."""
    language_code = language_code.upper()
    forward_config = getattr(dc_module, f"WMT24PP_EN_{language_code}_CONFIG")
    reverse_config = getattr(dc_module, f"WMT24PP_{language_code}_EN_CONFIG")

    assert forward_config.unofficial is False
    assert reverse_config.unofficial is True


def test_wmt24pp_configs_for_both_directions() -> None:
    """Test that WMT24++ configs exist for both directions."""
    assert hasattr(dc_module, "WMT24PP_EN_BG_CONFIG")
    en_bg_config = dc_module.WMT24PP_EN_BG_CONFIG
    assert en_bg_config.main_language == (ENGLISH, BULGARIAN)
    assert en_bg_config.languages == [BULGARIAN]

    assert hasattr(dc_module, "WMT24PP_BG_EN_CONFIG")
    bg_en_config = dc_module.WMT24PP_BG_EN_CONFIG
    assert bg_en_config.main_language == (BULGARIAN, ENGLISH)
    assert bg_en_config.languages == [BULGARIAN]

    assert en_bg_config.unofficial is False
    assert bg_en_config.unofficial is True


def test_wmt24pp_configs_for_croatian() -> None:
    """Test WMT24++ configs for Croatian."""
    assert hasattr(dc_module, "WMT24PP_EN_HR_CONFIG")
    assert hasattr(dc_module, "WMT24PP_HR_EN_CONFIG")


def test_wmt24pp_configs_for_slovak() -> None:
    """Test WMT24++ configs for Slovak."""
    assert hasattr(dc_module, "WMT24PP_EN_SK_CONFIG")
    assert hasattr(dc_module, "WMT24PP_SK_EN_CONFIG")


def test_wmt24pp_configs_for_slovene() -> None:
    """Test WMT24++ configs for Slovene."""
    assert hasattr(dc_module, "WMT24PP_EN_SL_CONFIG")
    assert hasattr(dc_module, "WMT24PP_SL_EN_CONFIG")
