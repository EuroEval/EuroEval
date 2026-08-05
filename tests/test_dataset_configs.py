"""Tests for the `dataset_configs` module."""

import os
from collections import defaultdict
from pathlib import Path
from typing import Generator

import pytest

from euroeval import dataset_configs as dc_module
from euroeval.data_models import DatasetConfig
from euroeval.dataset_configs import get_all_dataset_configs
from euroeval.languages import BULGARIAN, ENGLISH
from euroeval.tasks import TRANSLATION


class TestDatasetConfigLanguageNormalisation:
    """Tests for DatasetConfig language normalisation."""

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

    def test_single_language_object_is_normalised_to_list(self) -> None:
        """Test that a single Language object is normalised to a list."""
        config = DatasetConfig(
            name="test",
            pretty_name="Test",
            source="test",
            task=TRANSLATION,
            languages=BULGARIAN,
        )
        assert isinstance(config.languages, list)
        assert len(config.languages) == 1
        assert config.languages[0] == BULGARIAN


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


class TestTranslationDirectionMetadata:
    """Tests for explicit translation direction metadata."""

    def test_backward_compatibility_without_explicit_direction(self) -> None:
        """Test backward compatibility when explicit direction is not provided."""
        config = DatasetConfig(
            name="test",
            pretty_name="Test",
            source="test",
            task=TRANSLATION,
            languages=[ENGLISH, BULGARIAN],
        )
        main_lang = config.main_language
        assert isinstance(main_lang, tuple)
        assert main_lang[0] == ENGLISH
        assert main_lang[1] == BULGARIAN

    def test_main_language_returns_reverse_tuple(self) -> None:
        """Test that main_language returns correct tuple for reverse translation."""
        config = DatasetConfig(
            name="wmt24pp-bg-en",
            pretty_name="WMT24++-bg-en",
            source="EuroEval/wmt24pp-bg-en",
            task=TRANSLATION,
            languages=BULGARIAN,
            source_language=BULGARIAN,
            target_language=ENGLISH,
        )
        main_lang = config.main_language
        assert isinstance(main_lang, tuple)
        assert len(main_lang) == 2
        assert main_lang[0] == BULGARIAN
        assert main_lang[1] == ENGLISH

    def test_main_language_returns_tuple_with_explicit_direction(self) -> None:
        """Test that main_language returns correct tuple with explicit source/target."""
        config = DatasetConfig(
            name="wmt24pp-en-bg",
            pretty_name="WMT24++-bg",
            source="EuroEval/wmt24pp-en-bg",
            task=TRANSLATION,
            languages=BULGARIAN,
            source_language=ENGLISH,
            target_language=BULGARIAN,
        )
        main_lang = config.main_language
        assert isinstance(main_lang, tuple)
        assert len(main_lang) == 2
        assert main_lang[0] == ENGLISH
        assert main_lang[1] == BULGARIAN


def test_no_duplicate_dataset_config_variable_names() -> None:
    """Test that there are no duplicate variable names for dataset configs."""
    # Create a mapping from language name to list of variable names for the dataset
    # configs of that language
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

    # Count the number of occurences of each dataset config variable name
    dataset_variable_name_counts: dict[str, int] = defaultdict(int)
    for var_names in language_to_dataset_vars.values():
        for var_name in var_names:
            dataset_variable_name_counts[var_name] += 1

    # Raise an error if any variable name occurs more than once
    duplicate_variable_names = [
        name for name, count in dataset_variable_name_counts.items() if count > 1
    ]
    assert not duplicate_variable_names, (
        f"Duplicate dataset config variable names found: {duplicate_variable_names}. "
        "Please ensure that each dataset config variable has a unique name."
    )


def test_translation_not_included_in_english_leaderboard() -> None:
    """Test that WMT24++ configs don't accidentally pollute English selection."""
    # Check that there's no WMT24PP config with English as the only language
    all_configs = [
        cfg for cfg in vars(dc_module).values() if isinstance(cfg, DatasetConfig)
    ]
    wmt24pp_en_configs = [
        cfg
        for cfg in all_configs
        if cfg.name.startswith("wmt24pp-en-") and cfg.task == TRANSLATION
    ]
    # All en-X configs should have non-English language in their languages field
    for config in wmt24pp_en_configs:
        assert ENGLISH not in config.languages
        assert len(config.languages) == 1


def test_wmt24pp_configs_for_both_directions() -> None:
    """Test that WMT24++ configs exist for both directions."""
    # Check that English->Bulgarian exists
    assert hasattr(dc_module, "WMT24PP_EN_BG_CONFIG")
    en_bg_config = dc_module.WMT24PP_EN_BG_CONFIG
    assert en_bg_config.source_language == ENGLISH
    assert en_bg_config.target_language == BULGARIAN
    assert en_bg_config.languages == [BULGARIAN]

    # Check that Bulgarian->English exists
    assert hasattr(dc_module, "WMT24PP_BG_EN_CONFIG")
    bg_en_config = dc_module.WMT24PP_BG_EN_CONFIG
    assert bg_en_config.source_language == BULGARIAN
    assert bg_en_config.target_language == ENGLISH
    assert bg_en_config.languages == [BULGARIAN]

    # Check both are official
    assert en_bg_config.unofficial is False
    assert bg_en_config.unofficial is False


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
