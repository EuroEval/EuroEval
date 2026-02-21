"""Tests for the `dataset_configs` module."""

import os
from collections import defaultdict
from pathlib import Path
from typing import Generator

import pytest

from euroeval import dataset_configs as dc_module
from euroeval.data_models import DatasetConfig
from euroeval.dataset_configs import get_all_dataset_configs


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

    def test_dataset_configs_is_dict(
        self, dataset_configs: dict[str, DatasetConfig]
    ) -> None:
        """Test that the dataset configs are a dict."""
        assert isinstance(dataset_configs, dict)

    def test_dataset_configs_are_objects(
        self, dataset_configs: dict[str, DatasetConfig]
    ) -> None:
        """Test that the dataset configs are `DatasetConfig` objects."""
        for dataset_config in dataset_configs.values():
            assert isinstance(dataset_config, DatasetConfig)


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
