"""Unit tests for the `dataset_configs` module."""

from typing import Generator

import pytest

from euroeval.data_models import DatasetConfig
from euroeval.dataset_configs import get_all_dataset_configs, get_dataset_config


class TestGetAllDatasetConfigs:
    """Unit tests for the `get_all_dataset_configs` function."""

    @pytest.fixture(scope="class")
    def dataset_configs(self) -> Generator[dict[str, DatasetConfig], None, None]:
        """Yields all dataset configurations."""
        yield get_all_dataset_configs()

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


class TestGetDatasetConfig:
    """Unit tests for the `get_dataset_config` function."""

    def test_get_angry_tweets_config(self) -> None:
        """Test that the angry tweets dataset config can be retrieved."""
        dataset_config = get_dataset_config("angry-tweets")
        assert dataset_config.name == "angry-tweets"

    def test_error_when_dataset_does_not_exist(self) -> None:
        """Test that an error is raised when the dataset does not exist."""
        with pytest.raises(ValueError):
            get_dataset_config("does-not-exist")
