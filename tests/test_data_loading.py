"""Tests for the `data_loading` module."""

from collections.abc import Generator

import pytest
from datasets import DatasetDict
from numpy.random import default_rng

from euroeval.data_loading import load_data
from euroeval.data_models import BenchmarkConfig
from euroeval.dataset_configs import get_dataset_config


class TestLoadData:
    """Tests for the `load_data` function."""

    @pytest.fixture(scope="class")
    def datasets(
        self, benchmark_config: BenchmarkConfig
    ) -> Generator[list[DatasetDict], None, None]:
        """A loaded dataset."""
        yield load_data(
            rng=default_rng(seed=4242),
            dataset_config=get_dataset_config("angry-tweets"),
            benchmark_config=benchmark_config,
        )

    def test_load_data_is_list_of_dataset_dicts(
        self, datasets: list[DatasetDict]
    ) -> None:
        """Test that the `load_data` function returns a list of `DatasetDict`."""
        assert isinstance(datasets, list)
        assert all(isinstance(d, DatasetDict) for d in datasets)

    def test_split_names_are_correct(self, datasets: list[DatasetDict]) -> None:
        """Test that the split names are correct."""
        assert all(set(d.keys()) == {"train", "val", "test"} for d in datasets)

    def test_number_of_iterations_is_correct(
        self, datasets: list[DatasetDict], benchmark_config: BenchmarkConfig
    ) -> None:
        """Test that the number of iterations is correct."""
        assert len(datasets) == benchmark_config.num_iterations

    def test_no_empty_examples(self, datasets: list[DatasetDict]) -> None:
        """Test that there are no empty examples in the datasets."""
        for dataset in datasets:
            for split in dataset.values():
                for feature in ["text", "tokens"]:
                    if feature in split.features:
                        assert all(len(x) > 0 for x in split[feature])
