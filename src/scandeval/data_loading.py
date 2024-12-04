"""Functions related to the loading of the data."""

import sys

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub.utils import HfHubHTTPError
from numpy.random import Generator

from .data_models import BenchmarkConfig, DatasetConfig
from .exceptions import InvalidBenchmark
from .utils import unscramble


def load_data(
    rng: Generator, dataset_config: "DatasetConfig", benchmark_config: "BenchmarkConfig"
) -> list[DatasetDict]:
    """Load the raw bootstrapped datasets.

    Args:
        rng:
            The random number generator to use.
        dataset_config:
            The configuration for the dataset.
        benchmark_config:
            The configuration for the benchmark.

    Returns:
        A list of bootstrapped datasets, one for each iteration.
    """
    try:
        dataset = load_dataset(
            path=dataset_config.huggingface_id,
            cache_dir=benchmark_config.cache_dir,
            token=unscramble("HjccJFhIozVymqXDVqTUTXKvYhZMTbfIjMxG_"),
        )
    except HfHubHTTPError:
        raise InvalidBenchmark("The Hugging Face Hub seems to be down.")

    assert isinstance(dataset, DatasetDict)

    dataset = DatasetDict({key: dataset[key] for key in ["train", "val", "test"]})

    if not benchmark_config.evaluate_test_split:
        dataset["test"] = dataset["val"]

    # Remove empty examples from the datasets
    for text_feature in ["tokens", "text"]:
        if text_feature in dataset["train"].features:
            dataset = dataset.filter(lambda x: len(x[text_feature]) > 0)

    # If we are testing then truncate the test set
    if hasattr(sys, "_called_from_test"):
        dataset["test"] = dataset["test"].select(range(1))

    # Bootstrap the splits
    bootstrapped_splits: dict[str, list[Dataset]] = dict()
    for split in ["train", "val", "test"]:
        bootstrap_indices = rng.integers(
            0,
            len(dataset[split]),
            size=(benchmark_config.num_iterations, len(dataset[split])),
        )
        bootstrapped_splits[split] = [
            dataset[split].select(bootstrap_indices[idx])
            for idx in range(benchmark_config.num_iterations)
        ]

    datasets = [
        DatasetDict(
            {
                split: bootstrapped_splits[split][idx]
                for split in ["train", "val", "test"]
            }
        )
        for idx in range(benchmark_config.num_iterations)
    ]
    return datasets
