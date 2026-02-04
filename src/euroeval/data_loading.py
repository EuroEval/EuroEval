"""Functions related to the loading of the data."""

import collections.abc as c
import logging
import sys
import time
import typing as t

import requests
from datasets import DatasetDict, load_dataset
from datasets.exceptions import DatasetsError
from huggingface_hub.errors import HfHubHTTPError
from numpy.random import Generator

from .constants import SUPPORTED_FILE_FORMATS_FOR_LOCAL_DATASETS
from .exceptions import HuggingFaceHubDown, InvalidBenchmark
from .logging_utils import log, no_terminal_output
from .tasks import EUROPEAN_VALUES
from .utils import unscramble

if t.TYPE_CHECKING:
    from datasets import Dataset

    from .data_models import BenchmarkConfig, DatasetConfig


def load_data(
    rng: Generator, dataset_config: "DatasetConfig", benchmark_config: "BenchmarkConfig"
) -> list["DatasetDict"]:
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

    Raises:
        InvalidBenchmark:
            If the dataset cannot be loaded.
        HuggingFaceHubDown:
            If the Hugging Face Hub is down.
    """
    dataset = load_raw_data(
        dataset_config=dataset_config, cache_dir=benchmark_config.cache_dir
    )

    if not benchmark_config.evaluate_test_split and "val" in dataset:
        dataset["test"] = dataset["val"]

    # Remove empty examples from the datasets
    for text_feature in ["tokens", "text"]:
        for split in dataset_config.splits:
            if text_feature in dataset[split].features:
                dataset = dataset.filter(lambda x: len(x[text_feature]) > 0)

    # If we are testing then truncate the test set, unless we need the full set for
    # evaluation
    if hasattr(sys, "_called_from_test") and dataset_config.task != EUROPEAN_VALUES:
        dataset["test"] = dataset["test"].select(range(1))  # type: ignore[unsupported-operation]

    # Bootstrap the splits, if applicable
    if dataset_config.bootstrap_samples:
        bootstrapped_splits: dict[str, c.Sequence["Dataset"]] = dict()
        for split in dataset_config.splits:
            bootstrap_indices = rng.integers(
                0,
                len(dataset[split]),
                size=(benchmark_config.num_iterations, len(dataset[split])),
            )
            bootstrapped_splits[split] = [  # type: ignore[unsupported-operation]
                dataset[split].select(bootstrap_indices[idx])
                for idx in range(benchmark_config.num_iterations)
            ]
        datasets = [
            DatasetDict(  # type: ignore[no-matching-overload]
                {
                    split: bootstrapped_splits[split][idx]
                    for split in dataset_config.splits
                }
            )
            for idx in range(benchmark_config.num_iterations)
        ]
    else:
        datasets = [dataset] * benchmark_config.num_iterations

    return datasets


def load_raw_data(dataset_config: "DatasetConfig", cache_dir: str) -> "DatasetDict":
    """Load the raw dataset.

    Args:
        dataset_config:
            The configuration for the dataset.
        cache_dir:
            The directory to cache the dataset.

    Returns:
        The dataset.
    """
    # Case where the dataset source is a Hugging Face ID
    if isinstance(dataset_config.source, str):
        num_attempts = 5
        for _ in range(num_attempts):
            try:
                with no_terminal_output():
                    dataset = load_dataset(
                        path=dataset_config.source.split("::")[0],
                        name=(
                            dataset_config.source.split("::")[1]
                            if "::" in dataset_config.source
                            else None
                        ),
                        cache_dir=cache_dir,
                        token=unscramble("XbjeOLhwebEaSaDUMqqaPaPIhgOcyOfDpGnX_"),
                    )
                break
            except (
                FileNotFoundError,
                ConnectionError,
                DatasetsError,
                requests.ConnectionError,
                requests.ReadTimeout,
            ) as e:
                log(
                    f"Failed to load dataset {dataset_config.source!r}, due to "
                    f"the following error: {e}. Retrying...",
                    level=logging.DEBUG,
                )
                time.sleep(1)
                continue
            except HfHubHTTPError:
                raise HuggingFaceHubDown()
        else:
            raise InvalidBenchmark(
                f"Failed to load dataset {dataset_config.source!r} after "
                f"{num_attempts} attempts. Run with verbose mode to see the individual "
                "errors."
            )

    # Case where the dataset source is a dictionary with keys "train", "val" and "test",
    # with the values pointing to local CSV files
    else:
        data_files = {
            split: dataset_config.source[split]
            for split in dataset_config.splits
            if split in dataset_config.source
        }

        # Get the file extension and ensure that all files have the same extension
        file_extensions = {
            split: dataset_config.source[split].split(".")[-1]
            for split in dataset_config.splits
            if split in dataset_config.source
        }
        if len(set(file_extensions.values())) != 1:
            raise InvalidBenchmark(
                "All data files in a custom dataset must have the same file extension. "
                f"Got the extensions {', '.join(file_extensions.values())} for the "
                f"dataset {dataset_config.name!r}."
            )
        file_extension = list(file_extensions.values())[0]

        # Check that the file extension is supported
        if file_extension not in SUPPORTED_FILE_FORMATS_FOR_LOCAL_DATASETS:
            raise InvalidBenchmark(
                "Unsupported file extension for custom dataset. Supported file "
                "extensions are "
                f"{', '.join(SUPPORTED_FILE_FORMATS_FOR_LOCAL_DATASETS)}, but got "
                f"{file_extension!r}."
            )

        # Load the dataset
        with no_terminal_output():
            dataset = load_dataset(
                path=file_extension, data_files=data_files, cache_dir=cache_dir
            )

    assert isinstance(dataset, DatasetDict)  # type: ignore[used-before-def]

    if dataset_config.preprocess_fn is not None:
        dataset = dataset_config.preprocess_fn(dataset)
        if not isinstance(dataset, DatasetDict):
            raise InvalidBenchmark(
                "Dataset preprocessing must return a DatasetDict, but got "
                f"{type(dataset)}."
            )

    dataset = _split_dataset_if_needed(dataset=dataset, dataset_config=dataset_config)

    missing_keys = [key for key in dataset_config.splits if key not in dataset]
    if missing_keys:
        raise InvalidBenchmark(
            "The dataset is missing the following required splits: "
            f"{', '.join(missing_keys)}"
        )
    return DatasetDict({key: dataset[key] for key in dataset_config.splits})  # type: ignore[no-matching-overload]


def _split_dataset_if_needed(
    dataset: DatasetDict, dataset_config: "DatasetConfig"
) -> DatasetDict:
    """Split a dataset from its train split if required splits are missing."""
    missing_keys = [key for key in dataset_config.splits if key not in dataset]
    if not missing_keys:
        return dataset

    if dataset_config.split_sizes is None:
        return dataset

    if "train" not in dataset:
        raise InvalidBenchmark(
            "Cannot create missing splits without a 'train' split."
        )

    split_sizes = dataset_config.split_sizes
    if any(split not in split_sizes for split in dataset_config.splits):
        raise InvalidBenchmark(
            "Split sizes must be provided for all requested splits."
        )

    sizes: list[int | None] = [split_sizes[split] for split in dataset_config.splits]
    none_count = sum(1 for size in sizes if size is None)
    if none_count > 1:
        raise InvalidBenchmark(
            "At most one split size can be None to absorb the remainder."
        )

    base = dataset["train"].shuffle(seed=dataset_config.split_seed or 42)
    total_specified = sum(size for size in sizes if size is not None)
    remainder = len(base) - total_specified
    if remainder < 0:
        raise InvalidBenchmark(
            "Split sizes exceed the available number of samples."
        )

    if none_count == 1:
        none_index = sizes.index(None)
        sizes[none_index] = remainder
    elif total_specified != len(base):
        raise InvalidBenchmark(
            "Split sizes must sum to the dataset size when no remainder is specified."
        )

    start = 0
    split_datasets: dict[str, c.Sequence] = dict()
    for split_name, size in zip(dataset_config.splits, sizes):
        assert size is not None
        if size < 0:
            raise InvalidBenchmark("Split sizes must be non-negative.")
        end = start + size
        split_datasets[split_name] = base.select(range(start, end))
        start = end

    return DatasetDict(split_datasets)  # type: ignore[no-matching-overload]
