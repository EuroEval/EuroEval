"""ModelCache class for caching model outputs."""

import hashlib
import json
import logging
import sys
import typing as t
from collections import defaultdict
from dataclasses import asdict

from tqdm.auto import tqdm

from .data_models import GenerativeModelOutput, SingleGenerativeModelOutput

if t.TYPE_CHECKING:
    from pathlib import Path

    from datasets import Dataset


logger = logging.getLogger("euroeval")


class ModelCache:
    """A cache for model outputs.

    Attributes:
        model_cache_dir:
            The directory to store the cache in.
        cache_path:
            The path to the cache file.
        cache:
            The model output cache.
        max_generated_tokens:
            The maximum number of tokens to generate for each example.
    """

    def __init__(
        self, model_cache_dir: "Path", cache_name: str, max_generated_tokens: int
    ) -> None:
        """Initialise the model output cache.

        Args:
            model_cache_dir:
                The directory to store the cache in.
            cache_name:
                The name of the cache file.
            max_generated_tokens:
                The maximum number of tokens to generate for each example.
        """
        self.model_cache_dir = model_cache_dir
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.model_cache_dir / cache_name.replace("/", "--")
        self.max_generated_tokens = max_generated_tokens

    def load(self) -> None:
        """Load the model output cache."""
        if not self.cache_path.exists():
            with self.cache_path.open("w") as f:
                json.dump(dict(), f)

        try:
            with self.cache_path.open() as f:
                json_cache = json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to load the cache from {self.cache_path}. The cache will be "
                f"re-initialised."
            )
            json_cache = dict()
            with self.cache_path.open("w") as f:
                json.dump(dict(), f)

        cache: dict[str, SingleGenerativeModelOutput] = dict()
        for key in json_cache:
            cache[key] = SingleGenerativeModelOutput(**json_cache[key])

        self.cache = cache

    def save(self) -> None:
        """Save the model output cache to disk."""
        dumpable_cache: dict[str, dict] = defaultdict(dict)
        for key, value in self.cache.items():
            dumpable_cache[key] = asdict(value)

        try:
            with self.cache_path.open("w") as f:
                json.dump(dumpable_cache, f)
        except KeyError:
            logger.warning(
                f"Failed to load the cache from {self.cache_path}. The cache will be "
                f"re-initialised."
            )
            self.cache = dict()
            with self.cache_path.open("w") as f:
                json.dump(dict(), f)

    def _hash_key(self, key: str | list[dict[str, str]]) -> str:
        """Hash the key to use as an index in the cache.

        Args:
            key:
                The key to hash.

        Returns:
            The hashed key.
        """
        return hashlib.md5(string=str(key).encode()).hexdigest()

    def __getitem__(
        self, key: str | list[dict[str, str]]
    ) -> SingleGenerativeModelOutput:
        """Get an item from the cache.

        Args:
            key:
                The key to use to index the cache.

        Returns:
            The model output.
        """
        hashed_key = self._hash_key(key=key)
        return self.cache[hashed_key]

    def __setitem__(
        self, key: str | list[dict[str, str]], value: SingleGenerativeModelOutput
    ) -> None:
        """Set an item in the cache.

        Args:
            key:
                The key to use to index the cache.
            value:
                The value to set in the cache.
        """
        hashed_key = self._hash_key(key=key)
        self.cache[hashed_key] = value

    def remove(self) -> None:
        """Remove the cache from memory and delete it from disk."""
        self.cache_path.unlink()
        del self.cache

    def __contains__(self, key: str | list[dict[str, str]]) -> bool:
        """Check if a key is in the cache.

        Args:
            key:
                The key to check.

        Returns:
            Whether the key is in the cache.
        """
        hashed_key = self._hash_key(key=key)
        return hashed_key in self.cache

    def add_to_cache(
        self, model_inputs: dict, model_output: GenerativeModelOutput
    ) -> None:
        """Add the model input/output to the cache.

        Args:
            model_inputs:
                The model inputs.
            model_output:
                The model output.
        """
        input_column = "messages" if "messages" in model_inputs else "text"
        model_inputs = model_inputs[input_column]

        # Store the generated sequences in the cache, one by one
        with tqdm(
            iterable=model_inputs,
            desc="Caching model outputs",
            leave=False,
            disable=hasattr(sys, "_called_from_test"),
        ) as pbar:
            for sample_idx, model_input in enumerate(pbar):
                # Extract the scores from the model output, to be cached. We only store
                # the indices of the top scores, to save space. Further, we only store
                # the scores if the generated sequence is shorter than the maximum
                # length
                if model_output.scores is not None and self.max_generated_tokens < 8:
                    assert model_output.scores is not None
                    scores = model_output.scores[sample_idx]
                else:
                    scores = None
                self[model_input] = SingleGenerativeModelOutput(
                    sequence=model_output.sequences[sample_idx], scores=scores
                )


def split_dataset_into_cached_and_non_cached(
    dataset: "Dataset", cache: ModelCache
) -> tuple["Dataset", "Dataset"]:
    """Split a dataset into a cached and non-cached part.

    Args:
        dataset:
            The dataset to split.
        cache:
            The model output cache.

    Returns:
        The cached and non-cached parts of the dataset.
    """
    # Get the sample indices of the non-cached examples, which are unique with respect
    # to the "text" column.
    input_column = "messages" if "messages" in dataset.column_names else "text"
    dataset_texts = dataset[input_column]
    unique_non_cached_ids = set()
    unique_texts = list()
    for idx, dataset_text in enumerate(dataset_texts):
        if dataset_text not in cache and dataset_text not in unique_texts:
            unique_non_cached_ids.add(idx)
            unique_texts.append(dataset_text)

    # The cached examples are the ones that are not in the non-cached examples. This
    # means that if the dataset has duplicates, only a single copy of the duplicate
    # will be put in the non-cached part, and the rest in the cached part.
    cached_ids = set(range(len(dataset))) - unique_non_cached_ids

    cached = dataset.select(cached_ids)
    non_cached = dataset.select(unique_non_cached_ids)
    return cached, non_cached


def load_cached_model_outputs(
    cached_dataset: "Dataset", cache: ModelCache
) -> GenerativeModelOutput:
    """Load the cached model outputs.

    Args:
        cached_dataset:
            The dataset containing the cached examples.
        cache:
            The model output cache.

    Returns:
        The model output containing the cached sequences.
    """
    input_column = "messages" if "messages" in cached_dataset.column_names else "text"
    cached_model_outputs: list[SingleGenerativeModelOutput] = [
        cache[prompt] for prompt in cached_dataset[input_column]
    ]

    cached_sequences = [model_output.sequence for model_output in cached_model_outputs]

    if cached_model_outputs[0].scores is None:
        return GenerativeModelOutput(sequences=cached_sequences)

    cached_scores = [model_output.scores or [] for model_output in cached_model_outputs]
    return GenerativeModelOutput(sequences=cached_sequences, scores=cached_scores)
