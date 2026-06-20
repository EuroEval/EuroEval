"""Cached lookup of dataset split sizes from the Hugging Face Hub.

Split sizes are needed to turn the raw failure counts stored on each result
record into a proportion (failures are summed across bootstrap iterations, so
the denominator is ``num_iterations * split_size``). They are fetched lazily
from the Hub and cached to disk, since they effectively never change. If a
dataset is ever resized, delete its entry from the cache file to force a
refetch.
"""

from __future__ import annotations

import json
import logging

from datasets import load_dataset_builder

from .constants import DATASET_SPLIT_SIZES_CACHE

logger = logging.getLogger(__name__)


def _load_cache() -> dict[str, dict[str, int]]:
    """Load the split-size cache from disk.

    Returns:
        The cached mapping of source -> {split_name: num_examples}, or an empty
        dict if the cache file is missing or unreadable.
    """
    if not DATASET_SPLIT_SIZES_CACHE.exists():
        return {}
    try:
        with DATASET_SPLIT_SIZES_CACHE.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read split-size cache: {e}")
        return {}


_CACHE: dict[str, dict[str, int]] = _load_cache()


def get_split_sizes(source: str) -> dict[str, int] | None:
    """Get the split sizes for a dataset, fetching and caching on a miss.

    Args:
        source:
            The Hugging Face dataset id (e.g. ``"EuroEval/conll-nl-mini"``).
            A ``"repo::config"`` form is supported, matching how datasets are
            loaded in ``euroeval.data_loading``.

    Returns:
        A mapping of split name (e.g. ``"test"``) to the number of examples, or
        None if the sizes could not be fetched.
    """
    if source in _CACHE:
        return _CACHE[source]

    try:
        path, _, config = source.partition("::")
        builder = load_dataset_builder(path, name=config or None)
        sizes = {name: info.num_examples for name, info in builder.info.splits.items()}
    except Exception as e:  # noqa: BLE001 - the Hub can fail in many ways
        logger.warning(f"Could not fetch split sizes for {source!r}: {e}")
        return None

    _CACHE[source] = sizes
    _save_cache()
    return sizes


def _save_cache() -> None:
    """Persist the in-memory split-size cache to disk."""
    try:
        DATASET_SPLIT_SIZES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with DATASET_SPLIT_SIZES_CACHE.open("w") as f:
            json.dump(_CACHE, f, indent=2, sort_keys=True)
            f.write("\n")
    except OSError as e:
        logger.warning(f"Could not write split-size cache: {e}")
