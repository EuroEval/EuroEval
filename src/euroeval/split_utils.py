"""Utilities for detecting and mapping dataset splits."""

import typing as t
from pathlib import Path

from huggingface_hub import HfApi

from .caching_utils import cache_arguments


def get_repo_splits(
    hf_api: HfApi, dataset_id: str, config_name: str | None = None
) -> tuple[str | None, str | None, str | None]:
    """Return the (train, val, test) split names for a Hugging Face dataset repo.

    Args:
        hf_api:
            The Hugging Face API object.
        dataset_id:
            The ID of the dataset to get the split names for.
        config_name (optional):
            The name of the configuration to get the split names for.

    Returns:
        A 3-tuple (train_split, val_split, test_split) where each element is either
            the name of the matching split or None if no such split exists.
    """
    splits = get_repo_split_names(
        hf_api=hf_api, dataset_id=dataset_id, config_name=config_name
    )
    if splits is None:
        return None, None, None
    return (
        find_split(splits=splits, keyword="train"),
        find_split(splits=splits, keyword="val"),
        find_split(splits=splits, keyword="test"),
    )


def find_split(splits: list[str], keyword: str) -> str | None:
    """Return the shortest split name containing `keyword`, or None.

    Args:
        splits:
            A list of split names.
        keyword:
            The keyword to search for.

    Returns:
        The shortest split name containing `keyword`, or None if no such split
            exists.
    """
    candidates = sorted([s for s in splits if keyword in s.lower()], key=len)
    return candidates[0] if candidates else None


def get_repo_split_names(
    hf_api: HfApi, dataset_id: str, config_name: str | None = None
) -> list[str] | None:
    """Extract split names from a Hugging Face dataset repo.

    Args:
        hf_api:
            The Hugging Face API object.
        dataset_id:
            The ID of the dataset to get the split names for.
        config_name (optional):
            The name of the configuration to get the split names for. Repositories
            with multiple configurations can have different splits in each of them,
            so the information is filtered by configuration when it is known.

    Returns:
        A list of split names, or None if the split names are not available.
    """
    card_info, parquet_file_names = _get_repo_split_info(
        hf_api=hf_api, dataset_id=dataset_id
    )

    if card_info is not None:
        # Cards of repositories with multiple configurations are keyed by
        # configuration name, while single-configuration cards are not
        if config_name is not None and config_name in card_info:
            card_info = card_info[config_name]
        if "splits" in card_info:
            return [
                split["name"]
                for split in card_info["splits"]  # ty: ignore[not-subscriptable]
            ]

    # If we don't have access to the split names directly, we look at the data files,
    # since they tend to be of the form "data/test-00000-of-00001.parquet", or
    # "<config>/test-00000-of-00001.parquet" for multi-configuration repositories
    if config_name is not None:
        config_file_names = [
            fname
            for fname in parquet_file_names
            if Path(fname).parent.name == config_name
        ]
        # Only trust the filtering when the repository actually stores its files per
        # configuration, as the split names would otherwise be lost entirely
        if config_file_names:
            parquet_file_names = config_file_names

    split_names = [Path(fname).stem.split("-")[0] for fname in parquet_file_names]
    # A multi-configuration repository has the same split names several times
    return list(dict.fromkeys(split_names)) if split_names else None


@cache_arguments("dataset_id")
def _get_repo_split_info(hf_api: HfApi, dataset_id: str) -> tuple[t.Any, list[str]]:
    """Look up the raw split information of a Hugging Face dataset repo.

    Args:
        hf_api:
            The Hugging Face API object.
        dataset_id:
            The ID of the dataset to look up.

    Returns:
        A pair with the `dataset_info` field of the dataset card, if any, and the
        names of the Parquet files in the repository.
    """
    dataset_info = hf_api.dataset_info(repo_id=dataset_id)
    card_info = (
        dataset_info.card_data.dataset_info
        if dataset_info.card_data is not None
        and hasattr(dataset_info.card_data, "dataset_info")
        else None
    )
    parquet_file_names = (
        [
            sibling.rfilename
            for sibling in dataset_info.siblings
            if sibling.rfilename.endswith(".parquet")
        ]
        if dataset_info.siblings is not None
        else []
    )
    return card_info, parquet_file_names
