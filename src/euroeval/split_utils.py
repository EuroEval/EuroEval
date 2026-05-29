"""Utilities for detecting and mapping dataset splits."""

from pathlib import Path

from huggingface_hub import HfApi

from .caching_utils import cache_arguments


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


@cache_arguments("dataset_id")
def get_repo_split_names(hf_api: HfApi, dataset_id: str) -> list[str] | None:
    """Extract split names from a Hugging Face dataset repo.

    Args:
        hf_api:
            The Hugging Face API object.
        dataset_id:
            The ID of the dataset to get the split names for.

    Returns:
        A list of split names, or None if the split names are not available.
    """
    dataset_info = hf_api.dataset_info(repo_id=dataset_id)

    if (
        dataset_info.card_data is not None
        and hasattr(dataset_info.card_data, "dataset_info")
        and "splits" in dataset_info.card_data.dataset_info
    ):
        return [
            split["name"] for split in dataset_info.card_data.dataset_info["splits"]
        ]

    # If we don't have access to the split names directly, we look at the data files,
    # since they tend to be of the form "data/test-00000-of-00001.parquet"
    elif dataset_info.siblings is not None:
        parquet_file_names = [
            sibling.rfilename
            for sibling in dataset_info.siblings
            if sibling.rfilename.endswith(".parquet")
        ]
        split_names = [Path(fname).stem.split("-")[0] for fname in parquet_file_names]
        if split_names:
            return split_names

    return None


def get_repo_splits(
    hf_api: HfApi, dataset_id: str
) -> tuple[str | None, str | None, str | None]:
    """Return the (train, val, test) split names for a Hugging Face dataset repo.

    Args:
        hf_api:
            The Hugging Face API object.
        dataset_id:
            The ID of the dataset to get the split names for.

    Returns:
        A 3-tuple (train_split, val_split, test_split) where each element is either
            the name of the matching split or None if no such split exists.
    """
    splits = get_repo_split_names(hf_api=hf_api, dataset_id=dataset_id)
    if splits is None:
        return None, None, None
    return (
        find_split(splits=splits, keyword="train"),
        find_split(splits=splits, keyword="val"),
        find_split(splits=splits, keyword="test"),
    )
