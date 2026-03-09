"""Utilities for detecting and mapping dataset splits."""

from huggingface_hub import HfApi


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


def get_repo_split_names(hf_api: HfApi, dataset_id: str) -> list[str]:
    """Extract split names from a Hugging Face dataset repo.

    Args:
        hf_api:
            The Hugging Face API object.
        dataset_id:
            The ID of the dataset to get the split names for.

    Returns:
        A list of split names.
    """
    return [
        split["name"]
        for split in hf_api.dataset_info(repo_id=dataset_id).card_data.dataset_info[
            "splits"
        ]
    ]


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
    return (
        find_split(splits=splits, keyword="train"),
        find_split(splits=splits, keyword="val"),
        find_split(splits=splits, keyword="test"),
    )
