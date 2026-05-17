"""Cached wrappers around the Hugging Face Hub API."""

from huggingface_hub import HfApi

from .caching_utils import cache_arguments


@cache_arguments("dataset_id", "revision")
def _list_repo_files(hf_api: HfApi, dataset_id: str, revision: str) -> list[str]:
    """Cached wrapper around ``HfApi.list_repo_files`` for dataset repos.

    Args:
        hf_api:
            The Hugging Face API client.
        dataset_id:
            The dataset repo id.
        revision:
            The git revision to list files from.

    Returns:
        The list of file paths in the dataset repo at ``revision``.
    """
    return list(
        hf_api.list_repo_files(
            repo_id=dataset_id, repo_type="dataset", revision=revision
        )
    )


@cache_arguments("dataset_id")
def _repo_exists(hf_api: HfApi, dataset_id: str) -> bool:
    """Cached wrapper around ``HfApi.repo_exists`` for dataset repos.

    Args:
        hf_api:
            The Hugging Face API client.
        dataset_id:
            The dataset repo id.

    Returns:
        True if the dataset repo exists on the Hub, otherwise False.
    """
    return hf_api.repo_exists(repo_id=dataset_id, repo_type="dataset")
