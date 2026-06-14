"""Cached wrappers around the Hugging Face Hub API."""

import logging

from huggingface_hub import HfApi
from huggingface_hub.errors import OfflineModeIsEnabled

from .caching_utils import cache_arguments
from .logging_utils import log_once


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

    When ``HF_HUB_OFFLINE=1`` is set, ``HfApi.repo_exists`` raises rather than
    returning ``False``; we treat that as "not reachable, so not present" so
    callers can fall back to local configs instead of crashing.

    Args:
        hf_api:
            The Hugging Face API client.
        dataset_id:
            The dataset repo id.

    Returns:
        True if the dataset repo exists on the Hub, otherwise False.
    """
    try:
        return hf_api.repo_exists(repo_id=dataset_id, repo_type="dataset")
    except OfflineModeIsEnabled:
        log_once(
            f"HF_HUB_OFFLINE is set; assuming {dataset_id!r} is not on the Hub.",
            level=logging.WARNING,
        )
        return False
