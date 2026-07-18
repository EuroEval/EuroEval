"""Caching of model metadata."""

from __future__ import annotations

import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from tqdm.auto import tqdm

from euroeval.string_utils import split_model_id

from .constants import UNKNOWN_RESULTS_FILENAME
from .jsonl_io import load_records_from_jsonl_files
from .records import plain_model_id

logger = logging.getLogger(__name__)


def _normalise_model_id_for_hf_matching(model_id: str) -> str:
    """Normalise a model ID for Hugging Face URL/repo matching.

    Strips HTML anchor, variant suffixes, AND parameter/revision suffixes.
    This is narrower than :func:`plain_model_id` which preserves #param and
    @revision for meaningful model differentiation.

    Args:
        model_id:
            The model ID to normalise.

    Returns:
        The base repo ID suitable for HF URL comparison (e.g. "org/repo").
    """
    # Strip anchor and variant suffix first
    model_id = plain_model_id(model_id)
    # Then strip #param and @revision for HF repo comparison
    return split_model_id(model_id).model_id


def _is_hf_url_for_model(model_url: str, model_id: str) -> bool:
    """Check if a model URL is a Hugging Face URL for the given model.

    Uses exact repo-path matching to avoid false positives from prefix
    matching (e.g., https://hf.co/org/repo2 should not match org/repo).

    Args:
        model_url:
            The model URL to check.
        model_id:
            The model ID (e.g., ``org/repo``). May contain anchors,
            variant suffixes, or #param/@revision suffixes.

    Returns:
        True if the URL is an HF Hub URL for the model, False otherwise.
    """
    model_id = _normalise_model_id_for_hf_matching(model_id)
    parsed = urllib.parse.urlparse(model_url)
    if parsed.netloc not in (
        "hf.co",
        "huggingface.co",
        "www.hf.co",
        "www.huggingface.co",
    ):
        return False
    # Path should be exactly /{model_id}
    path = parsed.path.rstrip("/")
    return path == f"/{model_id}"


@dataclass
class Cache:
    """A cache for model metadata.

    Attributes:
        generative_type:
            A mapping from model IDs to their generative type.
        merge:
            A mapping from model IDs to whether they are merges of other models.
        commercially_licensed:
            A mapping from model IDs to whether they are commercially licensed.
        open:
            A mapping from model IDs to whether they are open (open-weight) or
            closed.
        trained_from_scratch:
            A mapping from model IDs to whether they were trained from scratch.
        model_url:
            A mapping from model IDs to their model URL.
    """

    generative_type: dict[str, str | None] = field(default_factory=dict)
    merge: dict[str, bool] = field(default_factory=dict)
    commercially_licensed: dict[str, bool] = field(default_factory=dict)
    open: dict[str, bool] = field(default_factory=dict)
    trained_from_scratch: dict[str, bool] = field(default_factory=dict)
    model_url: dict[str, str | None] = field(default_factory=dict)

    @classmethod
    def from_results_dir(cls, results_dir: Path) -> "Cache":
        """Create a cache from records in the results directory.

        Args:
            results_dir:
                The path to the directory containing per-model JSONL files
                with metadata.

        Returns:
            A Cache instance populated with model metadata.

        Raises:
            FileNotFoundError:
                If the results directory does not exist.
        """
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory {results_dir} not found.")

        jsonl_files = sorted(
            path
            for path in results_dir.glob("*.jsonl")
            if path.name != UNKNOWN_RESULTS_FILENAME
        )
        records = load_records_from_jsonl_files(paths=jsonl_files)

        return cls._from_records(
            records=records, desc="Building caches from results dir"
        )

    @classmethod
    def _from_records(cls, records: list[dict[str, object]], desc: str) -> "Cache":
        """Populate a cache from parsed EEE result records.

        Metadata is read from ``model_info.additional_details``.

        Args:
            records:
                Parsed result records in EEE format.
            desc:
                Progress-bar description.

        Returns:
            A Cache instance populated with model metadata.
        """
        cache = cls()
        for record in tqdm(records, desc=desc):
            model_name = record["model_info"]["name"]

            model_id: str = model_name
            if (match := re.search(r">(.+?)<", model_name)) is not None:
                model_id = match.group(1)
            model_id = _normalise_model_id_for_hf_matching(model_id)

            additional = record["model_info"]["additional_details"]
            if "generative_type" in additional:
                cache.generative_type[model_id] = additional["generative_type"]
            if "merge" in additional:
                cache.merge[model_id] = additional["merge"] == "true"
            if "commercially_licensed" in additional:
                cache.commercially_licensed[model_id] = additional[
                    "commercially_licensed"
                ]
            if "open" in additional:
                cache.open[model_id] = additional["open"]
            if "trained_from_scratch" in additional:
                cache.trained_from_scratch[model_id] = additional[
                    "trained_from_scratch"
                ]
            if "model_url" in additional and additional["model_url"] is not None:
                cache.model_url[model_id] = additional["model_url"]

        return cache
