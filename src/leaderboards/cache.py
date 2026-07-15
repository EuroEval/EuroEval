"""Caching of model metadata."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from tqdm.auto import tqdm

from euroeval.string_utils import split_model_id

from .jsonl_io import load_records_from_jsonl_files
from .records import plain_model_id

logger = logging.getLogger(__name__)


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

        jsonl_files = sorted(results_dir.glob("*.jsonl"))
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
            model_id = split_model_id(model_id=plain_model_id(model_id)).model_id

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
