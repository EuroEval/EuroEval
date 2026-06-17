"""Caching of model metadata."""

from __future__ import annotations

import json
import logging
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from tqdm.auto import tqdm

from euroeval.string_utils import split_model_id

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
        anchor_tag:
            A mapping from model IDs to their anchor tag.
        open:
            A mapping from model IDs to whether they are open (open-weight) or
            closed.
        trained_from_scratch:
            A mapping from model IDs to whether they were trained from scratch.
    """

    generative_type: dict[str, str | None] = field(default_factory=dict)
    merge: dict[str, bool] = field(default_factory=dict)
    commercially_licensed: dict[str, bool] = field(default_factory=dict)
    anchor_tag: dict[str, str] = field(default_factory=dict)
    open: dict[str, bool] = field(default_factory=dict)
    trained_from_scratch: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_processed_records(
        cls,
        compressed_results_path: Path | None = None,
        results_dir: Path | None = None,
    ) -> "Cache":
        """Create a cache from processed records.

        Args:
            compressed_results_path:
                The path to the compressed results file (results.tar.gz).
            results_dir:
                The path to the directory containing per-model JSONL files
                with metadata. If provided, takes precedence over
                compressed_results_path.

        Returns:
            A Cache instance populated with model metadata.

        Raises:
            FileNotFoundError:
                If neither the results file nor directory is found.
            ValueError:
                If the results file contains invalid JSON.
        """
        # Prefer results_dir if provided
        if results_dir is not None and results_dir.exists():
            return cls.from_results_dir(results_dir)

        if compressed_results_path is None or not compressed_results_path.exists():
            raise FileNotFoundError(
                f"Results file {compressed_results_path} not found."
            )

        with tarfile.open(compressed_results_path, "r:gz") as tar:
            results_file = tar.extractfile(member="results/results.jsonl")
            if results_file is None:
                logger.warning(
                    "Results file does not exist in tar.gz. Using an empty cache."
                )
                return cls()
            result_lines = results_file.read().decode(encoding="utf-8").splitlines()

        records: list[dict[str, object]] = []
        for line_idx, line in enumerate(result_lines):
            if not line.strip():
                continue
            for sub_line in line.replace("}{", "}\n{").split("\n"):
                if not sub_line.strip():
                    continue
                try:
                    records.append(json.loads(sub_line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_idx:,}: {sub_line}."
                    ) from e

        return cls._from_records(records=records, desc="Building caches")

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
            ValueError:
                If any JSONL file contains invalid JSON.
            FileNotFoundError:
                If the results directory does not exist.
        """
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory {results_dir} not found.")

        records: list[dict[str, object]] = []
        jsonl_files = sorted(results_dir.glob("*.jsonl"))
        for jsonl_file in jsonl_files:
            content = jsonl_file.read_text(encoding="utf-8")
            for line_idx, line in enumerate(content.splitlines()):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in {jsonl_file.name} line {line_idx:,}: {line}."
                    ) from e

        return cls._from_records(
            records=records, desc="Building caches from results dir"
        )

    @classmethod
    def _from_records(cls, records: list[dict[str, object]], desc: str) -> "Cache":
        """Populate a cache from parsed result records.

        Supports both the EEE format (metadata under ``model_info``) and the
        old EuroEval format (metadata at the top level).

        Args:
            records:
                Parsed result records in either EEE or old EuroEval format.
            desc:
                Progress-bar description.

        Returns:
            A Cache instance populated with model metadata.
        """
        cache = cls()
        for record in tqdm(records, desc=desc):
            if "model_info" in record and "name" in record["model_info"]:
                model_name = record["model_info"]["name"]
            else:
                model_name = record["model"]

            model_id: str = model_name
            if (match := re.search(r">(.+?)<", model_name)) is not None:
                model_id = match.group(1)
            model_id = split_model_id(model_id=model_id).model_id

            if "model_info" in record and "additional_details" in record["model_info"]:
                additional = record["model_info"]["additional_details"]
                if "generative_type" in additional:
                    cache.generative_type[model_id] = additional["generative_type"]
                if "merge" in additional:
                    cache.merge[model_id] = additional["merge"] == "true"
            else:
                if "generative_type" in record:
                    cache.generative_type[model_id] = record["generative_type"]
                if "merge" in record:
                    cache.merge[model_id] = record["merge"]

            if "commercially_licensed" in record:
                cache.commercially_licensed[model_id] = record["commercially_licensed"]
            if "open" in record:
                value = record["open"]
                if isinstance(value, str):
                    value = value in {"open-source", "open-weight"}
                cache.open[model_id] = value
            if "trained_from_scratch" in record:
                cache.trained_from_scratch[model_id] = record["trained_from_scratch"]
            if model_name.startswith("<a href="):
                inner_model_id_match = re.search(r">(.+?)<", model_name)
                if inner_model_id_match:
                    inner_model_id = inner_model_id_match.group(1)
                    inner_model_id = re.sub(r" *\(.*?\)", "", inner_model_id)
                    cache.anchor_tag[inner_model_id] = model_name
        return cache
