"""Fix the results in the results/ directory.

This enforces EEE formatting, minimum EuroEval version, non-empty name, and ensures that
all the extra metadata fields are present.
"""

import json
import typing as t
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from euroeval.data_models import BenchmarkResult
from leaderboards.constants import (
    MINIMUM_VERSION,
    REQUIRED_METADATA_FIELDS,
    RESULTS_DIR,
)


def main() -> None:
    """Validate the results in the results/ directory."""
    files_to_remove: list[Path] = []
    for record_file in tqdm(
        list(RESULTS_DIR.rglob("*/*.json")), desc="Validating results"
    ):
        if not record_file.is_file():
            continue

        # Extract model_id from parent directory name (reverse sanitisation)
        model_dir_name = record_file.parent.name
        model_id = model_dir_name.replace("_", "/", 1)

        # Read single JSON record
        try:
            record: dict[str, t.Any] = json.loads(
                record_file.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"Error parsing {record_file}: {exc}")
            files_to_remove.append(record_file)
            continue

        model_results: list[dict[str, t.Any]] = [record]

        # Enforce EEE
        model_results = [
            BenchmarkResult.from_dict(result_dict).to_eee_dict()
            for result_dict in model_results
        ]

        # Enforce minimum EuroEval version
        model_results = [
            result_dict
            for result_dict in model_results
            if t.cast(dict[str, t.Any], result_dict["eval_library"])["version"].replace(
                r".dev0", ""
            )
            >= MINIMUM_VERSION
        ]
        if not model_results:
            files_to_remove.append(record_file)
            continue

        # Enforce non-empty name
        model_results = [
            result_dict
            for result_dict in model_results
            if t.cast(dict[str, t.Any], result_dict["model_info"])["name"].strip() != ""
        ]
        if not model_results:
            files_to_remove.append(record_file)
            continue

        # Validate metadata
        cache: dict[str, dict[str, bool]] = defaultdict(dict)
        for result_dict in model_results:
            for metadata_field in REQUIRED_METADATA_FIELDS:
                model_info = t.cast(dict[str, t.Any], result_dict["model_info"])
                if metadata_field not in model_info["additional_details"]:
                    if model_id in cache and metadata_field in cache[model_id]:
                        value = cache[model_id][metadata_field]
                    else:
                        input_prompt = (
                            f"{metadata_field} for https://hf.co/{model_id} (y/n)? "
                        )
                        value = input(input_prompt)
                        while value not in ["y", "n"]:
                            value = input(input_prompt)
                        value = value == "y"
                        cache[model_id][metadata_field] = value
                    model_info["additional_details"][metadata_field] = value

        # Write back as single JSON dict
        for result_dict in model_results:
            record_file.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")

    for path in files_to_remove:
        path.unlink()


if __name__ == "__main__":
    main()
