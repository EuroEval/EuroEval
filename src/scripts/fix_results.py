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


def _validate_metadata(
    result_dict: dict[str, t.Any], model_id: str, cache: dict[str, dict[str, bool]]
) -> None:
    """Validate and fill in missing metadata fields.

    Args:
        result_dict:
            The result dictionary to update.
        model_id:
            The model identifier.
        cache:
            Cache of previously entered metadata values.
    """
    model_info = t.cast(dict[str, t.Any], result_dict["model_info"])
    for metadata_field in REQUIRED_METADATA_FIELDS:
        if metadata_field not in model_info["additional_details"]:
            if model_id in cache and metadata_field in cache[model_id]:
                value = cache[model_id][metadata_field]
            else:
                input_prompt = f"{metadata_field} for https://hf.co/{model_id} (y/n)? "
                value = input(input_prompt)
                while value not in ["y", "n"]:
                    value = input(input_prompt)
                value = value == "y"
                cache[model_id][metadata_field] = value
            model_info["additional_details"][metadata_field] = value


def _validate_record(record_file: Path) -> tuple[list[dict[str, t.Any]], bool]:
    """Validate a single result record.

    Args:
        record_file:
            Path to the JSON record file.

    Returns:
        Tuple of (model_results, should_remove).
    """
    try:
        record: dict[str, t.Any] = json.loads(record_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Error parsing {record_file}: {exc}")
        return [], True

    model_info = record.get("model_info", {})
    model_id = model_info.get("id") or model_info.get("name")
    if not model_id:
        print(f"Dropping {record_file}: missing model_info.id and model_info.name")
        return [], True

    model_results: list[dict[str, t.Any]] = [
        BenchmarkResult.from_dict(record).to_eee_dict()
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
        return [], True

    # Enforce non-empty name
    model_results = [
        result_dict
        for result_dict in model_results
        if t.cast(dict[str, t.Any], result_dict["model_info"])["name"].strip() != ""
    ]
    if not model_results:
        return [], True

    return model_results, False


def main() -> None:
    """Validate the results in the results/ directory."""
    files_to_remove: list[Path] = []
    cache: dict[str, dict[str, bool]] = defaultdict(dict)

    for record_file in tqdm(
        list(RESULTS_DIR.rglob("*/*.json")), desc="Validating results"
    ):
        if not record_file.is_file():
            continue

        model_results, should_remove = _validate_record(record_file=record_file)
        if should_remove:
            files_to_remove.append(record_file)
            continue

        model_info = model_results[0].get("model_info", {})
        model_id = model_info.get("id") or model_info.get("name")
        if model_id:
            for result_dict in model_results:
                _validate_metadata(
                    result_dict=result_dict, model_id=model_id, cache=cache
                )

        for result_dict in model_results:
            record_file.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")

    for path in files_to_remove:
        path.unlink()


if __name__ == "__main__":
    main()
