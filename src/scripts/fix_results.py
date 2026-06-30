"""Fix the results in the results/ directory.

This enforces EEE formatting, minimum EuroEval version, non-empty name, and ensures that
all the extra metadata fields are present.
"""

import json
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from euroeval.data_models import BenchmarkResult
from leaderboards.constants import (
    MINIMUM_VERSION,
    REQUIRED_METADATA_FIELDS,
    RESULTS_DIR,
)
from leaderboards.utils import parse_jsonl_lines


def main() -> None:
    """Validate the results in the results/ directory."""
    files_to_remove: list[Path] = []
    for jsonl_path in tqdm(
        list(RESULTS_DIR.glob("*.jsonl")), desc="Validating results"
    ):
        model_results: list[dict] = []
        skip_model = False
        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
            model_results = parse_jsonl_lines(lines=lines, source=str(jsonl_path))
        except ValueError as exc:
            print(f"Error parsing {jsonl_path}: {exc}")
            skip_model = True
            files_to_remove.append(jsonl_path)
        if skip_model:
            continue

        # Enforce EEE
        model_results = [
            BenchmarkResult.from_dict(result_dict).to_eee_dict()
            for result_dict in model_results
        ]

        # Enforce minimum EuroEval version
        model_results = [
            result_dict
            for result_dict in model_results
            if result_dict["eval_library"]["version"].replace(r".dev0", "")
            >= MINIMUM_VERSION
        ]
        if not model_results:
            files_to_remove.append(jsonl_path)
            continue

        # Enforce non-empty name
        model_results = [
            result_dict
            for result_dict in model_results
            if result_dict["model_info"]["name"].strip() != ""
        ]

        # Validate metadata
        cache: dict[str, dict[str, bool]] = defaultdict(dict)
        for result_dict in model_results:
            for metadata_field in REQUIRED_METADATA_FIELDS:
                if (
                    metadata_field
                    not in result_dict["model_info"]["additional_details"]
                ):
                    model_id = jsonl_path.stem.replace("_", "/", 1)
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
                    result_dict["model_info"]["additional_details"][metadata_field] = (
                        value
                    )

        with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            for result_dict in model_results:
                jsonl_file.write(json.dumps(result_dict) + "\n")

    for path in files_to_remove:
        path.unlink()


if __name__ == "__main__":
    main()
