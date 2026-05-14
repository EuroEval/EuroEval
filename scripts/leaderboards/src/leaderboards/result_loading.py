"""Loading of results, to be converted into leaderboards."""

import json
import logging
import re
import tarfile
from functools import cache
from pathlib import Path

logger = logging.getLogger(__name__)


def load_raw_results() -> list[dict]:
    """Load raw results.

    Returns:
        The raw results.

    Raises:
        FileNotFoundError:
            If the raw results file is not found.
        ValueError:
            If the raw results file contains invalid JSON.
    """
    results_path = Path("results.tar.gz")
    if not results_path.exists():
        raise FileNotFoundError(f"Results file {results_path} not found.")

    logger.info(f"Loading raw results from {results_path}...")

    # Unpack the tar.gz file in memory and read the JSONL file
    with tarfile.open(results_path, "r:gz") as tar:
        results_file = tar.extractfile(member="results/results.jsonl")
        if results_file is None:
            raise FileNotFoundError(
                "results/results.jsonl not found in the tar.gz file."
            )
        result_lines = results_file.read().decode(encoding="utf-8").splitlines()
        logger.info(f"Loaded {len(result_lines):,} existing results.")

    # If there are new results, add them to the existing results
    new_results_path = Path("new_results.jsonl")
    if new_results_path.exists():
        with new_results_path.open() as f:
            new_result_lines = f.read().splitlines()
        result_lines.extend(new_result_lines)
        logger.info(f"Loaded {len(new_result_lines):,} new results.")
        new_results_path.unlink()

    # Parse each line as JSON, skipping empty lines
    records = list()
    for line_idx, line in enumerate(result_lines):
        if not line.strip():
            continue

        # We split on '}{' to handle cases where multiple JSON objects are on the
        # same line
        for record in re.split(pattern=r"(?<=})(?={)", string=line):
            if not record.strip():
                continue
            try:
                parsed_record = json.loads(record)
                # Convert from new Every Eval format to old EuroEval format
                converted_record = convert_to_old_format(record=parsed_record)
                records.append(converted_record)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON on line {line_idx:,}: {record}.")

    return records


def convert_to_old_format(record: dict) -> dict:
    """Convert a record from the new Every Eval format to the old EuroEval format.

    The new EEE format has:
    - evaluation_results: list of evaluation results with score_details.score
    - eval_library.additional_details.raw_results: JSON string of raw per-fold results

    The old format has:
    - results.raw: list of raw score dicts
    - results.total: dict with aggregated scores

    Args:
        record: A record from the JSONL file.

    Returns:
        The record with results converted to the old format.
    """
    # If it's already of the old non-EEE format, we don't need to convert it
    if "schema_version" not in record:
        return record

    # Convert from new format to old format
    additional_details = record["eval_library"].get("additional_details", {}) | record[
        "model_info"
    ].get("additional_details", {})
    new_record = dict(
        model=record["model_info"]["name"],
        results=dict(raw={}, total={}),
        euroeval_version=record["eval_library"]["version"],
    )
    new_record |= additional_details

    # Convert dtypes
    bool_columns = ["few_shot", "validation_split", "generative"]
    int_columns = ["num_model_parameters", "vocabulary_size", "max_sequence_length"]
    for column in bool_columns:
        new_record[column] = True if new_record[column] == "true" else False
    for column in int_columns:
        new_record[column] = int(  # pyrefly: ignore[no-matching-overload]
            new_record[column]
        )

    # Extract raw results from eval_library.additional_details.raw_results
    if new_record.get("raw_results", None) is not None:
        raw_results = json.loads(
            new_record["raw_results"]  # pyrefly: ignore[bad-argument-type]
        )
        new_record["results"]["raw"]["test"] = raw_results

    # Extract evaluation results and convert to old format
    # The evaluation_results contains entries like test_mcc, test_accuracy, etc.
    for eval_result in record.get("evaluation_results", []):
        evaluation_name = eval_result.get("evaluation_name", "")
        score = eval_result.get("score_details", {}).get("score", 0)

        # Convert metric name to match old format (e.g., "test_mcc" -> "test_mcc")
        # and add standard error (bootstrap CI width / 3.92 for 95% CI)
        if evaluation_name.startswith("test_"):
            metric_name = evaluation_name
        else:
            metric_name = f"test_{evaluation_name}"

        new_record["results"]["total"][metric_name] = score

        # Calculate standard error from confidence interval
        uncertainty = eval_result.get("score_details", {}).get("uncertainty", {})
        ci = uncertainty.get("confidence_interval", {})
        if ci.get("lower") is not None and ci.get("upper") is not None:
            # For 95% CI, width = 3.92 * SE, so SE = width / 3.92
            ci_width = ci["upper"] - ci["lower"]
            se = ci_width / 3.92
            new_record["results"]["total"][f"{metric_name}_se"] = se

    assert "schema_version" not in new_record, (
        f"Schema version should have been removed: {new_record}."
    )
    return new_record


@cache
def load_processed_results() -> list[dict]:
    """Load processed results.

    Returns:
        The processed results.

    Raises:
        FileNotFoundError:
            If the processed results file is not found.
        ValueError:
            If the processed results file contains invalid JSON.
    """
    results_path = Path("results.tar.gz")
    if not results_path.exists():
        raise FileNotFoundError("Processed results file not found.")

    logger.info(f"Loading processed results from {results_path}...")

    # Unpack the tar.gz file in memory and read the JSONL file
    with tarfile.open(results_path, "r:gz") as tar:
        results_file = tar.extractfile(member="results/results.processed.jsonl")
        if results_file is None:
            raise FileNotFoundError(
                "results/results.processed.jsonl not found in the tar.gz file."
            )
        result_lines = results_file.read().decode(encoding="utf-8").splitlines()

    # Parse each line as JSON, skipping empty lines
    results = list()
    for line_idx, line in enumerate(result_lines):
        if not line.strip():
            continue
        for record in line.replace("}{", "}\n{").split("\n"):
            if not record.strip():
                continue
            try:
                results.append(json.loads(record))
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON on line {line_idx:,}: {record}.")

    return results
