"""Accessors for fields on result records (EEE and old EuroEval formats)."""

from __future__ import annotations

import json
import re


def get_task(record: dict) -> str | None:
    """Get task name from record, supporting both EEE and old formats.

    Args:
        record:
            A result record in either EEE or old EuroEval format.

    Returns:
        The task name, or None if not found.
    """
    # EEE format: in eval_library.additional_details
    if "eval_library" in record:
        additional = record.get("eval_library", {}).get("additional_details", {})
        if "task" in additional:
            return additional["task"]
    # Old format: top-level task field
    return record.get("task")


def get_raw_results(record: dict) -> list[dict] | dict | None:
    """Get raw results from record, supporting both EEE and old formats.

    Args:
        record:
            A result record in either EEE or old EuroEval format.

    Returns:
        The raw results (list of dicts for EEE, dict or list for old format).
    """
    # EEE format: JSON string in eval_library.additional_details.raw_results
    if "eval_library" in record:
        additional = record.get("eval_library", {}).get("additional_details", {})
        raw_str = additional.get("raw_results")
        if raw_str and isinstance(raw_str, str):
            try:
                return json.loads(raw_str)
            except json.JSONDecodeError:
                return None
        if raw_str and isinstance(raw_str, list):
            return raw_str
    # Old format: record["results"]["raw"]
    results = record.get("results", {})
    if isinstance(results, dict):
        return results.get("raw")
    return None


def get_total_scores(record: dict) -> dict[str, float] | None:
    """Get total scores from record, supporting both EEE and old formats.

    Args:
        record:
            A result record in either EEE or old EuroEval format.

    Returns:
        Dict mapping score names to values, or None if not found.
    """
    # EEE format: aggregate from evaluation_results
    if "eval_library" in record and "evaluation_results" in record:
        scores = {}
        eval_results = record.get("evaluation_results", [])
        if isinstance(eval_results, list):
            for er in eval_results:
                if isinstance(er, dict):
                    name = er.get("evaluation_name", "")
                    score_details = er.get("score_details", {})
                    if isinstance(score_details, dict):
                        score = score_details.get("score")
                        if score is not None and isinstance(name, str) and name:
                            # Convert "test_mcc" -> {"test_mcc": 95.0}
                            scores[name] = float(score)
        return scores if scores else None
    # Old format: record["results"]["total"]
    results = record.get("results", {})
    if isinstance(results, dict):
        return results.get("total")
    return None


def get_version(record: dict) -> str | None:
    """Get version from record, supporting both EEE and old formats.

    Args:
        record:
            A result record in either EEE or old EuroEval format.

    Returns:
        The version string with .dev suffix stripped, or None if not found.
    """
    # EEE format: eval_library.version
    if "eval_library" in record:
        version = record.get("eval_library", {}).get("version")
        # Strip .dev suffix for comparison
        if version:
            return re.sub(r"\.dev\d+", "", version)
    # Old format: euroeval_version
    return record.get("euroeval_version")


def get_few_shot(record: dict) -> bool:
    """Get few_shot value from record, supporting both EEE and old formats.

    Args:
        record:
            A result record in either EEE or old EuroEval format.

    Returns:
        The few_shot boolean value, defaulting to True.
    """
    # EEE format: eval_library.additional_details.few_shot
    if "eval_library" in record:
        few_shot = (
            record.get("eval_library", {}).get("additional_details", {}).get("few_shot")
        )
        if few_shot is not None:
            if isinstance(few_shot, bool):
                return few_shot
            if isinstance(few_shot, str):
                return few_shot.lower() == "true"
    # Old format: top-level few_shot
    return record.get("few_shot", True)
