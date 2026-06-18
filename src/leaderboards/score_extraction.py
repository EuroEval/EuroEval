"""Extract and aggregate scores and metadata from processed result records."""

from __future__ import annotations

import logging
import statistics
import typing as t
from collections import defaultdict

from euroeval.logging_utils import log_once

from .link_generation import generate_model_url
from .record_fields import get_raw_results, get_task, get_total_scores, get_version
from .records import (
    extract_model_ids_from_record,
    get_dataset,
    get_model_name,
    plain_model_id,
)
from .task_metadata import task_metric_names

logger = logging.getLogger(__name__)


def group_results_by_model(
    results: list[dict[str, t.Any]],
) -> dict[str, dict[str, list[tuple[list[float], float, float]]]]:
    """Group results by model ID.

    Args:
        results:
            The processed results.

    Returns:
        The results grouped by model ID. The dict structure is
        model_id -> dataset -> list of (raw_scores, total_score, std_err).
    """
    model_scores: dict[str, dict[str, list[tuple[list[float], float, float]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for record in results:
        model_ids = extract_model_ids_from_record(record=record)
        dataset = get_dataset(record)
        if not dataset:
            continue

        task = get_task(record)
        if not task:
            continue
        primary, secondary = task_metric_names(task)
        metrics = [primary] + ([secondary] if secondary is not None else [])

        for metric_type, metric in zip(("primary", "secondary"), metrics):
            raw_results = get_raw_results(record)
            if raw_results is None:
                continue

            # Raw per-iteration scores are keyed by the bare metric name (e.g.
            # "mcc"), occasionally with a "test_" prefix.
            raw_scores: list[float] = []
            for result_dict in raw_results:
                if isinstance(result_dict, dict):
                    score = result_dict.get(
                        f"test_{metric}", result_dict.get(metric, -1)
                    )
                    if score >= 0:
                        raw_scores.append(score)

            if not raw_scores:
                continue

            total_scores = get_total_scores(record)
            if total_scores is None:
                continue

            # Total scores are keyed by evaluation name (e.g. "test_mcc"), but
            # fall back to the bare metric name when the prefix is absent.
            total_score_key = f"test_{metric}"
            std_err_key = f"test_{metric}_se"

            total_score_val = total_scores.get(total_score_key)
            if total_score_val is None:
                total_score_val = total_scores.get(metric)

            if total_score_val is None:
                log_once(
                    f"Could not find {metric_type} metric for {dataset!r} "
                    f"in {get_model_name(record)!r} ({total_score_key}). Only found "
                    f"{list(total_scores.keys())}.",
                    level=logging.WARNING,
                )
                continue

            total_score: float = float(total_score_val)

            # EEE records don't carry a std err, so compute it from raw scores.
            std_err: float = total_scores.get(std_err_key, 0.0)
            if std_err == 0.0 and len(raw_scores) > 1:
                try:
                    std_err = statistics.stdev(raw_scores) / (len(raw_scores) ** 0.5)
                except statistics.StatisticsError:
                    std_err = 0.0

            # Sometimes the raw scores are normalised to [0, 1], so we need to scale
            # them back to [0, 100]
            if max(raw_scores) <= 1:
                raw_scores = [score * 100 for score in raw_scores]

            for model_id in model_ids:
                model_scores[model_id][dataset].append(
                    (raw_scores, total_score, std_err)
                )

    return model_scores


def extract_model_metadata(
    results: list[dict[str, t.Any]],
) -> dict[str, dict[str, t.Any]]:
    """Extract metadata from the results.

    Args:
        results:
            The processed results.

    Returns:
        The metadata.
    """
    logger.info("Extracting model metadata...")
    metadata_dict: dict[str, dict[str, t.Any]] = defaultdict(dict)
    for record in results:
        model_ids = extract_model_ids_from_record(record=record)

        additional = record.get("model_info", {}).get("additional_details", {})
        num_params_raw = additional.get("num_model_parameters", "-1")
        vocab_size_raw = additional.get("vocabulary_size", "-1")
        context_raw = additional.get("max_sequence_length", "-1")
        merge_raw = additional.get("merge", "false")
        generative_type = additional.get("generative_type", None)
        commercially_licensed = additional.get("commercially_licensed", False)
        open_weights = additional.get("open", None)
        trained_from_scratch = additional.get("trained_from_scratch", None)
        # Models below MINIMUM_NUMBER_OF_MODEL_RECORDS are dropped by
        # `process_results` before `add_missing_entries` runs, so their stored
        # records never get a `model_url`, yet they still appear in the
        # leaderboard (which reads raw records without that filter). Generate
        # the URL on the fly when it's missing so they still get an anchor tag.
        model_url = additional.get("model_url", None)
        if model_url is None:
            model_url = generate_model_url(
                model_id=plain_model_id(get_model_name(record))
            )

        num_params = _to_float_or_nan(num_params_raw)
        vocab_size = _to_float_or_nan(vocab_size_raw)
        context = _to_float_or_nan(context_raw)
        merge = _to_bool(merge_raw)

        # The frontend hides version columns and doesn't sort by them, so the
        # plain version string is sufficient.
        version = get_version(record) or "<9.2.0"
        dataset = get_dataset(record)

        for model_id in model_ids:
            metadata_dict[model_id].update(
                {
                    "parameters": num_params,
                    "vocabulary_size": vocab_size,
                    "context": context,
                    "generative_type": generative_type,
                    "commercial": commercially_licensed,
                    "merge": merge,
                    "open": open_weights,
                    "trained_from_scratch": trained_from_scratch,
                    "model_url": model_url,
                }
            )
            if dataset:
                metadata_dict[model_id][f"{dataset}_version"] = version

    logger.info("Extracted model metadata.")
    return metadata_dict


def _to_float_or_nan(val: str | float | int | None) -> float:
    """Coerce a metadata value to a non-negative float, else NaN.

    Args:
        val:
            The raw value, which may be a number, numeric string, or None.

    Returns:
        The value as a float if it is non-negative, otherwise NaN.
    """
    if isinstance(val, int | float):
        return val if val >= 0 else float("nan")
    if isinstance(val, str):
        try:
            num = float(val)
            return num if num >= 0 else float("nan")
        except ValueError:
            return float("nan")
    return float("nan")


def _to_bool(val: str | bool | None) -> bool:
    """Coerce a metadata value to a boolean.

    Args:
        val:
            The raw value, which may be a bool, "true"/"false" string, or None.

    Returns:
        The boolean value, defaulting to False.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() == "true"
    return False
