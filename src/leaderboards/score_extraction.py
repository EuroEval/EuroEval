"""Extract and aggregate scores and metadata from processed result records."""

from __future__ import annotations

import logging
import math
import statistics
import typing as t
from collections import defaultdict

from euroeval.logging_utils import log_once

from .link_generation import generate_model_url
from .record_fields import (
    deduplicate_records,
    get_num_failed_instances,
    get_raw_results,
    get_task,
    get_total_scores,
    get_validation_split,
    get_version,
)
from .records import (
    extract_model_ids_from_record,
    get_dataset,
    get_model_name,
    plain_model_id,
)
from .split_sizes import get_split_sizes
from .task_metadata import dataset_sources, task_metric_names

logger = logging.getLogger(__name__)


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
    model_url_explicit: dict[str, bool] = {}

    for record in results:
        model_ids = extract_model_ids_from_record(record=record)
        (
            extracted,
            presence_flags,
            explicit_flags,  # noqa: F841
            model_url,
            model_url_explicit_rec,
            version,
            num_failed,
        ) = _extract_metadata_from_record(record)
        dataset = get_dataset(record)

        for model_id in model_ids:
            existing = metadata_dict[model_id]

            # Update float fields
            _update_metadata_field(
                existing=existing,
                new_value=extracted["parameters"],
                field="parameters",
                is_present=True,  # Always set (defaults to NaN)
            )
            _update_metadata_field(
                existing=existing,
                new_value=extracted["vocabulary_size"],
                field="vocabulary_size",
                is_present=True,
            )
            _update_metadata_field(
                existing=existing,
                new_value=extracted["context"],
                field="context",
                is_present=True,
            )

            # Update presence-checked fields
            for field in (
                "generative_type",
                "commercial",
                "merge",
                "open",
                "trained_from_scratch",
            ):
                _update_metadata_field(
                    existing=existing,
                    new_value=extracted[field],
                    field=field,
                    is_present=presence_flags[field],
                )

            # Handle model_url separately (special explicit vs generated logic)
            _update_model_url(
                existing=existing,
                model_url=model_url,
                model_url_explicit=model_url_explicit_rec,
                model_url_explicit_map=model_url_explicit,
                model_id=model_id,
            )

            # Add dataset-specific fields
            if dataset:
                existing[f"{dataset}_version"] = version
                if num_failed is not None:
                    existing[f"{dataset}_failures"] = num_failed
                    scored = _scored_count(record=record, dataset=dataset)
                    if scored is not None:
                        existing[f"{dataset}_scored"] = scored

    _ensure_standard_metadata_keys(metadata_dict=metadata_dict)
    logger.info("Extracted model metadata.")
    return metadata_dict


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
    # Deduplicate up front so each leaderboard row shows one score per metric.
    # `load_raw_results` is cached and populated before `process_results`
    # rewrites the per-model files, so the records reaching here can still
    # contain the pre-dedup duplicates; collapse them by hash regardless.
    results = deduplicate_records(records=results)
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

            # Sometimes the raw scores are normalised to [0, 1], so we need to scale
            # them back to [0, 100]
            scale_factor = 100.0 if max(raw_scores) <= 1 else 1.0
            raw_scores = [score * scale_factor for score in raw_scores]

            # EEE records don't carry a std err, so compute it from raw scores.
            # Fallback computed after scaling so std_err matches the displayed scores.
            std_err: float = total_scores.get(std_err_key, 0.0)
            # Scale std_err to match the scaled raw scores
            std_err = std_err * scale_factor
            if std_err == 0.0 and len(raw_scores) > 1:
                try:
                    std_err = statistics.stdev(raw_scores) / (len(raw_scores) ** 0.5)
                except statistics.StatisticsError:
                    std_err = 0.0

            for model_id in model_ids:
                model_scores[model_id][dataset].append(
                    (raw_scores, total_score, std_err)
                )

    return model_scores


def _update_metadata_field(
    existing: dict[str, t.Any],
    new_value: bool | str | float | int | None,
    field: str,
    is_present: bool,
) -> None:
    """Update a single metadata field if the new value is better.

    Args:
        existing:
            The existing metadata dict to update.
        new_value:
            The new value to potentially store.
        field:
            The field name.
        is_present:
            Whether the field is explicitly present in the record.
    """
    if not is_present:
        return
    if field in existing:
        if _is_better_metadata(
            new_value=new_value, old_value=existing[field], field=field
        ):
            existing[field] = new_value
    else:
        existing[field] = new_value


def _ensure_standard_metadata_keys(metadata_dict: dict[str, dict[str, t.Any]]) -> None:
    """Ensure every model has all standard metadata keys with defaults.

    This prevents KeyError/AssertionError in generate_dataframe() which
    expects these columns to exist for all models.

    Args:
        metadata_dict:
            The metadata dict to ensure keys for.
    """
    standard_keys_defaults: dict[str, t.Any] = {
        "parameters": math.nan,
        "vocabulary_size": math.nan,
        "context": math.nan,
        "generative_type": None,
        "commercial": False,
        "merge": False,
        "open": None,
        "trained_from_scratch": None,
        "model_url": None,
    }
    for metadata in metadata_dict.values():
        for key, default_value in standard_keys_defaults.items():
            if key not in metadata:
                metadata[key] = default_value


def _scored_count(record: dict[str, t.Any], dataset: str) -> int | None:
    """Compute the total number of scored samples for a (model, dataset) eval.

    Failure counts are summed across the bootstrap iterations, so the matching
    denominator is ``num_iterations * split_size``, where the split is the
    validation split for validation-split runs and the test split otherwise.

    Args:
        record:
            A result record in EEE format.
        dataset:
            The dataset name (e.g. ``"conll-nl"``).

    Returns:
        The total number of scored samples, or None if it cannot be determined.
    """
    raw_results = get_raw_results(record)
    if not raw_results:
        return None
    source = dataset_sources().get(dataset)
    if source is None:
        return None
    sizes = get_split_sizes(source)
    if not sizes:
        return None
    split = "val" if get_validation_split(record) else "test"
    size = sizes.get(split)
    if size is None:
        return None
    return len(raw_results) * size


def _extract_metadata_from_record(
    record: dict[str, t.Any],
) -> tuple[
    dict[str, t.Any],
    dict[str, bool],
    dict[str, bool],
    str | None,
    bool,
    str,
    int | None,
]:
    """Extract metadata fields from a single record.

    Args:
        record:
            The record to extract metadata from.

    Returns:
        Tuple of (metadata_dict, presence_flags, explicit_flags, model_url,
        model_url_explicit, version, num_failed).
    """
    additional = record.get("model_info", {}).get("additional_details", {})
    num_params_raw = additional.get("num_model_parameters", "-1")
    vocab_size_raw = additional.get("vocabulary_size", "-1")
    context_raw = additional.get("max_sequence_length", "-1")

    # Build metadata dict
    metadata: dict[str, t.Any] = {
        "parameters": _to_float_or_nan(num_params_raw),
        "vocabulary_size": _to_float_or_nan(vocab_size_raw),
        "context": _to_float_or_nan(context_raw),
        "generative_type": additional.get("generative_type", None),
        "commercial": additional.get("commercially_licensed", False),
        "merge": _to_bool(additional.get("merge", "false")),
        "open": additional.get("open", None),
        "trained_from_scratch": additional.get("trained_from_scratch", None),
    }

    # Track which fields are explicitly present (not None/empty)
    presence_flags: dict[str, bool] = {
        "generative_type": "generative_type" in additional
        and additional["generative_type"] is not None,
        "commercial": "commercially_licensed" in additional
        and additional["commercially_licensed"] is not None,
        "merge": "merge" in additional and additional["merge"] is not None,
        "open": "open" in additional and additional["open"] is not None,
        "trained_from_scratch": "trained_from_scratch" in additional
        and additional["trained_from_scratch"] is not None,
    }

    # Handle model_url - generate fallback if missing
    model_url = additional.get("model_url", None)
    model_url_explicit = (
        "model_url" in additional
        and additional["model_url"] is not None
        and additional["model_url"] != ""
    )
    if model_url is None:
        model_url = generate_model_url(model_id=plain_model_id(get_model_name(record)))

    version = get_version(record) or "<9.2.0"
    num_failed = get_num_failed_instances(record)

    return (
        metadata,
        presence_flags,
        {"model_url": model_url_explicit},  # Kept for API compatibility
        model_url,
        model_url_explicit,
        version,
        num_failed,
    )


def _update_model_url(
    existing: dict[str, t.Any],
    model_url: str | None,
    model_url_explicit: bool,
    model_url_explicit_map: dict[str, bool],
    model_id: str,
) -> None:
    """Update model_url field with explicit vs generated URL logic.

    Args:
        existing:
            The existing metadata dict to update.
        model_url:
            The model URL (explicit or generated).
        model_url_explicit:
            Whether the URL is explicit (from record) vs generated.
        model_url_explicit_map:
            Map tracking which models have explicit URLs.
        model_id:
            The model ID.
    """
    if model_url is None:
        return
    if "model_url" in existing:
        existing_is_explicit = model_url_explicit_map.get(model_id, False)
        if not existing_is_explicit and model_url_explicit:
            # Existing is generated; explicit wins
            existing["model_url"] = model_url
            model_url_explicit_map[model_id] = True
        elif existing_is_explicit and model_url_explicit:
            # Both explicit; use _is_better_metadata comparison
            if _is_better_metadata(
                new_value=model_url, old_value=existing["model_url"], field="model_url"
            ):
                existing["model_url"] = model_url
    else:
        existing["model_url"] = model_url
        model_url_explicit_map[model_id] = model_url_explicit


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


def _is_better_metadata(
    new_value: bool | str | float | None,
    old_value: bool | str | float | None,
    field: str,
) -> bool:
    """Check if new metadata value is "better" than the old one.

    A value is "better" if it's more informative (non-null/non-default) when
    the old value is null/default. Used to prevent stale records from
    overwriting enriched metadata during extraction.

    Args:
        new_value:
            The new metadata value from the current record.
        old_value:
            The existing metadata value already stored.
        field:
            The field name being compared.

    Returns:
        True if the new value should replace the old one.
    """
    # Prefer non-None over None
    if old_value is None and new_value is not None:
        return True
    if old_value is not None and new_value is None:
        return False

    # For float fields (parameters, vocabulary_size, context), prefer non-NaN
    # over NaN
    if field in ("parameters", "vocabulary_size", "context"):
        if isinstance(old_value, float) and math.isnan(old_value):
            if isinstance(new_value, float) and not math.isnan(new_value):
                return True
            return False
        if isinstance(new_value, float) and math.isnan(new_value):
            return False

    # For boolean fields, prefer present (non-None) over missing (None).
    # Explicit False is legitimate metadata (e.g. merge=False, open=False)
    # and should be preserved against later stale/conflicting records.
    # When both are present (even if different), neither is "better" -
    # returning False means the new value won't overwrite the old one.
    if field in ("commercial", "merge", "open", "trained_from_scratch"):
        if old_value is None and new_value is not None:
            return True
        if old_value is not None and new_value is None:
            return False
        # Both present: don't overwrite (preserve existing)
        return False

    # For generative_type, prefer non-empty over empty
    # When both are non-empty, preserve existing (don't overwrite)
    if field == "generative_type":
        if not old_value and new_value:
            return True
        if old_value and not new_value:
            return False
        # Both non-empty: preserve existing
        return False

    # For model_url, prefer non-empty over empty.
    # Note: explicit vs generated fallback distinction is handled in
    # extract_model_metadata, not here. This function only handles
    # empty vs non-empty comparison.
    if field == "model_url":
        if not old_value and new_value:
            return True
        if old_value and not new_value:
            return False
        # Both non-empty: preserve existing
        return False

    # Default: prefer new value (preserves existing behaviour for equal values)
    return True


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


def _process_record_scores(
    record: dict[str, t.Any],
    model_ids: list[str],
    dataset: str,
    raw_results: list[t.Any],
    total_scores: dict[str, t.Any],
    model_scores: dict[str, dict[str, list[tuple[list[float], float, float]]]],
) -> None:
    """Process scores for a single record and add to model_scores.

    Args:
        record:
            The record being processed.
        model_ids:
            List of model IDs from the record.
        dataset:
            Dataset name.
        raw_results:
            Raw results list.
        total_scores:
            Total scores dict.
        model_scores:
            Accumulator dict to update.
    """
    task = get_task(record)
    if not task:
        return
    primary, secondary = task_metric_names(task)
    metrics: list[tuple[str, str]] = [("primary", primary)]
    if secondary is not None:
        metrics.append(("secondary", secondary))

    model_name = get_model_name(record)
    for metric_type, metric in metrics:
        raw_scores = _extract_raw_scores(raw_results=raw_results, metric=metric)
        if not raw_scores:
            continue

        total_score = _get_total_score_value(
            total_scores=total_scores,
            metric=metric,
            dataset=dataset,
            model_name=model_name,
            metric_type=metric_type,
        )
        if total_score is None:
            continue

        # Scale raw scores to [0, 100] if normalised to [0, 1]
        scale_factor = 100.0 if max(raw_scores) <= 1 else 1.0
        raw_scores = [score * scale_factor for score in raw_scores]

        std_err = _compute_std_err(
            raw_scores=raw_scores,
            total_scores=total_scores,
            metric=metric,
            scale_factor=scale_factor,
        )

        for model_id in model_ids:
            model_scores[model_id][dataset].append((raw_scores, total_score, std_err))


def _compare_float_metadata(new_value: float | None, old_value: float | None) -> bool:
    """Compare float metadata fields, preferring non-NaN over NaN.

    Args:
        new_value:
            The new float value.
        old_value:
            The existing float value.

    Returns:
        True if new_value is better (non-NaN when old is NaN).
    """
    if isinstance(old_value, float) and math.isnan(old_value):
        if isinstance(new_value, float) and not math.isnan(new_value):
            return True
        return False
    if isinstance(new_value, float) and math.isnan(new_value):
        return False
    return True


def _compute_std_err(
    raw_scores: list[float],
    total_scores: dict[str, t.Any],
    metric: str,
    scale_factor: float,
) -> float:
    """Compute standard error from raw scores.

    EEE records don't carry a std err, so compute it from raw scores.
    Fallback computed after scaling so std_err matches the displayed scores.

    Args:
        raw_scores:
            List of raw scores (already scaled).
        total_scores:
            Dictionary of total scores.
        metric:
            The metric name.
        scale_factor:
            Factor to scale std_err by.

    Returns:
        The computed standard error.
    """
    std_err_key = f"test_{metric}_se"
    std_err: float = total_scores.get(std_err_key, 0.0) * scale_factor
    if std_err == 0.0 and len(raw_scores) > 1:
        try:
            std_err = statistics.stdev(raw_scores) / (len(raw_scores) ** 0.5)
        except statistics.StatisticsError:
            std_err = 0.0
    return std_err


def _get_total_score_value(
    total_scores: dict[str, t.Any],
    metric: str,
    dataset: str,
    model_name: str,
    metric_type: str,
) -> float | None:
    """Extract total score value from total scores dict.

    Total scores are keyed by evaluation name (e.g. "test_mcc"), but fall back
    to the bare metric name when the prefix is absent.

    Args:
        total_scores:
            Dictionary of total scores.
        metric:
            The metric name.
        dataset:
            Dataset name for logging.
        model_name:
            Model name for logging.
        metric_type:
            Type of metric ("primary" or "secondary") for logging.

    Returns:
        The total score as float, or None if not found.
    """
    total_score_key = f"test_{metric}"
    total_score_val = total_scores.get(total_score_key)
    if total_score_val is None:
        total_score_val = total_scores.get(metric)

    if total_score_val is None:
        log_once(
            f"Could not find {metric_type} metric for {dataset!r} "
            f"in {model_name!r} ({total_score_key}). Only found "
            f"{list(total_scores.keys())}.",
            level=logging.WARNING,
        )
        return None
    return float(total_score_val)


def _extract_raw_scores(raw_results: list[t.Any], metric: str) -> list[float]:
    """Extract raw scores from raw results for a given metric.

    Raw per-iteration scores are keyed by the bare metric name (e.g. "mcc"),
    occasionally with a "test_" prefix.

    Args:
        raw_results:
            List of raw result dictionaries.
        metric:
            The metric name to extract.

    Returns:
        List of extracted scores (>=0).
    """
    raw_scores: list[float] = []
    for result_dict in raw_results:
        if isinstance(result_dict, dict):
            score = result_dict.get(f"test_{metric}", result_dict.get(metric, -1))
            if score >= 0:
                raw_scores.append(score)
    return raw_scores


def _compare_presence_metadata(
    new_value: bool | str | float | int | None,
    old_value: bool | str | float | int | None,
) -> bool:
    """Compare fields by presence.

    Prefer present (non-None/non-empty) over missing (None/empty). For string
    fields (generative_type, model_url), empty strings are treated as absent.
    Explicit False is legitimate metadata and should be preserved.
    When both are present, neither is "better" (return False to preserve).

    Args:
        new_value:
            The new value.
        old_value:
            The existing value.

    Returns:
        True if new_value should replace old_value.
    """
    # Treat empty strings as absent for string fields
    new_is_present = new_value is not None and (
        not isinstance(new_value, str) or new_value != ""
    )
    old_is_present = old_value is not None and (
        not isinstance(old_value, str) or old_value != ""
    )

    if not old_is_present and new_is_present:
        return True
    if old_is_present and not new_is_present:
        return False
    # Both present or both absent: don't overwrite (preserve existing)
    return False
