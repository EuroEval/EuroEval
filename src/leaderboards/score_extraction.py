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
    # Track whether each model's URL is explicit (from record) vs generated fallback
    model_url_explicit: dict[str, bool] = {}
    for record in results:
        model_ids = extract_model_ids_from_record(record=record)

        additional = record.get("model_info", {}).get("additional_details", {})
        num_params_raw = additional.get("num_model_parameters", "-1")
        vocab_size_raw = additional.get("vocabulary_size", "-1")
        context_raw = additional.get("max_sequence_length", "-1")
        merge_raw = additional.get("merge", "false")
        # Track which metadata fields are explicitly present (not None/empty) vs missing
        generative_type = additional.get("generative_type", None)
        generative_type_present = (
            "generative_type" in additional
            and additional["generative_type"] is not None
        )
        commercially_licensed = additional.get("commercially_licensed", False)
        commercial_present = (
            "commercially_licensed" in additional
            and additional["commercially_licensed"] is not None
        )
        open_weights = additional.get("open", None)
        open_present = "open" in additional and additional["open"] is not None
        trained_from_scratch = additional.get("trained_from_scratch", None)
        trained_present = (
            "trained_from_scratch" in additional
            and additional["trained_from_scratch"] is not None
        )
        # Models below MINIMUM_NUMBER_OF_MODEL_RECORDS are dropped by
        # `process_results` before `add_missing_entries` runs, so their stored
        # records never get a `model_url`, yet they still appear in the
        # leaderboard (which reads raw records without that filter). Generate
        # the URL on the fly when it's missing so they still get an anchor tag.
        model_url = additional.get("model_url", None)
        model_url_explicit_record = (
            "model_url" in additional
            and additional["model_url"] is not None
            and additional["model_url"] != ""
        )
        # model_url_present indicates if we have a URL to store (explicit or generated)
        model_url_present = model_url_explicit_record
        if model_url is None:
            model_url = generate_model_url(
                model_id=plain_model_id(get_model_name(record))
            )
            # Mark as present (we have a URL to potentially store)
            # but model_url_explicit_record remains False (it's generated, not explicit)
            if model_url is not None:
                model_url_present = True

        num_params = _to_float_or_nan(num_params_raw)
        vocab_size = _to_float_or_nan(vocab_size_raw)
        context = _to_float_or_nan(context_raw)
        merge_raw = additional.get("merge", "false")
        merge = _to_bool(merge_raw)
        # For merge, key must exist and not be None to be considered present
        # ("false" string is a valid explicit value, not a default)
        merge_present = "merge" in additional and additional["merge"] is not None

        # The frontend hides version columns and doesn't sort by them, so the
        # plain version string is sufficient.
        version = get_version(record) or "<9.2.0"
        dataset = get_dataset(record)
        num_failed = get_num_failed_instances(record)

        for model_id in model_ids:
            existing = metadata_dict[model_id]
            # Merge metadata: only update if the new value is "better" than the
            # existing one. This prevents stale or misfiled records from
            # overwriting enriched metadata with missing/default values. Only
            # compare when the field is explicitly present in the record.
            # Float fields (parameters, vocabulary_size, context) - use key
            # presence checks (not truthiness) to preserve 0 values
            if "parameters" in existing:
                if _is_better_metadata(
                    new_value=num_params,
                    old_value=existing["parameters"],
                    field="parameters",
                ):
                    existing["parameters"] = num_params
            else:
                existing["parameters"] = num_params
            if "vocabulary_size" in existing:
                if _is_better_metadata(
                    new_value=vocab_size,
                    old_value=existing["vocabulary_size"],
                    field="vocabulary_size",
                ):
                    existing["vocabulary_size"] = vocab_size
            else:
                existing["vocabulary_size"] = vocab_size
            if "context" in existing:
                if _is_better_metadata(
                    new_value=context, old_value=existing["context"], field="context"
                ):
                    existing["context"] = context
            else:
                existing["context"] = context
            # Boolean/string fields - only update if explicitly present
            # Use explicit key presence checks (not truthiness) to preserve False values
            if generative_type_present:
                if "generative_type" in existing:
                    if _is_better_metadata(
                        new_value=generative_type,
                        old_value=existing["generative_type"],
                        field="generative_type",
                    ):
                        existing["generative_type"] = generative_type
                else:
                    existing["generative_type"] = generative_type
            if commercial_present:
                if "commercial" in existing:
                    if _is_better_metadata(
                        new_value=commercially_licensed,
                        old_value=existing["commercial"],
                        field="commercial",
                    ):
                        existing["commercial"] = commercially_licensed
                else:
                    existing["commercial"] = commercially_licensed
            if merge_present:
                if "merge" in existing:
                    if _is_better_metadata(
                        new_value=merge, old_value=existing["merge"], field="merge"
                    ):
                        existing["merge"] = merge
                else:
                    existing["merge"] = merge
            if open_present:
                if "open" in existing:
                    if _is_better_metadata(
                        new_value=open_weights, old_value=existing["open"], field="open"
                    ):
                        existing["open"] = open_weights
                else:
                    existing["open"] = open_weights
            if trained_present:
                if "trained_from_scratch" in existing:
                    if _is_better_metadata(
                        new_value=trained_from_scratch,
                        old_value=existing["trained_from_scratch"],
                        field="trained_from_scratch",
                    ):
                        existing["trained_from_scratch"] = trained_from_scratch
                else:
                    existing["trained_from_scratch"] = trained_from_scratch
            if model_url_present:
                if "model_url" in existing:
                    # Check if existing URL is explicit (from record) vs generated
                    existing_is_explicit = model_url_explicit.get(model_id, False)
                    # Only allow update if:
                    # 1. Existing is generated (not explicit), or
                    # 2. Both are explicit and new is "better" per _is_better_metadata
                    if not existing_is_explicit:
                        # Existing is generated; explicit always wins, generated
                        # only fills if missing (already handled by else branch)
                        if model_url_explicit_record:
                            # New is explicit, replace generated
                            existing["model_url"] = model_url
                            model_url_explicit[model_id] = True
                        # else: both are generated, keep existing (first-come)
                    else:
                        # Existing is explicit; only replace with better explicit
                        if model_url_explicit_record and _is_better_metadata(
                            new_value=model_url,
                            old_value=existing["model_url"],
                            field="model_url",
                        ):
                            existing["model_url"] = model_url
                            # model_url_explicit already True
                else:
                    existing["model_url"] = model_url
                    model_url_explicit[model_id] = model_url_explicit_record
            if dataset:
                existing[f"{dataset}_version"] = version
                # Include failure counts for all versions. For versions after
                # 17.5.0, these count only genuine scoring failures. For 17.5.0
                # and earlier, they also include samples where the fallback
                # label was correct (recoverable errors) — the frontend adds a
                # caveat in the tooltip for those versions.
                if num_failed is not None:
                    existing[f"{dataset}_failures"] = num_failed
                    scored = _scored_count(record=record, dataset=dataset)
                    if scored is not None:
                        existing[f"{dataset}_scored"] = scored

    # Ensure every model has all standard metadata keys with defaults.
    # This prevents KeyError/AssertionError in generate_dataframe() which
    # expects these columns to exist for all models. Note that the record
    # loop above already generates fallback URLs when appropriate, so the
    # final fill just uses None for model_url.
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
    for model_id, metadata in metadata_dict.items():
        for key, default_value in standard_keys_defaults.items():
            if key not in metadata:
                metadata[key] = default_value

    logger.info("Extracted model metadata.")
    return metadata_dict


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
