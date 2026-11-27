"""Validation for submitted EuroEval Evaluation Exchange records."""

from __future__ import annotations

import json
import math
import re
import typing as t

from fastapi import HTTPException

from .canary import CANARY_RESULT_DATASET, validate_canary_evidence
from .policy import ModelType, language_group


class CheckedRecord(t.TypedDict):
    """Validated result identity and failure summary."""

    identity: str
    failed: int
    warnings: list[str]


def validate_record(
    record: object,
    model_id: str,
    revision: str,
    language: str,
    euroeval_version: str | None = None,
    model_type: ModelType | None = None,
    require_revision: bool = True,
) -> CheckedRecord:
    """Validate an EEE result against its leased model and language.

    Args:
        record: Submitted record.
        model_id: Expected model identifier.
        revision: Expected model revision.
        language: Leased language.
        euroeval_version: Optional required EuroEval version.
        model_type: Optional encoder or generative model type.
        require_revision: Whether historical missing revisions are rejected.

    Returns:
        Canonical record identity, failed count, and warnings.

    """
    if not _object(record):
        _fail("record must be an EEE JSON object.")
    info, library, results = (
        record.get("model_info"),
        record.get("eval_library"),
        record.get("evaluation_results"),
    )
    if (
        record.get("schema_version") not in ("0.2.1", "0.3.0")
        or not _object(info)
        or not _object(library)
        or not isinstance(results, list)
    ):
        _fail("record is not a supported EEE record.")
    extra = info.get("additional_details")
    aliases: list[object] = [info.get("id"), info.get("name"), info.get("aliases")]
    if _object(extra):
        aliases.append(extra.get("aliases"))
    flattened = [
        item
        for alias in aliases
        for item in (alias if isinstance(alias, list) else [alias])
    ]
    if model_id not in flattened:
        _fail(f"record model identity does not match {model_id}.")
    missing = "revision" not in info
    if missing and require_revision or not missing and info.get("revision") != revision:
        _fail("record model_info.revision is missing or does not match the lease.")
    version = library.get("version")

    def strip(value: str) -> str:
        return re.sub(r"\.dev\d+$", "", value)

    if (
        library.get("name") != "euroeval"
        or not isinstance(version, str)
        or not version
        or euroeval_version
        and strip(version) != strip(euroeval_version)
    ):
        _fail("record has an invalid eval_library name or version.")
    details = library.get("additional_details")
    if (
        not _object(details)
        or not isinstance(details.get("dataset"), str)
        or not details["dataset"]
        or not isinstance(details.get("task"), str)
        or not details["task"]
    ):
        _fail("record has no canonical dataset/task identity.")
    dataset_value = details["dataset"]
    dataset = t.cast(str, dataset_value)
    if dataset == CANARY_RESULT_DATASET or dataset.startswith(
        CANARY_RESULT_DATASET + "-"
    ):
        if details["task"] != "contamination-detection":
            _fail("canary result has an invalid task.")
        evidence = _json(
            details.get("contamination_canary_evidence"),
            "contamination_canary_evidence is not valid JSON.",
        )
        validate_canary_evidence(evidence, model_id, revision)
        if (
            _object(evidence)
            and model_type == "encoder"
            and (
                evidence.get("status") != "not_applicable"
                or evidence.get("reason") != "encoder"
            )
        ):
            _fail("encoder canary evidence must be not_applicable/encoder.")
        if (
            _object(evidence)
            and model_type == "generative"
            and evidence.get("reason") == "encoder"
        ):
            _fail("generative canary evidence cannot use the encoder status.")
    raw = _json(
        details.get("raw_results"), "additional_details.raw_results is not valid JSON."
    )
    if not isinstance(raw, list):
        _fail("additional_details.raw_results is not valid JSON.")
    has_raw, raw_count = _failures(raw)
    claims = [
        record.get(key, ...) for key in ("num_failed_instances", "failed_instances")
    ] + [details.get(key, ...) for key in ("num_failed_instances", "failed_instances")]
    claims += [
        r.get("score_details", {}).get("details", {}).get("num_failed_instances", ...)
        if _object(r)
        and _object(r.get("score_details"))
        and _object(_object(r.get("score_details")).get("details"))
        else ...
        for r in results
    ]
    claims = [claim for claim in claims if claim is not ...]
    failed = (
        raw_count
        if has_raw
        else _count(details.get("num_failed_instances", claims[0] if claims else 0))
    )
    if any(_count(claim) != failed for claim in claims):
        _fail("failed instance count is inconsistent with raw results.")
    warnings = [f"{failed} failed instances"] if failed else []
    if missing:
        warnings.append("Legacy record is missing model_info.revision")
    group = language_group(language)
    langs = _json(
        details.get("languages"), "additional_details.languages is not valid JSON."
    )
    if (
        group is None
        or not isinstance(langs, list)
        or not langs
        or any(
            not isinstance(item, str) or language_group(item) != group for item in langs
        )
    ):
        _fail(
            "record languages must be a non-empty subset of the leased language group."
        )
    rec_lang = details.get("language", record.get("language"))
    if rec_lang is not None and (
        not isinstance(rec_lang, str) or rec_lang not in langs
    ):
        _fail("record language is inconsistent with languages.")

    def flag(value: object) -> bool | None | t.Literal["invalid"]:
        if value is None or isinstance(value, bool):
            return value
        if value == "true":
            return True
        if value == "false":
            return False
        return "invalid"

    split, shot = flag(details.get("validation_split")), flag(details.get("few_shot"))
    if split == "invalid" or shot == "invalid":
        _fail("validation_split and few_shot must be boolean or null.")
    for result in results:
        _metric(result, dataset, warnings)
    if not results:
        _fail("record contains no results.")
    return {
        "identity": json.dumps([model_id, dataset, split, shot], separators=(",", ":")),
        "failed": failed,
        "warnings": warnings,
    }


def _count(value: object) -> int:
    if isinstance(value, bool):
        _fail("failed instance count is invalid.")
    if isinstance(value, str):
        try:
            value = json.loads(
                value,
                parse_constant=lambda _: _fail("failed instance count is invalid."),
            )
        except (ValueError, TypeError):
            try:
                value = int(value)
            except ValueError:
                _fail("failed instance count is invalid.")
    if isinstance(value, list):
        value = len(value)
    if not isinstance(value, int) or value < 0 or value > 2**53 - 1:
        _fail("failed instance count is invalid.")
    return value


def _fail(message: str) -> t.NoReturn:
    raise HTTPException(status_code=422, detail=message)


def _failures(raw: list[object]) -> tuple[bool, int]:
    entries: list[int] = []

    def visit(value: object, found: list[int]) -> None:
        if isinstance(value, list):
            for child in value:
                visit(child, found)
        elif _object(value):
            for key, child in value.items():
                if key == "failed_instances":
                    parsed = _json(child, "failed instance count is invalid.")
                    found.append(_count(parsed))
                else:
                    visit(child, found)

    for iteration in raw:
        found: list[int] = []
        visit(iteration, found)
        if found:
            entries.append(max(found))
    return bool(entries), sum(entries)


def _json(value: object, message: str) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value, parse_constant=lambda _: _fail(message))
    except (ValueError, TypeError):
        _fail(message)


def _object(value: object) -> t.TypeGuard[dict[str, object]]:
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def _metric(result: object, dataset: str, warnings: list[str]) -> None:
    if not _object(result):
        _fail("evaluation_results contains an invalid item.")
    name = result.get("evaluation_name", result.get("metric_name", result.get("name")))
    score_details = result.get("score_details")
    source = result.get("source_data")
    metric = result.get("metric_config")
    if (
        not isinstance(name, str)
        or not name
        or not _object(score_details)
        or not _object(source)
        or not _object(metric)
    ):
        _fail(
            "each result needs a metric, score details, source data, and metric config."
        )
    source_dataset = source.get(
        "dataset_name", source.get("dataset", source.get("name"))
    )
    global_speed = bool(re.search(r"(^|[_ -])speed($|[_ -])", name, re.I)) and (
        not source_dataset
        or isinstance(source_dataset, str)
        and re.search(r"global.*speed|^global$", source_dataset, re.I)
    )
    if not global_speed and source_dataset != dataset:
        _fail("evaluation source_data is inconsistent with the record dataset.")
    if (
        not isinstance(metric.get("lower_is_better"), bool)
        or "score_type" in metric
        and not isinstance(metric["score_type"], str)
    ):
        _fail("every evaluation result needs a valid metric config.")
    for bound in (metric.get("min_score"), metric.get("max_score")):
        valid_number = (
            isinstance(bound, (int, float))
            and not isinstance(bound, bool)
            and math.isfinite(bound)
        )
        valid_infinity = isinstance(bound, str) and bound in ("Infinity", "-Infinity")
        if bound is not None and not (valid_number or valid_infinity):
            _fail("metric bounds must be finite numbers or explicit infinities.")
    score = score_details.get("score")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(score)
    ):
        _fail("every evaluation score must be finite.")
    if (
        isinstance(metric.get("min_score"), (int, float))
        and score < metric["min_score"]
        or isinstance(metric.get("max_score"), (int, float))
        and score > metric["max_score"]
    ):
        warnings.append(f"{name}: score {score} is outside declared metric bounds")
    uncertainty = score_details.get("uncertainty")
    if uncertainty is not None:
        if not _object(uncertainty):
            _fail("evaluation uncertainty is invalid.")
        interval = uncertainty.get("confidence_interval")
        if interval is not None:
            if not _object(interval):
                _fail("evaluation uncertainty is invalid.")
            lower, upper = interval.get("lower"), interval.get("upper")
            if (
                any(
                    not isinstance(x, (int, float))
                    or isinstance(x, bool)
                    or not math.isfinite(x)
                    for x in (lower, upper)
                )
                or lower > upper
                or score < lower
                or score > upper
            ):
                _fail("evaluation uncertainty is invalid.")
        if any(
            isinstance(x, float) and not math.isfinite(x) for x in uncertainty.values()
        ):
            _fail("evaluation uncertainty is invalid.")
