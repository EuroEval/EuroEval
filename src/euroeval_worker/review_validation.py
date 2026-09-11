"""Validation and report construction for volunteer-result review."""

from __future__ import annotations

import collections.abc as c
import json
import os
import re
import typing as t
from pathlib import Path

from leaderboards.eee_validation import validate_eee_record
from leaderboards.result_identity import (
    ResultIdentity,
    identity_from_eee_record,
    identity_to_path,
    raise_on_collision,
)

from .review_models import JsonObject, ReviewError, ReviewReport, ValidatedRecord
from .review_storage import BucketStore, _digest
from .types import PROTOCOL_VERSION

_VERSION_SUFFIX_RE = re.compile(r"\.dev\d+$")


def _validate_manifest(
    manifest: JsonObject, store: BucketStore, scope_policy: JsonObject
) -> ReviewReport:
    submission_id, issue_number, contributor, language = _manifest_summary(manifest)
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ReviewError("Manifest has an unsupported protocol_version")
    model = _required_object(manifest, "model")
    model_id = _required_string(model, "id")
    revision = _required_string(model, "revision")
    euroeval_version = _required_string(manifest, "euroeval_version")
    _required_string(manifest, "language_group")
    model_type = _required_string(manifest, "model_type")
    if model_type not in {"encoder", "generative"}:
        raise ReviewError("Manifest model_type is unsupported")
    for field in ("worker_version", "image_digest", "created_at"):
        _required_string(manifest, field)
    expected_scope = _required_object(manifest, "expected_scope")
    _required_string(expected_scope, "policy_version")
    _validate_scope_policy(
        manifest=manifest, expected_scope=expected_scope, scope_policy=scope_policy
    )
    expected = _expected_identities(
        value=expected_scope.get("identity_suffixes"), model_id=model_id
    )
    if expected_scope.get("count") != len(expected):
        raise ReviewError("Manifest expected-scope count is inconsistent")
    raw_results = manifest.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        raise ReviewError("Manifest results must be a non-empty list")
    records = tuple(
        _validate_result_entry(
            entry=entry,
            store=store,
            submission_id=submission_id,
            model_id=model_id,
            revision=revision,
            language=language,
            euroeval_version=euroeval_version,
        )
        for entry in raw_results
    )
    actual = tuple(record.identity for record in records)
    _raise_on_identity_collisions(actual)
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ReviewError("Manifest expected and actual canonical identities differ")
    automated = _required_object(manifest, "automated_checks")
    if automated.get("result_count") != len(records):
        raise ReviewError("Manifest automated result_count is inconsistent")
    if automated.get("identities_unique") is not True:
        raise ReviewError("Manifest does not assert unique identities")
    if automated.get("failed_instances") != 0:
        raise ReviewError("Manifest reports failed instances")
    warnings = sorted(
        {warning for record in records for warning in record.warnings}
        | set(_string_list(automated.get("warnings"), "automated warnings"))
    )
    provenance = {
        key: manifest[key]
        for key in (
            "model_type",
            "language_group",
            "worker_version",
            "image_digest",
            "hardware",
            "created_at",
        )
        if key in manifest
    }
    return ReviewReport(
        submission_id=submission_id,
        issue_number=issue_number,
        contributor=contributor,
        model_id=model_id,
        model_revision=revision,
        language=language,
        euroeval_version=euroeval_version,
        provenance=provenance,
        expected_identities=expected,
        records=records,
        warnings=tuple(warnings),
        checks=(
            "manifest protocol and schema",
            "exact UTF-8 SHA256 digests",
            "EEE schema and metadata",
            "canonical identities and expected scope",
            "model, language, version, and failure consistency",
        ),
    )


def _expected_identities(value: object, model_id: str) -> tuple[ResultIdentity, ...]:
    if not isinstance(value, list) or not value:
        raise ReviewError(
            "Manifest expected identity suffixes must be a non-empty list"
        )
    identities: list[ResultIdentity] = []
    for suffix in value:
        parsed = _json_value(suffix, "expected identity suffix")
        if (
            not isinstance(parsed, list)
            or len(parsed) != 3
            or not isinstance(parsed[0], str)
        ):
            raise ReviewError("Expected identity suffix is malformed")
        split = _nullable_bool(parsed[1], "expected validation split")
        shot = _nullable_bool(parsed[2], "expected few-shot value")
        identities.append((model_id, parsed[0], split, shot))
    if len(set(identities)) != len(identities):
        raise ReviewError("Expected identities are not unique")
    return tuple(identities)


def _json_value(value: object, context: str) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ReviewError(f"{context} contains invalid JSON") from error


def _nullable_bool(value: object, context: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise ReviewError(f"{context} must be boolean or null")


def _manifest_summary(manifest: JsonObject) -> tuple[str, int, str, str]:
    submission_id = _required_string(manifest, "submission_id")
    contributor = _required_string(manifest, "verified_contributor")
    language = _required_string(manifest, "language")
    issue = manifest.get("issue_number")
    if isinstance(issue, bool) or not isinstance(issue, int) or issue <= 0:
        raise ReviewError("Manifest issue_number must be a positive integer")
    return submission_id, issue, contributor, language


def _required_string(parent: JsonObject, key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ReviewError(f"{key} must be a non-empty string")
    return value


def _raise_on_identity_collisions(identities: c.Iterable[ResultIdentity]) -> None:
    """Reject distinct identities that sanitise to one canonical path.

    Raises:
        ReviewError:
            If two identities map to the same canonical path.
    """
    seen: list[ResultIdentity] = []
    for identity in identities:
        for previous in seen:
            try:
                raise_on_collision(previous, identity)
            except ValueError as error:
                raise ReviewError(str(error)) from error
        seen.append(identity)


def _required_object(parent: JsonObject, key: str) -> JsonObject:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ReviewError(f"{key} must be an object")
    return value


def _string_list(value: object, context: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ReviewError(f"{context} must be a list of strings")
    return t.cast(list[str], value)


def _validate_result_entry(
    entry: object,
    store: BucketStore,
    submission_id: str,
    model_id: str,
    revision: str,
    language: str,
    euroeval_version: str,
) -> ValidatedRecord:
    if not isinstance(entry, dict):
        raise ReviewError("Manifest result entry must be an object")
    digest = _required_string(entry, "digest")
    path = _required_string(entry, "path")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ReviewError(f"Invalid SHA256 digest in {path}")
    expected_prefix = f"volunteer/submissions/{submission_id}/results/"
    if path != f"{expected_prefix}{digest}.json":
        raise ReviewError(
            f"Staged result path is not canonical under {expected_prefix}"
        )
    content = store.read(store.staging_bucket, path)
    if _digest(content) != digest:
        raise ReviewError(f"Staged bytes do not match SHA256 for {path}")
    record = _load_object(content=content, context=path)
    if record.get("schema_version") not in {"0.2.1", "0.3.0"}:
        raise ReviewError(f"Unsupported EEE schema version in {path}")
    try:
        validate_eee_record(record=record, context=path)
        identity = identity_from_eee_record(record=record)
    except (TypeError, ValueError) as error:
        raise ReviewError(str(error)) from error
    listed_identity = _identity_from_manifest(entry.get("identity"))
    if identity != listed_identity or identity[0] != model_id:
        raise ReviewError(f"Canonical identity mismatch for {path}")
    _validate_record_contract(
        record=record,
        model_id=model_id,
        revision=revision,
        language=language,
        euroeval_version=euroeval_version,
        path=path,
    )
    scores, warnings = _scores(record=record, path=path)
    listed_warnings = _string_list(entry.get("warnings", []), f"warnings in {path}")
    return ValidatedRecord(
        identity=identity,
        digest=digest,
        staged_path=path,
        canonical_path=identity_to_path(identity).as_posix(),
        content=content,
        scores=scores,
        warnings=tuple(sorted(set(warnings) | set(listed_warnings))),
    )


def _identity_from_manifest(value: object) -> ResultIdentity:
    parsed = _json_value(value, "manifest result identity")
    if (
        not isinstance(parsed, list)
        or len(parsed) != 4
        or not all(isinstance(item, str) for item in parsed[:2])
    ):
        raise ReviewError("Manifest result identity is malformed")
    return (
        t.cast(str, parsed[0]),
        t.cast(str, parsed[1]),
        _nullable_bool(parsed[2], "validation split"),
        _nullable_bool(parsed[3], "few-shot value"),
    )


def _load_object(content: bytes, context: str) -> JsonObject:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewError(f"{context} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ReviewError(f"{context} must contain a JSON object")
    return value


def _scores(
    record: JsonObject, path: str
) -> tuple[tuple[tuple[str, float], ...], list[str]]:
    raw = record.get("evaluation_results")
    if not isinstance(raw, list) or not raw:
        raise ReviewError(f"No evaluation results in {path}")
    scores: list[tuple[str, float]] = []
    warnings: list[str] = []
    for result in raw:
        if not isinstance(result, dict):
            raise ReviewError(f"Invalid evaluation result in {path}")
        name = result.get(
            "evaluation_name", result.get("metric_name", result.get("name"))
        )
        details = result.get("score_details")
        metric = result.get("metric_config")
        if (
            not isinstance(name, str)
            or not isinstance(details, dict)
            or not isinstance(metric, dict)
        ):
            raise ReviewError(f"Invalid score schema in {path}")
        source = result.get("source_data")
        library = t.cast(JsonObject, record["eval_library"])
        record_details = library.get("additional_details")
        dataset = (
            record_details.get("dataset") if isinstance(record_details, dict) else None
        )
        source_dataset = (
            source.get("dataset_name", source.get("dataset", source.get("name")))
            if isinstance(source, dict)
            else None
        )
        if not isinstance(source, dict) or source_dataset != dataset:
            raise ReviewError(f"Score source dataset mismatch in {path}")
        if not isinstance(metric.get("lower_is_better"), bool):
            raise ReviewError(f"Invalid metric direction in {path}")
        score = details.get("score")
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise ReviewError(f"Non-numeric score in {path}")
        numeric = float(score)
        if not _finite(numeric):
            raise ReviewError(f"Non-finite score in {path}")
        scores.append((name, numeric))
        minimum = metric.get("min_score")
        maximum = metric.get("max_score")
        if (
            isinstance(minimum, int | float)
            and not isinstance(minimum, bool)
            and numeric < minimum
            or isinstance(maximum, int | float)
            and not isinstance(maximum, bool)
            and numeric > maximum
        ):
            warnings.append(
                f"{name}: score {numeric:g} is outside declared metric bounds"
            )
    return tuple(scores), warnings


def _finite(value: float) -> bool:
    return value != float("inf") and value != float("-inf") and value == value


def _validate_record_contract(
    record: JsonObject,
    model_id: str,
    revision: str,
    language: str,
    euroeval_version: str,
    path: str,
) -> None:
    model = _required_object(record, "model_info")
    aliases: list[str] = []
    for value in (model.get("id"), model.get("name"), model.get("aliases")):
        if isinstance(value, str):
            aliases.append(value)
        elif isinstance(value, list):
            aliases.extend(item for item in value if isinstance(item, str))
    if model_id not in aliases or model.get("revision") not in {None, revision}:
        raise ReviewError(f"Model identity or revision mismatch in {path}")
    library = _required_object(record, "eval_library")
    version = _required_string(library, "version")
    if library.get("name") != "euroeval" or _normalise_version(
        version
    ) != _normalise_version(euroeval_version):
        raise ReviewError(f"EuroEval version mismatch in {path}")
    details = _required_object(library, "additional_details")
    raw_results = _json_value(details.get("raw_results"), f"raw_results in {path}")
    if not isinstance(raw_results, list):
        raise ReviewError(f"raw_results is not a list in {path}")
    languages = _json_string_list(details.get("languages"), f"languages in {path}")
    record_language = details.get("language", record.get("language"))
    if language not in languages or record_language not in {None, language}:
        raise ReviewError(f"Language mismatch in {path}")
    if _contains_failure(record):
        raise ReviewError(f"Failed instances found in {path}")


def _contains_failure(value: object) -> bool:
    if isinstance(value, list):
        return any(_contains_failure(item) for item in value)
    if not isinstance(value, dict):
        return False
    for key, item in value.items():
        if key in {"num_failed_instances", "failed_instances"} and _failure_value(item):
            return True
        if _contains_failure(item):
            return True
    return False


def _failure_value(value: object) -> bool:
    parsed = _json_value(value, "failure field")
    if parsed is None or parsed is False or parsed == 0 or parsed == "":
        return False
    if isinstance(parsed, list | dict):
        return len(parsed) > 0
    return True


def _json_string_list(value: object, context: str) -> list[str]:
    parsed = _json_value(value, context)
    return _string_list(parsed, context)


def _normalise_version(value: str) -> str:
    return _VERSION_SUFFIX_RE.sub("", value)


def _validate_scope_policy(
    manifest: JsonObject, expected_scope: JsonObject, scope_policy: JsonObject
) -> None:
    policies = scope_policy.get("policies")
    if expected_scope.get("policy_version") != scope_policy.get(
        "policy_version"
    ) or not isinstance(policies, list):
        raise ReviewError("Manifest scope policy version is not trusted")
    model_type = manifest.get("model_type")
    language = manifest.get("language")
    version = manifest.get("euroeval_version")
    matching = [
        item
        for item in policies
        if isinstance(item, dict)
        and item.get("model_type") == model_type
        and item.get("language") == language
        and _normalise_version(str(item.get("euroeval_version")))
        == _normalise_version(str(version))
    ]
    if len(matching) != 1:
        raise ReviewError("Manifest scope has no unique trusted policy entry")
    trusted = t.cast(JsonObject, matching[0])
    manifest_language_group = _required_string(manifest, "language_group")
    expected_language_group = _required_string(expected_scope, "language_group")
    trusted_language_group = _required_string(trusted, "language_group")
    if not (
        manifest_language_group == expected_language_group == trusted_language_group
    ):
        raise ReviewError("Manifest language_group differs from trusted policy")
    for field in ("identity_suffixes", "count", "warnings"):
        if expected_scope.get(field) != trusted.get(field):
            raise ReviewError(f"Manifest scope differs from policy field {field}")


def load_scope_policy() -> JsonObject:
    """Load the deployed override or generated volunteer scope policy.

    Returns:
        The trusted scope policy object.

    Raises:
        ReviewError:
            If the configured policy is not valid JSON.
    """
    configured = os.environ.get("VOLUNTEER_SCOPE_POLICY_JSON")
    if configured is not None:
        return _load_object(content=configured.encode("utf-8"), context="scope policy")
    path = Path(__file__).parents[2] / "api" / "worker" / "scope-policy.json"
    if not path.is_file():
        raise ReviewError(f"Generated scope policy is missing: {path}")
    return _load_object(content=path.read_bytes(), context=str(path))
