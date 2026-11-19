"""Pure conflict policy for records sharing a canonical result identity."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum

from . import result_identity

POLICY_VERSION = "1"


class ConflictChoice(StrEnum):
    """An explicit source choice for an unresolved result conflict."""

    STORED = "stored"
    INCOMING = "incoming"


class ConflictReason(StrEnum):
    """Why the resolver did or did not select a record."""

    IDENTICAL = "identical"
    STORED_METADATA_SUPERSET = "stored_metadata_superset"
    INCOMING_METADATA_SUPERSET = "incoming_metadata_superset"
    NEWER_EVALUATION = "newer_evaluation"
    UNRESOLVED_METADATA = "unresolved_metadata"
    UNRESOLVED_TIMESTAMP = "unresolved_timestamp"
    UNRESOLVED_DIFFERENCE = "unresolved_difference"
    EXPLICIT_CHOICE = "explicit_choice"


@dataclass(frozen=True)
class DecisionEvidence:
    """Immutable evidence bound to both exact record byte sequences."""

    identity: result_identity.ResultIdentity
    current_sha256: str
    candidate_sha256: str
    policy_version: str
    choice: ConflictChoice | None


def _evaluation_time(*, record: dict[str, object]) -> datetime | None:
    flat: dict[str, object] = {}
    _flatten(record=record, prefix="", output=flat)
    evidence: list[datetime] = []
    for field in _EVALUATION_TIME_FIELDS:
        if field not in flat:
            continue
        value = flat[field]
        parsed = _parse_timestamp(value=value, epoch=field in _EPOCH_FIELDS)
        if parsed is None:
            return None
        evidence.append(parsed)
    if not evidence or len(set(evidence)) != 1:
        return None
    return evidence[0]


def _flatten(
    *, record: dict[str, object], prefix: str, output: dict[str, object]
) -> None:
    for key, value in record.items():
        field = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and not field.endswith("raw_results"):
            _flatten(record=value, prefix=field, output=output)
        else:
            output[field] = value


def _parse_timestamp(*, value: object, epoch: bool) -> datetime | None:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            if not epoch:
                return None
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if not isinstance(value, str) or not value.strip():
            return None
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return (
            parsed.replace(tzinfo=timezone.utc)
            if parsed.tzinfo is None
            else parsed.astimezone(timezone.utc)
        )
    except (OverflowError, OSError, ValueError):
        return None


def _is_known_metadata(*, field: str) -> bool:
    lowered = field.casefold()
    if lowered.startswith("model_info."):
        return True
    if lowered.startswith("eval_library.additional_details."):
        return lowered.rsplit(".", 1)[-1] in _KNOWN_METADATA
    return "." not in field and lowered in _KNOWN_METADATA


def _metadata_values(*, record: dict[str, object]) -> frozenset[tuple[str, str]]:
    values = set()
    for field, value in _comparison_record(record=record).items():
        if _field_category(field=field) != "metadata":
            continue
        normalised = _normalise_known(field=field, value=value)
        if _useful_metadata(field=field, value=normalised):
            values.add((field, _display(value=normalised)))
    return frozenset(values)


def _comparison_record(*, record: dict[str, object]) -> dict[str, object]:
    flat: dict[str, object] = {}
    _flatten(record=record, prefix="", output=flat)
    result: dict[str, object] = {}
    for field, value in flat.items():
        if _is_ignored_alias(field=field):
            continue
        if field.endswith("raw_results"):
            value = _normalise_embedded_json(value=value)
        result[field] = value
    return result


def _is_ignored_alias(*, field: str) -> bool:
    return (
        field
        in {
            "id",
            "name",
            "eval_id",
            "eval_timestamp",
            "evaluation_timestamp",
            "evaluated_at",
        }
        or field
        in {
            "retrieved_timestamp",
            "imported_at",
            "import_timestamp",
            "created_at",
            "updated_at",
            "eval_library.retrieved_timestamp",
            "eval_library.version",
        }
        or field.endswith("_transport")
    )


def _normalise_embedded_json(*, value: object) -> object:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return _canonical(value=value)


def _canonical(*, value: object) -> object:
    if isinstance(value, dict):
        return tuple((key, _canonical(value=value[key])) for key in sorted(value))
    if isinstance(value, list):
        return tuple(_canonical(value=item) for item in value)
    return value


def _display(*, value: object) -> str:
    if value is _MISSING:
        return "<missing>"
    return json.dumps(value, sort_keys=True, default=str)


def _field_category(*, field: str) -> str:
    lowered = field.casefold()
    if lowered.split(".", 1)[0] in _PAYLOAD_ROOTS or lowered.startswith(
        "eval_library.additional_details.raw_results"
    ):
        return "payload"
    return "metadata"


def _normalise_known(*, field: str, value: object) -> object:
    if value is _MISSING:
        return value
    name = field.rsplit(".", 1)[-1].casefold()
    if name in _BOOLEAN_METADATA and isinstance(value, str):
        lowered = value.casefold()
        if lowered in {"true", "false"}:
            return lowered == "true"
    if name in _NUMERIC_METADATA:
        if isinstance(value, str) and re.fullmatch(r"-?\d+(?:\.\d+)?", value.strip()):
            number = float(value) if "." in value else int(value)
            return number
    return value


def _useful_metadata(*, field: str, value: object) -> bool:
    if value is _MISSING:
        return False
    name = field.rsplit(".", 1)[-1].casefold()
    if value is None or value == "":
        return False
    if name in _NUMERIC_METADATA:
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            if not math.isfinite(value) or value <= 0 or not float(value).is_integer():
                return False
            if name == "max_sequence_length" and value > 1_000_000_000:
                return False
        else:
            return False
    if name in _BOOLEAN_METADATA and isinstance(value, str):
        return value.casefold() in {"true", "false"}
    if name in _KNOWN_METADATA and isinstance(value, str):
        return value.casefold() not in {"unknown", "n/a", "none", "null"}
    return True


@dataclass(frozen=True)
class FieldDifference:
    """A structured difference between corresponding record fields."""

    field: str
    current: str
    candidate: str
    category: str


@dataclass(frozen=True)
class ConflictDecision:
    """Resolver output; a missing selection requires owner choice."""

    selected: ConflictChoice | None
    reason: ConflictReason
    differences: tuple[FieldDifference, ...]
    evidence: DecisionEvidence

    @property
    def needs_choice(self) -> bool:
        """Whether a human choice is required."""
        return self.selected is None


def resolve_result_conflict(
    *,
    current: dict[str, object],
    candidate: dict[str, object],
    current_bytes: bytes | None = None,
    candidate_bytes: bytes | None = None,
    choice: ConflictChoice | None = None,
) -> ConflictDecision:
    """Resolve two records with matching identities without mutating either.

    Args:
        current: The currently stored EEE record.
        candidate: The incoming EEE record.
        current_bytes (optional): Exact stored bytes. Defaults to JSON encoding.
        candidate_bytes (optional): Exact incoming bytes. Defaults to JSON encoding.
        choice (optional): Explicit stored-or-incoming choice for a conflict.

    Returns:
        The automatic or explicitly selected decision and immutable evidence.

    Raises:
        ValueError: If record identities differ or cannot be extracted.
    """
    identity = result_identity.identity_from_eee_record(record=current)
    candidate_identity = result_identity.identity_from_eee_record(record=candidate)
    if identity != candidate_identity:
        raise ValueError(
            f"Records have different identities: {identity} vs {candidate_identity}"
        )
    if current_bytes is None:
        current_bytes = _stable_bytes(record=current)
    if candidate_bytes is None:
        candidate_bytes = _stable_bytes(record=candidate)
    differences = _record_differences(current=current, candidate=candidate)
    selection, reason = _automatic_selection(
        current=current, candidate=candidate, differences=differences
    )
    if choice is not None:
        selection, reason = choice, ConflictReason.EXPLICIT_CHOICE
    return ConflictDecision(
        selected=selection,
        reason=reason,
        differences=differences,
        evidence=DecisionEvidence(
            identity=identity,
            current_sha256=hashlib.sha256(current_bytes).hexdigest(),
            candidate_sha256=hashlib.sha256(candidate_bytes).hexdigest(),
            policy_version=POLICY_VERSION,
            choice=selection,
        ),
    )


def _automatic_selection(
    *,
    current: dict[str, object],
    candidate: dict[str, object],
    differences: tuple[FieldDifference, ...],
) -> tuple[ConflictChoice | None, ConflictReason]:
    if not differences:
        return ConflictChoice.STORED, ConflictReason.IDENTICAL
    metadata = tuple(item for item in differences if item.category == "metadata")
    payload = tuple(item for item in differences if item.category == "payload")
    if metadata and not payload:
        if any(not _is_known_metadata(field=item.field) for item in metadata):
            return None, ConflictReason.UNRESOLVED_METADATA
        current_values = _metadata_values(record=current)
        candidate_values = _metadata_values(record=candidate)
        if current_values > candidate_values:
            return ConflictChoice.STORED, ConflictReason.STORED_METADATA_SUPERSET
        if candidate_values > current_values:
            return ConflictChoice.INCOMING, ConflictReason.INCOMING_METADATA_SUPERSET
        return None, ConflictReason.UNRESOLVED_METADATA
    if payload and not metadata:
        times = (_evaluation_time(record=current), _evaluation_time(record=candidate))
        if times[0] is not None and times[1] is not None and times[0] != times[1]:
            selected = (
                ConflictChoice.STORED
                if times[0] > times[1]
                else ConflictChoice.INCOMING
            )
            return selected, ConflictReason.NEWER_EVALUATION
        return None, ConflictReason.UNRESOLVED_TIMESTAMP
    return None, ConflictReason.UNRESOLVED_DIFFERENCE


def _record_differences(
    *, current: dict[str, object], candidate: dict[str, object]
) -> tuple[FieldDifference, ...]:
    left = _comparison_record(record=current)
    right = _comparison_record(record=candidate)
    differences = []
    for field in sorted(left.keys() | right.keys()):
        left_value = left.get(field, _MISSING)
        right_value = right.get(field, _MISSING)
        left_value = _normalise_known(field=field, value=left_value)
        right_value = _normalise_known(field=field, value=right_value)
        if left_value == right_value:
            continue
        differences.append(
            FieldDifference(
                field=field,
                current=_display(value=left_value),
                candidate=_display(value=right_value),
                category=_field_category(field=field),
            )
        )
    return tuple(differences)


def _stable_bytes(*, record: dict[str, object]) -> bytes:
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), default=str
    ).encode()


_MISSING = object()
_PAYLOAD_ROOTS = {
    "evaluation_results",
    "score",
    "scores",
    "results",
    "raw_results",
    "metrics",
}
_NUMERIC_METADATA = {
    "num_model_parameters",
    "model_parameters",
    "parameter_count",
    "max_sequence_length",
    "vocabulary_size",
}
_BOOLEAN_METADATA = {
    "open",
    "commercially_licensed",
    "trained_from_scratch",
    "few_shot",
    "validation_split",
    "merge",
    "generative",
}
_KNOWN_METADATA = (
    _NUMERIC_METADATA
    | _BOOLEAN_METADATA
    | {"release_date", "model_url", "url", "generative_type"}
)
_EVALUATION_TIME_FIELDS = {"eval_timestamp", "evaluation_timestamp", "evaluated_at"}
_EPOCH_FIELDS = {"eval_timestamp", "evaluation_timestamp"}
