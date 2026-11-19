"""Pure conflict policy for records sharing a canonical result identity."""

from __future__ import annotations

import hashlib
import json
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum

from .result_identity import ResultIdentity, identity_from_eee_record

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

    identity: ResultIdentity
    current_sha256: str
    candidate_sha256: str
    policy_version: str
    choice: ConflictChoice | None


def _evaluation_time(*, record: dict[str, object]) -> datetime | None:
    found: list[datetime] = []
    for key, value in _walk_values(record=record):
        if _is_evaluation_timestamp(field=key):
            parsed = _parse_timestamp(value=value)
            if parsed is None:
                return None
            found.append(parsed)
    if not found or len(set(found)) != 1:
        return None
    return found[0]


def _is_evaluation_timestamp(*, field: str) -> bool:
    name = field.rsplit(".", 1)[-1].casefold()
    return (
        name in {"evaluation_timestamp", "evaluated_at", "evaluation_time", "timestamp"}
        and ("eval" in field.casefold() or name != "timestamp")
        or name in {"evaluation_timestamp", "evaluated_at", "evaluation_time"}
    )


def _parse_timestamp(*, value: object) -> datetime | None:
    try:
        if isinstance(value, (int, float)):
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


def _walk_values(
    *, record: dict[str, object], prefix: str = ""
) -> t.Iterator[tuple[str, object]]:
    for key, value in record.items():
        field = f"{prefix}.{key}" if prefix else key
        yield field, value
        if isinstance(value, dict):
            yield from _walk_values(record=value, prefix=field)


def _metadata_values(*, record: dict[str, object]) -> frozenset[tuple[str, str]]:
    values = set()
    for key, value in _comparison_record(record=record).items():
        if _field_category(field=key) != "metadata":
            continue
        useful = _useful_metadata(field=key, value=value)
        if useful is not _MISSING:
            values.add((key, _display(value=useful)))
    return frozenset(values)


def _comparison_record(*, record: dict[str, object]) -> dict[str, object]:
    result = dict(record)
    for key in _ENVELOPE_KEYS:
        value = result.get(key)
        if key == "eval_library":
            result.pop(key, None)
        elif isinstance(value, dict):
            result.update(
                {f"{key}.{subkey}": subvalue for subkey, subvalue in value.items()}
            )
            result.pop(key, None)
    for key, value in tuple(result.items()):
        if _is_bookkeeping(field=key):
            result.pop(key)
        elif (
            _field_category(field=key) == "metadata"
            and _useful_metadata(field=key, value=value) is _MISSING
        ):
            result.pop(key)
        elif key.endswith("raw_results"):
            result[key] = _normalise_embedded_json(value=value)
    return result


def _field_category(*, field: str) -> str:
    name = field.rsplit(".", 1)[-1].casefold()
    return (
        "payload"
        if any(token in name for token in ("score", "result", "metric"))
        else "metadata"
    )


def _is_bookkeeping(*, field: str) -> bool:
    name = field.rsplit(".", 1)[-1].casefold()
    return name in {
        "retrieved_timestamp",
        "imported_at",
        "import_timestamp",
        "created_at",
        "updated_at",
        "timestamp",
        "eval_library",
        "library_version",
        "evaluation_timestamp",
        "evaluated_at",
        "evaluation_time",
    } or name.endswith("_transport")


def _normalise_embedded_json(*, value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
    return _canonical(value=parsed)


def _canonical(*, value: object) -> object:
    if isinstance(value, dict):
        return tuple((key, _canonical(value=value[key])) for key in sorted(value))
    if isinstance(value, list):
        return tuple(_canonical(value=item) for item in value)
    return value


def _useful_metadata(*, field: str, value: object) -> object:
    if value is None or value == "" or value == -1 or value == "-1":
        return _MISSING
    if (
        field.endswith("num_model_parameters")
        and isinstance(value, (int, float))
        and value < 0
    ):
        return _MISSING
    if isinstance(value, str) and value.casefold() in {"unknown", "n/a", "none"}:
        return _MISSING
    return value


def _display(*, value: object) -> str:
    if value is _MISSING:
        return "<missing>"
    return json.dumps(value, sort_keys=True, default=str)


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

    Exact bytes should be supplied by storage adapters. When omitted, stable JSON
    bytes of each mapping are used to bind evidence for in-memory callers.

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
    identity = identity_from_eee_record(record=current)
    candidate_identity = identity_from_eee_record(record=candidate)
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
        selection = choice
        reason = ConflictReason.EXPLICIT_CHOICE
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
        if left.get(field, _MISSING) == right.get(field, _MISSING):
            continue
        differences.append(
            FieldDifference(
                field=field,
                current=_display(value=left.get(field, _MISSING)),
                candidate=_display(value=right.get(field, _MISSING)),
                category=_field_category(field=field),
            )
        )
    return tuple(differences)


def _stable_bytes(*, record: dict[str, object]) -> bytes:
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), default=str
    ).encode()


_MISSING = object()
_ENVELOPE_KEYS = {"model_info", "eval_library", "evaluation", "metadata"}
