"""Private scoring of contamination-canary evidence embedded in EEE records."""

# ruff: noqa: DOC201, DOC501

from __future__ import annotations

import collections.abc as c
import hashlib
import json
import logging
import math
import os
import stat
import sys
import typing as t
from collections import defaultdict
from pathlib import Path

from euroeval.canary_evidence import (
    CANARY_GROUP_COUNT,
    CANARY_PRIVATE_DIR_ENV,
    CANARY_RESULT_DATASET,
    CANARY_ROW_COUNT,
    CanaryEvidence,
    evidence_from_dict,
    load_canary_prompts,
    normalise_completion,
)

logger = logging.getLogger(__name__)

CANARY_KEY_ENV = "EUROEVAL_CANARY_KEY"
CANARY_REPORT_PATH_ENV = "EUROEVAL_CANARY_REPORT_PATH"
CANARY_EXACT_RATE_DIFFERENCE = 0.10
CANARY_SIGN_P_VALUE = 0.01
_REPORT_SCHEMA = "contamination-canary-report/v1"
_EXCLUSIONS_SCHEMA = "contamination-canary-exclusions/v1"
_EXCLUSIONS_FILENAME = "leaderboard-exclusions.json"


class _PrivateRecord(t.TypedDict):
    """Private scorer material for one canary row."""

    row_id: str
    group_id: str
    exposed_target: str
    control_target: str


def process_contamination_canaries(
    records: c.Sequence[dict[str, object]],
) -> tuple[
    list[dict[str, object]], list[dict[str, object]], set[str], dict[str, object]
]:
    """Score embedded evidence and ask whether detected models should be excluded.

    Args:
        records:
            Ordinary EEE records, including optional auxiliary canary records.

    Returns:
        Non-canary records, canary records, excluded model IDs, and the report.
    """
    ordinary, canary_records = partition_canary_records(records=records)
    report = score_canary_records(records=canary_records)
    canary_records = _reject_conflicting_immutable_records(
        records=canary_records, report=report
    )
    persisted = load_canary_exclusions()
    excluded = persisted | confirm_canary_exclusions(
        report=report, already_excluded=persisted
    )
    _store_canary_exclusions(excluded)
    return ordinary, canary_records, excluded, report


def partition_canary_records(
    records: c.Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Separate auxiliary canary records from rankable benchmark records."""
    ordinary: list[dict[str, object]] = []
    canaries: list[dict[str, object]] = []
    for record in records:
        target = canaries if is_canary_record(record=record) else ordinary
        target.append(record)
    return ordinary, canaries


def is_canary_record(record: dict[str, object]) -> bool:
    """Return whether an EEE record is an auxiliary contamination-canary result."""
    library = record.get("eval_library")
    if not isinstance(library, dict):
        return False
    details = library.get("additional_details")
    if not isinstance(details, dict):
        return False
    dataset = details.get("dataset")
    return isinstance(dataset, str) and (
        dataset == CANARY_RESULT_DATASET
        or dataset.startswith(f"{CANARY_RESULT_DATASET}-")
    )


def score_canary_records(records: c.Sequence[dict[str, object]]) -> dict[str, object]:
    """Score embedded evidence using private maintainer-only targets and controls."""
    if not records:
        return _base_report(status="missing", models=[])
    try:
        private_dir = _private_directory()
        _validate_private_directory(private_dir)
        key = _load_private_key(_key_path())
        private_records, manifest_hash = _load_private_records(
            private_dir=private_dir, key=key
        )
        prompts = load_canary_prompts(
            cache_dir=Path(
                os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
            )
        )
        prompt_digests = {item.row_id: item.prompt_sha256 for item in prompts}
    except Exception as error:  # noqa: BLE001 - audit setup is non-fatal
        report = _base_report(status="unavailable", models=[])
        report["reason"] = type(error).__name__
        _write_report(report=report)
        return report

    seen: set[str] = set()
    identity_fingerprints: dict[str, str] = {}
    conflicting_identities: set[str] = set()
    outcomes: list[dict[str, object]] = []
    for record in sorted(records, key=_record_timestamp):
        try:
            evidence = _evidence_from_record(record=record)
            canonical = json.dumps(
                evidence.to_dict(), sort_keys=True, separators=(",", ":")
            )
            fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            previous_fingerprint = identity_fingerprints.get(evidence.identity)
            if previous_fingerprint is not None and previous_fingerprint != fingerprint:
                if evidence.identity_kind == "immutable":
                    conflicting_identities.add(evidence.identity)
                    continue
                outcomes = [
                    item
                    for item in outcomes
                    if item.get("evidence_identity") != evidence.identity
                ]
            identity_fingerprints[evidence.identity] = fingerprint
            if evidence.status != "collected":
                outcomes.append(
                    {
                        "model_id": evidence.model_id,
                        "resolved_revision": evidence.resolved_revision,
                        "evidence_identity": evidence.identity,
                        "status": evidence.status,
                        "reason": evidence.reason,
                        "contamination_detected": False,
                    }
                )
            else:
                outcomes.append(
                    _score_one(
                        evidence=evidence,
                        records=private_records,
                        prompt_digests=prompt_digests,
                    )
                )
        except Exception as error:  # noqa: BLE001 - isolate malformed submissions
            outcomes.append(
                {
                    "model_id": _record_model_id(record),
                    "status": "invalid",
                    "reason": type(error).__name__,
                    "contamination_detected": False,
                }
            )
    if conflicting_identities:
        outcomes = [
            item
            for item in outcomes
            if item.get("evidence_identity") not in conflicting_identities
        ]
        outcomes.extend(
            {
                "evidence_identity": identity,
                "status": "invalid",
                "reason": "conflicting_immutable_evidence",
                "contamination_detected": False,
            }
            for identity in sorted(conflicting_identities)
        )
    report = {
        **_base_report(status="scored", models=outcomes),
        "private_manifest_sha256": manifest_hash,
        "decision_policy": {
            "name": "exact-group-sign-test/v1",
            "minimum_exact_rate_difference": CANARY_EXACT_RATE_DIFFERENCE,
            "maximum_p_value": CANARY_SIGN_P_VALUE,
        },
    }
    _write_report(report=report)
    return report


def _reject_conflicting_immutable_records(
    *, records: list[dict[str, object]], report: c.Mapping[str, object]
) -> list[dict[str, object]]:
    """Prevent conflicting immutable evidence from replacing accepted storage."""
    models = report.get("models")
    if not isinstance(models, list):
        return records
    conflicts = {
        item["evidence_identity"]
        for item in models
        if isinstance(item, dict)
        and item.get("reason") == "conflicting_immutable_evidence"
        and isinstance(item.get("evidence_identity"), str)
    }
    if not conflicts:
        return records
    accepted: list[dict[str, object]] = []
    for record in records:
        try:
            if _evidence_from_record(record=record).identity in conflicts:
                continue
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            pass
        accepted.append(record)
    return accepted


def confirm_canary_exclusions(
    report: c.Mapping[str, object], already_excluded: c.Set[str] = frozenset()
) -> set[str]:
    """Ask whether each detected model should be excluded, defaulting to yes."""
    models = report.get("models")
    if not isinstance(models, list):
        return set()
    detected: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in models:
        if (
            isinstance(item, dict)
            and item.get("contamination_detected") is True
            and isinstance(item.get("model_id"), str)
        ):
            detected[t.cast(str, item["model_id"])].append(item)
    excluded: set[str] = set()
    for model_id, outcomes in sorted(detected.items()):
        if model_id in already_excluded:
            excluded.add(model_id)
            continue
        strongest = max(
            outcomes, key=lambda item: _numeric_field(item, "exact_rate_difference")
        )
        logger.warning(
            "Contamination canary detected exposure for %s "
            "(exact difference %.1f%%, p=%s).",
            model_id,
            100 * _numeric_field(strongest, "exact_rate_difference"),
            strongest.get("paired_sign_p_value"),
        )
        if _confirm_exclusion(model_id=model_id):
            excluded.add(model_id)
            logger.warning("Removing %s from generated leaderboards.", model_id)
        else:
            logger.warning("Keeping %s on generated leaderboards.", model_id)
    return excluded


def load_canary_exclusions() -> set[str]:
    """Load durable removals, failing closed once private state is configured."""
    configured = os.getenv(CANARY_PRIVATE_DIR_ENV)
    if not configured:
        return set()
    path = Path(configured).expanduser() / _EXCLUSIONS_FILENAME
    if not path.exists():
        return set()
    try:
        _validate_private_file(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        models = value.get("models") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != _EXCLUSIONS_SCHEMA
            or not isinstance(models, list)
            or not all(isinstance(item, str) and item for item in models)
        ):
            raise ValueError("private canary exclusions are malformed")
        return set(models)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Cannot safely generate leaderboards: private canary exclusions are "
            "unreadable or invalid."
        ) from error


def _store_canary_exclusions(models: set[str]) -> None:
    """Persist confirmed removals outside the repository with mode 0600."""
    if not models:
        return
    try:
        path = _private_directory() / _EXCLUSIONS_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        content = json.dumps(
            {"schema_version": _EXCLUSIONS_SCHEMA, "models": sorted(models)},
            indent=2,
            sort_keys=True,
        )
        _atomic_private_write(path=path, content=f"{content}\n")
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "Cannot safely generate leaderboards: confirmed canary exclusions could "
            "not be persisted."
        ) from error


def _confirm_exclusion(*, model_id: str) -> bool:
    """Return an interactive exclusion decision, with safe yes defaults."""
    if not sys.stdin.isatty():
        logger.warning(
            "No interactive terminal is available; defaulting to removal for %s.",
            model_id,
        )
        return True
    try:
        answer = input(
            f"Remove {model_id} from the leaderboards due to the canary result? [Y/n] "
        )
    except EOFError:
        return True
    return answer.strip().lower() not in {"n", "no"}


def _record_timestamp(record: dict[str, object]) -> str:
    """Return an EEE timestamp suitable for deterministic snapshot ordering."""
    value = record.get("evaluation_timestamp")
    return value if isinstance(value, str) else ""


def _record_model_id(record: dict[str, object]) -> str:
    """Return a best-effort model identifier for an invalid record report."""
    model_info = record.get("model_info")
    if isinstance(model_info, dict):
        value = model_info.get("id") or model_info.get("name")
        if isinstance(value, str):
            return value
    return "unknown"


def _evidence_from_record(*, record: dict[str, object]) -> CanaryEvidence:
    """Extract and bind strict canary evidence from one EEE result."""
    library = record.get("eval_library")
    model_info = record.get("model_info")
    if not isinstance(library, dict) or not isinstance(model_info, dict):
        raise ValueError("canary result has an invalid EEE envelope")
    details = library.get("additional_details")
    if not isinstance(details, dict):
        raise ValueError("canary result has no additional details")
    raw = details.get("contamination_canary_evidence")
    if isinstance(raw, str):
        raw = json.loads(raw)
    evidence = evidence_from_dict(raw)
    model_name = model_info.get("id") or model_info.get("name")
    if not isinstance(model_name, str) or not (
        model_name == evidence.model_id
        or model_name.startswith(f"{evidence.model_id}@")
        or model_name.startswith(f"{evidence.model_id}#")
    ):
        raise ValueError("canary evidence model does not match its EEE record")
    return evidence


def _score_one(
    *,
    evidence: CanaryEvidence,
    records: dict[str, _PrivateRecord],
    prompt_digests: dict[str, str],
) -> dict[str, object]:
    """Score one complete canary observation set over 32 independent groups."""
    if len(evidence.observations) != CANARY_ROW_COUNT:
        raise ValueError("scoreable canary evidence must contain 256 rows")
    grouped: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    seen: set[str] = set()
    for observation in evidence.observations:
        record = records.get(observation.row_id)
        if (
            record is None
            or observation.row_id in seen
            or prompt_digests.get(observation.row_id) != observation.prompt_sha256
        ):
            raise ValueError("evidence rows do not match the frozen canary corpus")
        seen.add(observation.row_id)
        completion = observation.normalised_completion
        exposed = normalise_completion(record["exposed_target"])
        control = normalise_completion(record["control_target"])
        first = completion.split(maxsplit=1)[0] if completion else ""
        grouped[record["group_id"]].append(
            (
                int(completion == exposed),
                int(completion == control),
                int(first == exposed.split(maxsplit=1)[0]),
                int(first == control.split(maxsplit=1)[0]),
            )
        )
    if len(seen) != CANARY_ROW_COUNT or len(grouped) != CANARY_GROUP_COUNT:
        raise ValueError("evidence does not contain 32 complete private canary groups")
    group_values: list[tuple[float, float, float, float]] = []
    for values in grouped.values():
        if len(values) != CANARY_ROW_COUNT // CANARY_GROUP_COUNT:
            raise ValueError("private canary groups must contain eight rows")
        group_values.append(
            (
                sum(item[0] for item in values) / len(values),
                sum(item[1] for item in values) / len(values),
                sum(item[2] for item in values) / len(values),
                sum(item[3] for item in values) / len(values),
            )
        )
    exposed_exact = _mean([item[0] for item in group_values])
    control_exact = _mean([item[1] for item in group_values])
    exact_difference = exposed_exact - control_exact
    exposed_prefix = _mean([item[2] for item in group_values])
    control_prefix = _mean([item[3] for item in group_values])
    positive = sum(item[0] > item[1] for item in group_values)
    negative = sum(item[0] < item[1] for item in group_values)
    p_value = _two_sided_sign_p_value(positive=positive, negative=negative)
    detected = (
        exposed_exact >= CANARY_EXACT_RATE_DIFFERENCE
        and exact_difference >= CANARY_EXACT_RATE_DIFFERENCE
        and p_value <= CANARY_SIGN_P_VALUE
    )
    return {
        "model_id": evidence.model_id,
        "requested_revision": evidence.requested_revision,
        "resolved_revision": evidence.resolved_revision,
        "identity_kind": evidence.identity_kind,
        "backend": evidence.backend,
        "evidence_identity": evidence.identity,
        "status": "scored",
        "contamination_detected": detected,
        "group_count": CANARY_GROUP_COUNT,
        "exact_exposed_rate": exposed_exact,
        "exact_control_rate": control_exact,
        "exact_rate_difference": exact_difference,
        "prefix_exposed_rate": exposed_prefix,
        "prefix_control_rate": control_prefix,
        "paired_sign_p_value": p_value,
    }


def _load_private_records(
    *, private_dir: Path, key: bytes
) -> tuple[dict[str, _PrivateRecord], str]:
    """Load and validate private scoring records and provenance."""
    manifest_path = private_dir / "canary-manifest.json"
    records_path = private_dir / "canary-records.jsonl"
    _validate_private_file(manifest_path)
    _validate_private_file(records_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("private canary manifest is invalid")
    if (
        manifest.get("row_count") != CANARY_ROW_COUNT
        or manifest.get("group_count") != CANARY_GROUP_COUNT
        or manifest.get("key_sha256") != hashlib.sha256(key).hexdigest()
        or not isinstance(manifest.get("hash_version"), int)
    ):
        raise ValueError("private canary manifest provenance is invalid")
    records: dict[str, _PrivateRecord] = {}
    groups: set[str] = set()
    raw_records: list[dict[str, object]] = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("private canary record is invalid")
        selected: _PrivateRecord = {
            "row_id": _record_string(value, "row_id"),
            "group_id": _record_string(value, "group_id"),
            "exposed_target": _record_string(value, "exposed_target"),
            "control_target": _record_string(value, "control_target"),
        }
        if selected["row_id"] in records:
            raise ValueError("private canary record IDs are not unique")
        raw_records.append(value)
        records[selected["row_id"]] = selected
        groups.add(selected["group_id"])
    if len(records) != CANARY_ROW_COUNT or len(groups) != CANARY_GROUP_COUNT:
        raise ValueError("private canary records are incomplete")
    canonical = json.dumps(
        {"hash_version": manifest["hash_version"], "records": raw_records},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if manifest.get("canary_records_sha256") != hashlib.sha256(canonical).hexdigest():
        raise ValueError("private canary record hash is invalid")
    return records, hashlib.sha256(manifest_bytes).hexdigest()


def _base_report(*, status: str, models: list[dict[str, object]]) -> dict[str, object]:
    """Build a private checker report."""
    return {"schema_version": _REPORT_SCHEMA, "status": status, "models": models}


def _write_report(*, report: dict[str, object]) -> None:
    """Write the optional private report without affecting result processing."""
    configured = os.getenv(CANARY_REPORT_PATH_ENV)
    if not configured:
        return
    path = Path(configured).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _atomic_private_write(
            path=path, content=json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    except OSError:
        logger.warning("Could not write the private contamination-canary report.")


def _private_directory() -> Path:
    """Return the configured directory containing private scoring records."""
    configured = os.getenv(CANARY_PRIVATE_DIR_ENV)
    if not configured:
        raise ValueError(f"{CANARY_PRIVATE_DIR_ENV} is required")
    return Path(configured).expanduser()


def _key_path() -> Path:
    """Return the configured private key path."""
    configured = os.getenv(CANARY_KEY_ENV)
    if not configured:
        raise ValueError(f"{CANARY_KEY_ENV} is required")
    return Path(configured).expanduser()


def _load_private_key(path: Path) -> bytes:
    """Load a 32-byte owner-only key."""
    _validate_private_file(path)
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError("private canary key must contain exactly 32 bytes")
    return key


def _validate_private_directory(path: Path) -> None:
    """Require an existing owner-only private directory."""
    if not path.is_dir() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("private canary directory must exist with mode 0700")


def _validate_private_file(path: Path) -> None:
    """Require an existing owner-only private file."""
    if not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError(f"private canary file must exist with mode 0600: {path}")


def _record_string(value: dict[str, object], field: str) -> str:
    """Read one required private record string."""
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise ValueError(f"private canary record has invalid {field}")
    return item


def _numeric_field(value: dict[str, object], field: str) -> float:
    """Return a report number or zero for malformed internal data."""
    item = value.get(field)
    return float(item) if isinstance(item, int | float) else 0.0


def _mean(values: c.Sequence[float]) -> float:
    """Return the arithmetic mean of non-empty values."""
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _two_sided_sign_p_value(*, positive: int, negative: int) -> float:
    """Return the exact two-sided sign-test p-value, ignoring ties."""
    total = positive + negative
    if total == 0:
        return 1.0
    tail = min(positive, negative)
    probability = sum(math.comb(total, index) for index in range(tail + 1)) / (2**total)
    return min(1.0, 2 * probability)


def _atomic_private_write(*, path: Path, content: str) -> None:
    """Atomically write an owner-only private file."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if temporary.exists():
            temporary.unlink()
