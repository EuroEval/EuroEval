"""Offline, report-only checking of persisted contamination-canary evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import typing as t
from collections import defaultdict
from pathlib import Path

from huggingface_hub import BucketFile, HfApi

from euroeval.canary_evidence import (
    CANARY_CHECK_MODE_ENV,
    CANARY_CORPUS_SHA256,
    CANARY_EVIDENCE_PATH_ENV,
    CANARY_GROUP_COUNT,
    CANARY_PRIVATE_DIR_ENV,
    CANARY_ROW_COUNT,
    CanaryEvidence,
    evidence_from_dict,
    load_evidence_jsonl,
    normalise_completion,
)

CANARY_KEY_ENV = "EUROEVAL_CANARY_KEY"
CANARY_REPORT_PATH_ENV = "EUROEVAL_CANARY_REPORT_PATH"
CANARY_ROLLOUT_ENV = "EUROEVAL_CANARY_ROLLOUT_UTC"
CANARY_EVIDENCE_BUCKET_ENV = "EUROEVAL_CANARY_EVIDENCE_BUCKET"
_REPORT_SCHEMA = "contamination-canary-report/v1"


class _PrivateRecord(t.TypedDict):
    row_id: str
    group_id: str
    exposed_target: str
    control_target: str


def run_contamination_canary_check() -> dict[str, object]:
    """Run the offline checker from environment configuration.

    The function never imports model-loading code and never raises for disabled mode.
    In report-only mode configuration or evidence errors are represented in the private
    report and returned to the caller; they must not affect leaderboard generation.

    Returns:
        A private report-only status and any per-model evidence summary.
    """
    mode = os.getenv(CANARY_CHECK_MODE_ENV, "disabled")
    if mode == "disabled":
        return {"schema_version": _REPORT_SCHEMA, "mode": mode, "status": "disabled"}
    if mode != "report-only":
        return _failure_report(reason="invalid_check_mode", mode=mode)
    try:
        private_dir = _private_directory()
        _validate_private_directory(private_dir)
        key_path = _key_path()
        key = _load_private_key(key_path)
        records, manifest_hash = _load_private_records(private_dir=private_dir, key=key)
        evidence_path = _evidence_path(private_dir=private_dir)
        _sync_evidence_bucket(evidence_path=evidence_path)
        if not evidence_path.exists():
            report = _base_report(
                mode=mode,
                status="missing",
                manifest_hash=manifest_hash,
                evidence_path=evidence_path,
            )
            _write_report(private_dir=private_dir, report=report)
            return report
        evidence = load_evidence_jsonl(evidence_path)
        report = _score_all(
            evidence=evidence,
            records=records,
            manifest_hash=manifest_hash,
            evidence_path=evidence_path,
        )
        _write_report(private_dir=private_dir, report=report)
        return report
    except Exception as error:  # noqa: BLE001 - checker is deliberately isolated
        report = _failure_report(reason=type(error).__name__, mode=mode)
        configured = os.getenv(CANARY_PRIVATE_DIR_ENV)
        if configured:
            private_dir = Path(configured).expanduser()
            if private_dir.is_dir():
                try:
                    _write_report(private_dir=private_dir, report=report)
                except OSError:
                    pass
        return report


def _score_all(
    *,
    evidence: t.Sequence[CanaryEvidence],
    records: dict[str, _PrivateRecord],
    manifest_hash: str,
    evidence_path: Path,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for item in evidence:
        if item.status != "collected":
            results.append(
                {
                    "model_id": item.model_id,
                    "resolved_revision": item.resolved_revision,
                    "evidence_identity": item.identity,
                    "status": item.status,
                    "reason": item.reason,
                }
            )
            continue
        results.append(_score_one(evidence=item, records=records))
    return {
        **_base_report(
            mode="report-only",
            status="scored",
            manifest_hash=manifest_hash,
            evidence_path=evidence_path,
        ),
        "decision_policy": "unvalidated_report_only",
        "rollout_utc": os.getenv(CANARY_ROLLOUT_ENV),
        "models": results,
    }


def _score_one(
    *, evidence: CanaryEvidence, records: dict[str, _PrivateRecord]
) -> dict[str, object]:
    if len(evidence.observations) != CANARY_ROW_COUNT:
        raise ValueError("scoreable canary evidence must contain 256 rows")
    by_group: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    seen: set[str] = set()
    for observation in evidence.observations:
        record = records.get(observation.row_id)
        if record is None or observation.row_id in seen:
            raise ValueError("evidence row IDs do not match private canary records")
        seen.add(observation.row_id)
        completion = observation.normalised_completion
        exposed = normalise_completion(record["exposed_target"])
        control = normalise_completion(record["control_target"])
        first = completion.split(maxsplit=1)[0] if completion else ""
        by_group[record["group_id"]].append(
            (
                int(completion == exposed),
                int(completion == control),
                int(first == exposed.split(maxsplit=1)[0]),
                int(first == control.split(maxsplit=1)[0]),
            )
        )
    if len(seen) != CANARY_ROW_COUNT or len(by_group) != CANARY_GROUP_COUNT:
        raise ValueError("evidence does not contain 32 complete private canary groups")
    group_values: list[tuple[float, float, float, float]] = []
    for values in by_group.values():
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
    exposed_prefix = _mean([item[2] for item in group_values])
    control_prefix = _mean([item[3] for item in group_values])
    positive = sum(item[0] > item[1] for item in group_values)
    negative = sum(item[0] < item[1] for item in group_values)
    return {
        "model_id": evidence.model_id,
        "requested_revision": evidence.requested_revision,
        "resolved_revision": evidence.resolved_revision,
        "identity_kind": evidence.identity_kind,
        "backend": evidence.backend,
        "evidence_identity": evidence.identity,
        "status": "scored",
        "decision": "unvalidated_report_only",
        "group_count": CANARY_GROUP_COUNT,
        "exact_exposed_rate": exposed_exact,
        "exact_control_rate": control_exact,
        "exact_rate_difference": exposed_exact - control_exact,
        "prefix_exposed_rate": exposed_prefix,
        "prefix_control_rate": control_prefix,
        "paired_sign_p_value": _two_sided_sign_p_value(
            positive=positive, negative=negative
        ),
    }


def _load_private_records(
    *, private_dir: Path, key: bytes
) -> tuple[dict[str, _PrivateRecord], str]:
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
    canonical_records = json.dumps(
        {"hash_version": manifest["hash_version"], "records": raw_records},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if (
        manifest.get("canary_records_sha256")
        != hashlib.sha256(canonical_records).hexdigest()
    ):
        raise ValueError("private canary record hash is invalid")
    return records, hashlib.sha256(manifest_bytes).hexdigest()


def _base_report(
    *, mode: str, status: str, manifest_hash: str, evidence_path: Path
) -> dict[str, object]:
    return {
        "schema_version": _REPORT_SCHEMA,
        "mode": mode,
        "status": status,
        "corpus_sha256": CANARY_CORPUS_SHA256,
        "private_manifest_sha256": manifest_hash,
        "evidence_file_sha256": (
            hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            if evidence_path.exists()
            else None
        ),
    }


def _failure_report(*, reason: str, mode: str) -> dict[str, object]:
    return {
        "schema_version": _REPORT_SCHEMA,
        "mode": mode,
        "status": "invalid",
        "reason": reason,
        "decision_policy": "unvalidated_report_only",
    }


def _write_report(*, private_dir: Path, report: dict[str, object]) -> None:
    configured = os.getenv(CANARY_REPORT_PATH_ENV)
    path = (
        Path(configured).expanduser()
        if configured
        else private_dir / "production-report.json"
    )
    if not path.expanduser().resolve().is_relative_to(private_dir.resolve()):
        raise ValueError("canary report must remain inside the private directory")
    _atomic_private_write(
        path=path, content=json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


def _private_directory() -> Path:
    configured = os.getenv(CANARY_PRIVATE_DIR_ENV)
    if not configured:
        raise ValueError(f"{CANARY_PRIVATE_DIR_ENV} is required in report-only mode")
    return Path(configured).expanduser()


def _evidence_path(*, private_dir: Path) -> Path:
    configured = os.getenv(CANARY_EVIDENCE_PATH_ENV)
    return (
        Path(configured).expanduser()
        if configured
        else private_dir / "production-evidence.jsonl"
    )


def _sync_evidence_bucket(*, evidence_path: Path) -> None:
    bucket = os.getenv(CANARY_EVIDENCE_BUCKET_ENV)
    if not bucket:
        return
    token = os.getenv("HF_TOKEN")
    directory = evidence_path.parent / "evidence-files"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    api = HfApi()
    downloads: list[tuple[str | BucketFile, str | Path]] = []
    local_files: list[Path] = []
    for entry in api.list_bucket_tree(bucket_id=bucket, recursive=True, token=token):
        if not isinstance(entry, BucketFile) or not entry.path.endswith(".json"):
            continue
        local = directory / entry.path
        local.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        local_files.append(local)
        if not local.exists():
            downloads.append((entry, local))
    if downloads:
        api.download_bucket_files(bucket_id=bucket, files=downloads, token=token)
    evidence_by_identity: dict[str, CanaryEvidence] = {}
    for path in sorted(local_files):
        os.chmod(path, 0o600)
        value = json.loads(path.read_text(encoding="utf-8"))
        item = evidence_from_dict(value)
        existing = evidence_by_identity.get(item.identity)
        if existing is not None and existing.to_dict() != item.to_dict():
            raise ValueError("private evidence bucket contains conflicting identities")
        evidence_by_identity[item.identity] = item
    if evidence_by_identity:
        content = "".join(
            json.dumps(item.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
            for item in sorted(
                evidence_by_identity.values(), key=lambda value: value.identity
            )
        )
        _atomic_private_write(path=evidence_path, content=content)


def _key_path() -> Path:
    configured = os.getenv(CANARY_KEY_ENV)
    if not configured:
        raise ValueError(f"{CANARY_KEY_ENV} is required in report-only mode")
    return Path(configured).expanduser()


def _load_private_key(path: Path) -> bytes:
    _validate_private_file(path)
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError("canary key must contain exactly 32 bytes")
    return key


def _validate_private_directory(path: Path) -> None:
    if not path.is_dir() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError("canary private directory must have mode 0700")


def _validate_private_file(path: Path) -> None:
    if not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError("canary private files must have mode 0600")


def _record_string(value: dict[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"private canary field {key!r} is invalid")
    return item


def _mean(values: t.Sequence[float]) -> float:
    return sum(values) / len(values)


def _two_sided_sign_p_value(*, positive: int, negative: int) -> float:
    total = positive + negative
    if total == 0:
        return 1.0
    lower = min(positive, negative)
    tail = sum(math.comb(total, index) for index in range(lower + 1)) / 2**total
    return min(1.0, 2 * tail)


def _atomic_private_write(*, path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        temporary.replace(path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)
