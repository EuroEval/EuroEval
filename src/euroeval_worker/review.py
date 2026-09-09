"""Maintainer-side validation and promotion of staged volunteer results."""

from __future__ import annotations

import collections.abc as c
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
import typing as t
import urllib.request
from pathlib import Path

from leaderboards.constants import HF_RESULTS_BUCKET
from leaderboards.eee_validation import validate_eee_record
from leaderboards.result_identity import (
    ResultIdentity,
    identity_from_eee_record,
    identity_to_path,
    raise_on_collision,
)

from .types import PROTOCOL_VERSION

_DECISION_PREFIX = "volunteer/decisions"
_MANIFEST_PREFIX = "volunteer/manifests"
_VERSION_SUFFIX_RE = re.compile(r"\.dev\d+$")
JsonObject: t.TypeAlias = dict[str, object]
BrokerPromoter: t.TypeAlias = t.Callable[
    [int, str, str, str, list[dict[str, str]]], None
]
BrokerReservation: t.TypeAlias = t.Callable[[int, str, str, list[dict[str, str]]], str]


class BucketInfo(t.Protocol):
    """Bucket visibility metadata used by the reviewer."""

    private: bool


class BucketEntry(t.Protocol):
    """Bucket file metadata used for exact verification."""

    type: str
    path: str
    size: int
    xet_hash: str


class BucketApi(t.Protocol):
    """HfApi bucket operations required by the reviewer."""

    def bucket_info(self, bucket_id: str, *, token: str) -> BucketInfo:
        """Return bucket visibility metadata."""
        ...

    def list_bucket_tree(
        self, bucket_id: str, prefix: str, *, recursive: bool, token: str
    ) -> c.Iterable[BucketEntry]:
        """List files beneath a bucket prefix."""
        ...

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[str, Path]],
        *,
        raise_on_missing_files: bool,
        token: str,
    ) -> None:
        """Download selected bucket files."""
        ...

    def get_bucket_paths_info(
        self, bucket_id: str, paths: list[str], *, token: str
    ) -> c.Iterable[BucketEntry]:
        """Return metadata for selected paths."""
        ...

    def batch_bucket_files(
        self, bucket_id: str, *, add: list[tuple[bytes, str]], token: str
    ) -> object:
        """Upload selected bucket files."""
        ...


class ReviewError(RuntimeError):
    """Raised when staged evidence cannot be safely promoted."""


@dataclasses.dataclass(frozen=True)
class ValidatedRecord:
    """One byte-verified, schema-verified staged result."""

    identity: ResultIdentity
    digest: str
    staged_path: str
    canonical_path: str
    content: bytes
    scores: tuple[tuple[str, float], ...]
    warnings: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ReviewReport:
    """Independently validated evidence for one submission."""

    submission_id: str
    issue_number: int
    contributor: str
    model_id: str
    model_revision: str
    language: str
    euroeval_version: str
    provenance: JsonObject
    expected_identities: tuple[ResultIdentity, ...]
    records: tuple[ValidatedRecord, ...]
    warnings: tuple[str, ...]
    checks: tuple[str, ...]


class BucketStore:
    """Small exact-file wrapper around Hugging Face bucket APIs."""

    def __init__(self, api: BucketApi, token: str, staging_bucket: str) -> None:
        """Initialise and verify the private staging bucket.

        Raises:
            ReviewError:
                If the staging bucket is public.
        """
        self.api = api
        self.token = token
        self.staging_bucket = staging_bucket
        info = self.api.bucket_info(staging_bucket, token=token)
        if not info.private:
            raise ReviewError("HF_STAGING_BUCKET must be private")

    def list_manifests(self) -> list[str]:
        """List durable submission manifests without consulting Redis.

        Returns:
            Sorted bucket paths for all manifests.
        """
        entries = self.api.list_bucket_tree(
            self.staging_bucket,
            prefix=_MANIFEST_PREFIX,
            recursive=True,
            token=self.token,
        )
        return sorted(
            entry.path
            for entry in entries
            if getattr(entry, "type", None) == "file" and entry.path.endswith(".json")
        )

    def read(self, bucket: str, path: str) -> bytes:
        """Download one exact bucket object.

        Returns:
            Exact object bytes.

        Raises:
            ReviewError:
                If the object cannot be found.
        """
        with tempfile.TemporaryDirectory(prefix="euroeval-review-") as directory:
            target = Path(directory) / "object"
            self.api.download_bucket_files(
                bucket, [(path, target)], raise_on_missing_files=True, token=self.token
            )
            if not target.is_file():
                raise ReviewError(f"Bucket object is missing: {bucket}/{path}")
            return target.read_bytes()

    def read_optional(self, bucket: str, path: str) -> bytes | None:
        """Read an object when present.

        Returns:
            Exact object bytes, or ``None`` when absent.
        """
        entries = list(self.api.get_bucket_paths_info(bucket, [path], token=self.token))
        if not entries:
            return None
        return self.read(bucket=bucket, path=path)

    def write_verified(self, bucket: str, path: str, content: bytes) -> None:
        """Upload bytes and verify destination metadata and content."""
        self.api.batch_bucket_files(
            bucket_id=bucket, add=[(content, path)], token=self.token
        )
        self.verify(bucket=bucket, path=path, content=content)

    def verify(self, bucket: str, path: str, content: bytes) -> None:
        """Verify an existing object's metadata, content, and digest.

        Raises:
            ReviewError:
                If the object does not match the expected bytes.
        """
        entries = list(self.api.get_bucket_paths_info(bucket, [path], token=self.token))
        if len(entries) != 1:
            raise ReviewError(f"Upload metadata is missing for {bucket}/{path}")
        entry = entries[0]
        if entry.path != path or entry.size != len(content) or not entry.xet_hash:
            raise ReviewError(f"Upload metadata is inconsistent for {bucket}/{path}")
        downloaded = self.read(bucket=bucket, path=path)
        if downloaded != content or _digest(downloaded) != _digest(content):
            raise ReviewError(f"Upload verification failed for {bucket}/{path}")


class VolunteerReviewer:
    """Review staged submissions and apply resumable terminal decisions."""

    def __init__(
        self,
        store: BucketStore,
        results_bucket: str = HF_RESULTS_BUCKET,
        promoter: t.Callable[..., None] | None = None,
        reserver: BrokerReservation | None = None,
        now: t.Callable[[], dt.datetime] | None = None,
        scope_policy: JsonObject | None = None,
    ) -> None:
        """Initialise the review service."""
        self.store = store
        self.results_bucket = results_bucket
        self._broker_promoter = promoter is None
        self.promoter = promoter or promote_with_broker
        self.reserver = reserver or (
            reserve_with_broker if self._broker_promoter else None
        )
        self.now = now or (lambda: dt.datetime.now(tz=dt.UTC))
        self.scope_policy = scope_policy or load_scope_policy()

    def list_submissions(self) -> list[tuple[str, int, str, str]]:
        """Return submission ID, issue, contributor, and language summaries."""
        summaries: list[tuple[str, int, str, str]] = []
        for path in self.store.list_manifests():
            manifest = _load_object(
                content=self.store.read(self.store.staging_bucket, path), context=path
            )
            submission_id, issue, contributor, language = _manifest_summary(manifest)
            summaries.append((submission_id, issue, contributor, language))
        return summaries

    def show(self, submission_id: str) -> ReviewReport:
        """Independently load and validate a staged submission.

        Returns:
            The validated submission report.

        Raises:
            ReviewError:
                If the manifest path and submission identity differ.
        """
        path = f"{_MANIFEST_PREFIX}/{submission_id}.json"
        manifest = _load_object(
            content=self.store.read(self.store.staging_bucket, path), context=path
        )
        if manifest.get("submission_id") != submission_id:
            raise ReviewError("Manifest submission_id does not match its durable path")
        return _validate_manifest(
            manifest=manifest, store=self.store, scope_policy=self.scope_policy
        )

    def decide(
        self,
        submission_id: str,
        outcome: t.Literal["accepted", "rejected"],
        reviewer: str,
        reasons: list[str] | None = None,
    ) -> ReviewReport:
        """Revalidate evidence, persist a decision, then notify the broker.

        Returns:
            The independently validated submission report.
        """
        report = self.show(submission_id=submission_id)
        decision_path = f"{_DECISION_PREFIX}/{submission_id}.json"
        existing = self.store.read_optional(self.store.staging_bucket, decision_path)
        if existing is not None:
            decision = _load_object(content=existing, context=decision_path)
            _validate_existing_decision(
                decision=decision, report=report, outcome=outcome
            )
        evidence = sorted(
            [
                {
                    "identity": json.dumps(record.identity, separators=(",", ":")),
                    "digest": record.digest,
                }
                for record in report.records
            ],
            key=lambda item: item["identity"],
        )
        token = (
            self.reserver(report.issue_number, submission_id, outcome, evidence)
            if self.reserver
            else "local-test-reservation"
        )
        if outcome == "accepted":
            self._promote_records(report=report)
        if existing is None:
            decision_bytes = _decision_bytes(
                report=report,
                outcome=outcome,
                reviewer=reviewer,
                reasons=reasons or [],
                decided_at=self.now(),
            )
            self.store.write_verified(
                bucket=self.store.staging_bucket,
                path=decision_path,
                content=decision_bytes,
            )
        if self._broker_promoter:
            t.cast(BrokerPromoter, self.promoter)(
                report.issue_number, submission_id, outcome, token, evidence
            )
        else:
            t.cast(t.Callable[[int, str, str], None], self.promoter)(
                report.issue_number, submission_id, outcome
            )
        return report

    def _promote_records(self, report: ReviewReport) -> None:
        _raise_on_identity_collisions(record.identity for record in report.records)
        existing: dict[str, bytes | None] = {
            record.canonical_path: self.store.read_optional(
                self.results_bucket, record.canonical_path
            )
            for record in report.records
        }
        for record in report.records:
            current = existing[record.canonical_path]
            if current is not None and current != record.content:
                raise ReviewError(
                    "Canonical result collision at "
                    f"{record.canonical_path}: {_digest(current)} != {record.digest}"
                )
        for record in report.records:
            if existing[record.canonical_path] is None:
                self.store.write_verified(
                    bucket=self.results_bucket,
                    path=record.canonical_path,
                    content=record.content,
                )
            else:
                self.store.verify(
                    bucket=self.results_bucket,
                    path=record.canonical_path,
                    content=record.content,
                )


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


def reserve_with_broker(
    issue_number: int, submission_id: str, outcome: str, records: list[dict[str, str]]
) -> str:
    """Reserve one immutable maintainer outcome at the broker.

    Returns:
        The opaque reservation token.

    Raises:
        ReviewError:
            If the secret is absent or the broker rejects the reservation.
    """
    secret = os.environ.get("VOLUNTEER_PROMOTION_SECRET")
    if not secret:
        raise ReviewError("VOLUNTEER_PROMOTION_SECRET is required")
    endpoint = os.environ.get(
        "VOLUNTEER_BROKER_RESERVATION_URL",
        "https://euroeval.com/api/worker/promotion-lock",
    )
    payload = json.dumps(
        {
            "protocol_version": PROTOCOL_VERSION,
            "issue_number": issue_number,
            "submission_id": submission_id,
            "outcome": outcome,
            "records": records,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"content-type": "application/json", "x-promotion-secret": secret},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewError("Broker promotion reservation failed") from error
    token = body.get("token") if isinstance(body, dict) else None
    if not isinstance(token, str) or not token:
        raise ReviewError("Broker did not return a promotion reservation token")
    return token


def promote_with_broker(
    issue_number: int,
    submission_id: str,
    outcome: str,
    reservation_token: str,
    records: list[dict[str, str]],
) -> None:
    """Call the authenticated broker promotion transition.

    Raises:
        ReviewError:
            If the secret is absent or the broker does not confirm the transition.
    """
    secret = os.environ.get("VOLUNTEER_PROMOTION_SECRET")
    if not secret:
        raise ReviewError("VOLUNTEER_PROMOTION_SECRET is required")
    endpoint = os.environ.get(
        "VOLUNTEER_BROKER_PROMOTION_URL", "https://euroeval.com/api/worker/promote"
    )
    payload = json.dumps(
        {
            "protocol_version": PROTOCOL_VERSION,
            "issue_number": issue_number,
            "submission_id": submission_id,
            "outcome": outcome,
            "reservation_token": reservation_token,
            "records": records,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"content-type": "application/json", "x-promotion-secret": secret},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict) or body.get("status") != outcome:
        raise ReviewError("Broker did not confirm the promotion transition")


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
    for field in ("model_profile", "worker_version", "image_digest", "created_at"):
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
            "model_profile",
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


def _validate_scope_policy(
    manifest: JsonObject, expected_scope: JsonObject, scope_policy: JsonObject
) -> None:
    policies = scope_policy.get("policies")
    if expected_scope.get("policy_version") != scope_policy.get(
        "policy_version"
    ) or not isinstance(policies, list):
        raise ReviewError("Manifest scope policy version is not trusted")
    profile = manifest.get("model_profile")
    language = manifest.get("language")
    version = manifest.get("euroeval_version")
    matching = [
        item
        for item in policies
        if isinstance(item, dict)
        and item.get("model_profile") == profile
        and item.get("language") == language
        and _normalise_version(str(item.get("euroeval_version")))
        == _normalise_version(str(version))
    ]
    if len(matching) != 1:
        raise ReviewError("Manifest scope has no unique trusted policy entry")
    trusted = matching[0]
    for field in ("language_group", "identity_suffixes", "count", "warnings"):
        if expected_scope.get(field) != trusted.get(field):
            raise ReviewError(f"Manifest scope differs from policy field {field}")


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


def _validate_existing_decision(
    decision: JsonObject, report: ReviewReport, outcome: str
) -> None:
    if decision.get("outcome") != outcome:
        raise ReviewError("Submission already has the opposite terminal decision")
    expected_records = [
        {
            "identity": list(record.identity),
            "digest": record.digest,
            "canonical_path": record.canonical_path,
        }
        for record in report.records
    ]
    if (
        decision.get("protocol_version") != PROTOCOL_VERSION
        or decision.get("artifact") != "volunteer-review-decision/v1"
        or decision.get("immutable") is not True
        or decision.get("submission_id") != report.submission_id
        or decision.get("issue_number") != report.issue_number
        or decision.get("records") != expected_records
        or not isinstance(decision.get("reviewer"), str)
        or not decision.get("reviewer")
    ):
        raise ReviewError("Existing decision artifact is malformed or inconsistent")


def _decision_bytes(
    report: ReviewReport,
    outcome: str,
    reviewer: str,
    reasons: list[str],
    decided_at: dt.datetime,
) -> bytes:
    if not reviewer.strip():
        raise ReviewError("A verified reviewer login is required")
    decision = {
        "protocol_version": PROTOCOL_VERSION,
        "artifact": "volunteer-review-decision/v1",
        "immutable": True,
        "submission_id": report.submission_id,
        "issue_number": report.issue_number,
        "reviewer": reviewer,
        "decided_at": decided_at.astimezone(dt.UTC).isoformat().replace("+00:00", "Z"),
        "outcome": outcome,
        "reasons": reasons,
        "warnings": list(report.warnings),
        "records": [
            {
                "identity": list(record.identity),
                "digest": record.digest,
                "canonical_path": record.canonical_path,
            }
            for record in report.records
        ],
    }
    return (
        json.dumps(decision, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _manifest_summary(manifest: JsonObject) -> tuple[str, int, str, str]:
    submission_id = _required_string(manifest, "submission_id")
    contributor = _required_string(manifest, "verified_contributor")
    language = _required_string(manifest, "language")
    issue = manifest.get("issue_number")
    if isinstance(issue, bool) or not isinstance(issue, int) or issue <= 0:
        raise ReviewError("Manifest issue_number must be a positive integer")
    return submission_id, issue, contributor, language


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


def _required_object(parent: JsonObject, key: str) -> JsonObject:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ReviewError(f"{key} must be an object")
    return value


def _required_string(parent: JsonObject, key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ReviewError(f"{key} must be a non-empty string")
    return value


def _string_list(value: object, context: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ReviewError(f"{context} must be a list of strings")
    return t.cast(list[str], value)


def _json_string_list(value: object, context: str) -> list[str]:
    parsed = _json_value(value, context)
    return _string_list(parsed, context)


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


def _normalise_version(value: str) -> str:
    return _VERSION_SUFFIX_RE.sub("", value)


def _finite(value: float) -> bool:
    return value != float("inf") and value != float("-inf") and value == value


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
