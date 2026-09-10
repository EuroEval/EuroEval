"""Promotion and decision transactions for volunteer-result review."""

from __future__ import annotations

import collections.abc as c
import datetime as dt
import json
import os
import typing as t
import urllib.request

from leaderboards.constants import HF_RESULTS_BUCKET

from .review_models import (
    BrokerBinder,
    BrokerPromoter,
    BrokerRenewer,
    BrokerReservation,
    BrokerReservationResult,
    JsonObject,
    ReviewError,
    ReviewReport,
)
from .review_storage import BucketStore, _digest
from .review_validation import (
    _load_object,
    _manifest_summary,
    _raise_on_identity_collisions,
    _validate_manifest,
    load_scope_policy,
)
from .types import PROTOCOL_VERSION

_DECISION_PREFIX = "volunteer/decisions"
_MANIFEST_PREFIX = "volunteer/manifests"
_LOCAL_DECISION_CREATED_AT = "1970-01-01T00:00:00Z"
_DECISION_ARTIFACT = "volunteer-review-decision/v1"


class VolunteerReviewer:
    """Review staged submissions and apply resumable terminal decisions."""

    def __init__(
        self,
        store: BucketStore,
        results_bucket: str = HF_RESULTS_BUCKET,
        promoter: t.Callable[..., None] | None = None,
        reserver: BrokerReservation | None = None,
        renewer: BrokerRenewer | None = None,
        binder: BrokerBinder | None = None,
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
        self.renewer = renewer or (renew_with_broker if self._broker_promoter else None)
        self.binder = binder or (bind_with_broker if self._broker_promoter else None)
        self.now = now
        self.scope_policy = scope_policy or load_scope_policy()

    def decide(
        self,
        submission_id: str,
        outcome: t.Literal["accepted", "rejected"],
        reviewer: str,
        reasons: list[str] | None = None,
    ) -> ReviewReport:
        """Persist one immutable decision and complete its broker transition.

        Returns:
            The independently validated submission report.

        Raises:
            ReviewError:
                If durable decision state or broker fencing is inconsistent.
        """
        report = self.show(submission_id=submission_id)
        reasons = reasons or []
        existing = _existing_decisions(store=self.store, report=report, outcome=outcome)
        evidence = sorted(
            [
                {
                    "identity": json.dumps(record.identity, separators=(",", ":")),
                    "canonical_path": record.canonical_path,
                    "digest": record.digest,
                }
                for record in report.records
            ],
            key=lambda item: item["identity"],
        )
        if outcome == "accepted":
            self._validate_canonical_records(report=report)
        reservation = (
            self.reserver(
                report.issue_number, submission_id, outcome, reviewer, evidence
            )
            if self.reserver
            else _local_reservation(
                reviewer=reviewer, existing=existing[0][1] if existing else None
            )
        )
        decision_bytes = (
            existing[0][1]
            if existing
            else _decision_bytes(
                report=report,
                outcome=outcome,
                reviewer=reservation.decision_reviewer,
                reasons=reasons,
                decided_at=reservation.decision_created_at,
            )
        )
        decision_digest = _digest(decision_bytes)
        token = reservation.token
        if self.binder:
            bound = self.binder(
                report.issue_number,
                submission_id,
                outcome,
                token,
                decision_digest,
                evidence,
            )
            if bound != token:
                raise ReviewError("Broker returned a different bound reservation token")
        current = _existing_decisions(store=self.store, report=report, outcome=outcome)
        if current and any(content != decision_bytes for _, content, _ in current):
            raise ReviewError("Conflicting decision artifacts exist")
        content_path = _decision_path(submission_id, decision_digest)
        if not any(path == content_path for path, _, _ in current):
            self.store.write_verified(
                bucket=self.store.staging_bucket,
                path=content_path,
                content=decision_bytes,
            )
        effective = _existing_decisions(
            store=self.store, report=report, outcome=outcome
        )
        if not effective or {digest for _, _, digest in effective} != {decision_digest}:
            raise ReviewError(
                "Decision artifacts do not resolve to one durable outcome"
            )
        if outcome == "accepted":
            self._renew_reservation(
                report=report,
                records=evidence,
                token=token,
                decision_digest=decision_digest,
            )
            self._promote_records(
                report=report,
                renew=lambda: self._renew_reservation(
                    report=report,
                    records=evidence,
                    token=token,
                    decision_digest=decision_digest,
                ),
            )
        if self._broker_promoter:
            t.cast(BrokerPromoter, self.promoter)(
                report.issue_number,
                submission_id,
                outcome,
                token,
                evidence,
                decision_digest,
            )
        else:
            t.cast(t.Callable[[int, str, str], None], self.promoter)(
                report.issue_number, submission_id, outcome
            )
        return report

    def _promote_records(
        self, report: ReviewReport, renew: c.Callable[[], None] | None = None
    ) -> None:
        self._validate_canonical_records(report=report)
        for record in report.records:
            if renew:
                renew()
            current = self.store.read_optional(
                self.results_bucket, record.canonical_path
            )
            if current is not None and current != record.content:
                raise ReviewError(
                    "Canonical result collision at "
                    f"{record.canonical_path}: {_digest(current)} != {record.digest}"
                )
            if current is None:
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

    def _validate_canonical_records(self, report: ReviewReport) -> None:
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

    def _renew_reservation(
        self,
        report: ReviewReport,
        records: list[dict[str, str]],
        token: str,
        decision_digest: str,
    ) -> None:
        if not self.renewer:
            return
        renewed = self.renewer(
            report.issue_number,
            report.submission_id,
            "accepted",
            token,
            records,
            decision_digest,
        )
        if renewed != token:
            raise ReviewError("Broker returned a different promotion reservation token")

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


def _decision_bytes(
    report: ReviewReport,
    outcome: str,
    reviewer: str,
    reasons: list[str],
    decided_at: str,
) -> bytes:
    if not reviewer.strip():
        raise ReviewError("A verified reviewer login is required")
    if not decided_at.strip():
        raise ReviewError("Broker returned an empty decision timestamp")
    decision = {
        "protocol_version": PROTOCOL_VERSION,
        "artifact": _DECISION_ARTIFACT,
        "immutable": True,
        "submission_id": report.submission_id,
        "issue_number": report.issue_number,
        "reviewer": reviewer,
        "decided_at": decided_at,
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


def _decision_path(submission_id: str, digest: str) -> str:
    """Return the immutable content-addressed path for a decision."""
    return f"{_DECISION_PREFIX}/{submission_id}/{digest}.json"


def _existing_decisions(
    store: BucketStore, report: ReviewReport, outcome: str
) -> list[tuple[str, bytes, str]]:
    """Load and validate every durable decision for a submission.

    Returns:
        Validated paths, bytes, and SHA256 digests.

    Raises:
        ReviewError:
            If an artifact is malformed, conflicting, or misplaced.
    """
    decisions: list[tuple[str, bytes, str]] = []
    for path in store.list_decisions(report.submission_id):
        content = store.read(store.staging_bucket, path)
        decision = _load_object(content=content, context=path)
        _validate_existing_decision(
            content=content, decision=decision, report=report, outcome=outcome
        )
        digest = _digest(content)
        legacy_path = f"{_DECISION_PREFIX}/{report.submission_id}.json"
        if path != legacy_path and path != _decision_path(report.submission_id, digest):
            raise ReviewError("Decision artifact path is not content-addressed")
        decisions.append((path, content, digest))
    digests = {digest for _, _, digest in decisions}
    if len(digests) > 1:
        raise ReviewError("Conflicting decision artifacts exist")
    return decisions


def _validate_existing_decision(
    content: bytes, decision: JsonObject, report: ReviewReport, outcome: str
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
    decision_reasons = decision.get("reasons")
    reviewer = decision.get("reviewer")
    decided_at = decision.get("decided_at")
    if (
        decision.get("protocol_version") != PROTOCOL_VERSION
        or decision.get("artifact") != _DECISION_ARTIFACT
        or decision.get("immutable") is not True
        or decision.get("submission_id") != report.submission_id
        or decision.get("issue_number") != report.issue_number
        or decision.get("records") != expected_records
        or not isinstance(reviewer, str)
        or not reviewer.strip()
        or not isinstance(decided_at, str)
        or not decided_at.strip()
        or not isinstance(decision_reasons, list)
        or any(not isinstance(reason, str) for reason in decision_reasons)
    ):
        raise ReviewError("Existing decision artifact is malformed or inconsistent")
    expected_content = _decision_bytes(
        report=report,
        outcome=outcome,
        reviewer=reviewer,
        reasons=t.cast(list[str], decision_reasons),
        decided_at=decided_at,
    )
    if content != expected_content:
        raise ReviewError("Existing decision artifact is malformed or inconsistent")


def _local_reservation(
    reviewer: str, existing: bytes | None
) -> BrokerReservationResult:
    if existing is not None:
        decision = _load_object(
            content=existing, context="existing volunteer decision artifact"
        )
        existing_reviewer = decision.get("reviewer")
        existing_created_at = decision.get("decided_at")
        if isinstance(existing_reviewer, str) and isinstance(existing_created_at, str):
            return BrokerReservationResult(
                token="local-test-reservation",
                decision_reviewer=existing_reviewer,
                decision_created_at=existing_created_at,
            )
    return BrokerReservationResult(
        token="local-test-reservation",
        decision_reviewer=reviewer,
        decision_created_at=_LOCAL_DECISION_CREATED_AT,
    )


def bind_with_broker(
    issue_number: int,
    submission_id: str,
    outcome: str,
    reservation_token: str,
    decision_digest: str,
    records: list[dict[str, str]],
) -> str:
    """Atomically bind a decision digest to an active broker reservation.

    Returns:
        The unchanged reservation token.

    Raises:
        ReviewError:
            If the broker cannot bind the digest.
    """
    result = _request_reservation(
        issue_number=issue_number,
        submission_id=submission_id,
        outcome=outcome,
        records=records,
        reservation_token=reservation_token,
        decision_digest=decision_digest,
    )
    if not isinstance(result, str):
        raise ReviewError("Broker returned decision metadata during binding")
    return result


def _request_reservation(
    issue_number: int,
    submission_id: str,
    outcome: str,
    records: list[dict[str, str]],
    reservation_token: str | None = None,
    reviewer: str | None = None,
    decision_digest: str | None = None,
) -> str | BrokerReservationResult:
    secret = os.environ.get("VOLUNTEER_PROMOTION_SECRET")
    if not secret:
        raise ReviewError("VOLUNTEER_PROMOTION_SECRET is required")
    endpoint = os.environ.get(
        "VOLUNTEER_BROKER_RESERVATION_URL",
        "https://euroeval.com/api/worker/promotion-lock",
    )
    request_body: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "issue_number": issue_number,
        "submission_id": submission_id,
        "outcome": outcome,
        "records": records,
    }
    if reservation_token is not None:
        request_body["reservation_token"] = reservation_token
    if reviewer is not None:
        request_body["reviewer"] = reviewer
    if decision_digest is not None:
        request_body["decision_digest"] = decision_digest
    payload = json.dumps(request_body).encode("utf-8")
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
    if reservation_token is not None:
        return token
    decision_reviewer = (
        body.get("decision_reviewer") if isinstance(body, dict) else None
    )
    decision_created_at = (
        body.get("decision_created_at") if isinstance(body, dict) else None
    )
    if not isinstance(decision_reviewer, str) or not decision_reviewer:
        raise ReviewError("Broker did not return a decision reviewer")
    if not isinstance(decision_created_at, str) or not decision_created_at:
        raise ReviewError("Broker did not return a decision timestamp")
    return BrokerReservationResult(
        token=token,
        decision_reviewer=decision_reviewer,
        decision_created_at=decision_created_at,
    )


def promote_with_broker(
    issue_number: int,
    submission_id: str,
    outcome: str,
    reservation_token: str,
    records: list[dict[str, str]],
    decision_digest: str,
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
            "decision_digest": decision_digest,
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


def renew_with_broker(
    issue_number: int,
    submission_id: str,
    outcome: str,
    reservation_token: str,
    records: list[dict[str, str]],
    decision_digest: str | None = None,
) -> str:
    """Renew a pending broker reservation without changing its token.

    Returns:
        The unchanged opaque reservation token.

    Raises:
        ReviewError:
            If the broker returns decision metadata during renewal.
    """
    reservation = _request_reservation(
        issue_number=issue_number,
        submission_id=submission_id,
        outcome=outcome,
        records=records,
        reservation_token=reservation_token,
        decision_digest=decision_digest,
    )
    if not isinstance(reservation, str):
        raise ReviewError("Broker returned decision metadata during renewal")
    return reservation


def reserve_with_broker(
    issue_number: int,
    submission_id: str,
    outcome: str,
    reviewer: str,
    records: list[dict[str, str]],
) -> BrokerReservationResult:
    """Reserve one immutable maintainer outcome at the broker.

    Returns:
        The reservation token and server-bound decision metadata.

    Raises:
        ReviewError:
            If the broker omits the reservation or decision metadata.
    """
    reservation = _request_reservation(
        issue_number=issue_number,
        submission_id=submission_id,
        outcome=outcome,
        records=records,
        reviewer=reviewer,
    )
    if not isinstance(reservation, BrokerReservationResult):
        raise ReviewError("Broker returned an invalid promotion reservation")
    return reservation
