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


class VolunteerReviewer:
    """Review staged submissions and apply resumable terminal decisions."""

    def __init__(
        self,
        store: BucketStore,
        results_bucket: str = HF_RESULTS_BUCKET,
        promoter: t.Callable[..., None] | None = None,
        reserver: BrokerReservation | None = None,
        renewer: BrokerRenewer | None = None,
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
        self.now = now
        self.scope_policy = scope_policy or load_scope_policy()

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
        reservation = (
            self.reserver(
                report.issue_number, submission_id, outcome, reviewer, evidence
            )
            if self.reserver
            else _local_reservation(reviewer=reviewer, existing=existing)
        )
        decision_bytes = _decision_bytes(
            report=report,
            outcome=outcome,
            reviewer=reservation.decision_reviewer,
            reasons=reasons or [],
            decided_at=reservation.decision_created_at,
        )
        if existing is not None:
            decision = _load_object(content=existing, context=decision_path)
            _validate_existing_decision(
                content=existing,
                decision=decision,
                expected_content=decision_bytes,
                report=report,
                outcome=outcome,
            )
        token = reservation.token
        if outcome == "accepted":
            self._promote_records(
                report=report,
                renew=lambda: self._renew_reservation(
                    report=report, records=evidence, token=token
                ),
            )
            self._renew_reservation(report=report, records=evidence, token=token)
        if existing is None:
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

    def _promote_records(
        self, report: ReviewReport, renew: c.Callable[[], None] | None = None
    ) -> None:
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

    def _renew_reservation(
        self, report: ReviewReport, records: list[dict[str, str]], token: str
    ) -> None:
        if not self.renewer:
            return
        renewed = self.renewer(
            report.issue_number, report.submission_id, "accepted", token, records
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
        "artifact": "volunteer-review-decision/v1",
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


def _validate_existing_decision(
    content: bytes,
    decision: JsonObject,
    expected_content: bytes,
    report: ReviewReport,
    outcome: str,
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
        or not isinstance(decision.get("decided_at"), str)
        or not decision.get("decided_at")
        or content != expected_content
    ):
        raise ReviewError("Existing decision artifact is malformed or inconsistent")


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


def renew_with_broker(
    issue_number: int,
    submission_id: str,
    outcome: str,
    reservation_token: str,
    records: list[dict[str, str]],
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
    )
    if not isinstance(reservation, str):
        raise ReviewError("Broker returned decision metadata during renewal")
    return reservation


def _request_reservation(
    issue_number: int,
    submission_id: str,
    outcome: str,
    records: list[dict[str, str]],
    reservation_token: str | None = None,
    reviewer: str | None = None,
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
