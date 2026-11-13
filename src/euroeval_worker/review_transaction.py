"""Promotion and decision transactions for volunteer-result review."""

from __future__ import annotations

import collections.abc as c
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import os
import typing as t
import urllib.request
from contextlib import contextmanager

from euroeval.canary_evidence import evidence_from_dict
from leaderboards.canonical_publication import CanonicalPublisher, PublicationSession
from leaderboards.constants import HF_RESULTS_BUCKET, REQUIRED_METADATA_FIELDS
from leaderboards.contamination_canary import process_contamination_canaries
from leaderboards.eee_validation import is_eee_record
from leaderboards.result_identity import dedup_newer_record, identity_from_eee_record

from .review_models import (
    BrokerBinder,
    BrokerRenewer,
    BrokerReservation,
    BrokerReservationResult,
    CanonicalObservation,
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
_COMPLETION_PREFIX = "volunteer/completions"
_MANIFEST_PREFIX = "volunteer/manifests"
_LOCAL_DECISION_CREATED_AT = "1970-01-01T00:00:00Z"
_DECISION_ARTIFACT = "volunteer-review-decision/v1"

BrokerStatus = t.Literal["absent", "reserved", "terminal"]
BrokerStatusChecker = c.Callable[
    [int, str, str, list[dict[str, str]], str], BrokerStatus
]


def _has_collected_canary(records: list[dict[str, object]]) -> bool:
    """Return whether any canary record reports collected observations.

    Args:
        records: Staged EEE records to inspect.

    Returns:
        ``True`` when a record's canary evidence status is ``collected``.
    """
    for record in records:
        library = record.get("eval_library")
        details = (
            library.get("additional_details") if isinstance(library, dict) else None
        )
        raw = (
            details.get("contamination_canary_evidence")
            if isinstance(details, dict)
            else None
        )
        if isinstance(raw, str):
            raw = json.loads(raw)
        if evidence_from_dict(raw).status == "collected":
            return True
    return False


def _is_canary_identity(identity: tuple[str, str, bool | None, bool | None]) -> bool:
    """Return whether an identity names a contamination-canary dataset.

    Args:
        identity: Validated result identity tuple.

    Returns:
        ``True`` for ``contamination-canary`` or a hyphenated variant.
    """
    return identity[1] == "contamination-canary" or identity[1].startswith(
        "contamination-canary-"
    )


logger = logging.getLogger(__name__)


class VolunteerReviewer:
    """Review staged submissions and apply resumable terminal decisions.

    Raises:
    ReviewError: Raised when the operation cannot be completed.
    """

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
        publisher: CanonicalPublisher | None = None,
        publisher_factory: c.Callable[[], CanonicalPublisher] | None = None,
        status_checker: BrokerStatusChecker | None = None,
    ) -> None:
        """Initialise the review service.

        Args:
            store:
                Store.
            results_bucket (optional):
                Results bucket.
            promoter (optional):
                Promoter.
            reserver (optional):
                Reserver.
            renewer (optional):
                Renewer.
            binder (optional):
                Binder.
            status_checker (optional):
                Read-only broker status lookup. Defaults to the broker transport
                only when using the production promoter.
            now (optional):
                Now.
            scope_policy (optional):
                Scope policy.
            publisher (optional):
                Canonical publisher used to open publication sessions.
            publisher_factory (optional):
                Factory for a canonical publisher when publication is deferred.
        """
        self.store = store
        self.results_bucket = results_bucket
        self._broker_promoter = promoter is None
        self.promoter = promoter or promote_with_broker
        self.reserver = reserver or (
            reserve_with_broker if self._broker_promoter else None
        )
        self.renewer = renewer or (renew_with_broker if self._broker_promoter else None)
        self.binder = binder or (bind_with_broker if self._broker_promoter else None)
        self.status_checker = status_checker or (
            inspect_with_broker if self._broker_promoter else None
        )
        self.now = now
        self.scope_policy = scope_policy or load_scope_policy()
        self.publisher = publisher
        self.publisher_factory = publisher_factory

    def decide(
        self,
        submission_id: str,
        outcome: t.Literal["accepted", "rejected"],
        reviewer: str,
        reasons: list[str] | None = None,
        plan_fingerprint: str | None = None,
    ) -> ReviewReport:
        """Apply a confirmed decision, fencing accepted publication.

        Args:
            submission_id:
                Submission being decided.
            outcome:
                Confirmed terminal outcome.
            reviewer:
                Reviewer applying the decision.
            reasons (optional):
                Reasons recorded with the decision.
            plan_fingerprint (optional):
                Confirmed canonical publication plan fingerprint.

        Returns:
            The review report for the decided submission.

        Raises:
            ReviewError:
                If the decision or its publication cannot be completed safely.
        """
        inspected = self.show(submission_id=submission_id)
        existing = _existing_decisions(
            store=self.store, report=inspected, outcome=outcome
        )
        if existing:
            decision = _load_object(content=existing[0][1], context=existing[0][0])
            decision_digest = existing[0][2]
            receipted = self._has_valid_completion_receipt(
                submission_id=submission_id,
                outcome=outcome,
                decision_digest=decision_digest,
            )
            terminal = (
                receipted
                or self._inspect_decision(
                    report=inspected, outcome=outcome, decision_digest=decision_digest
                )
                == "terminal"
            )
            if terminal:
                _validate_predecessor_archives(
                    store=self.store,
                    report=inspected,
                    decision=decision,
                    results_bucket=self.results_bucket,
                    completed=True,
                )
                if not receipted:
                    self._write_completion_receipt(
                        submission_id=submission_id,
                        outcome=outcome,
                        decision_digest=decision_digest,
                    )
                return inspected
        if outcome == "rejected":
            return self._decide(
                submission_id=submission_id,
                outcome=outcome,
                reviewer=reviewer,
                reasons=reasons,
                plan_fingerprint=plan_fingerprint,
                publication=None,
            )
        if (
            not existing
            and plan_fingerprint is not None
            and plan_fingerprint != inspected.plan_fingerprint
        ):
            raise ReviewError(
                "Canonical plan changed or was not confirmed; run show and retry"
            )
        with self._publication_session() as publication:
            return self._decide(
                submission_id=submission_id,
                outcome=outcome,
                reviewer=reviewer,
                reasons=reasons,
                plan_fingerprint=plan_fingerprint,
                publication=publication,
            )

    def _decide(
        self,
        submission_id: str,
        outcome: t.Literal["accepted", "rejected"],
        reviewer: str,
        reasons: list[str] | None,
        plan_fingerprint: str | None,
        publication: PublicationSession | None,
    ) -> ReviewReport:
        """Persist one immutable decision and complete its broker transition.

        Args:
                    submission_id:
                        Submission id.
                    outcome:
                        Outcome.
                    reviewer:
                        Reviewer.
                    reasons (optional):
                        Reasons.
                    plan_fingerprint (optional):
                        Fingerprint displayed to and confirmed by the reviewer.
                    publication (optional):
                        Active canonical publication session.

        Returns:
                    The independently validated submission report.

        Raises:
                    ReviewError:
                        If durable decision state or broker fencing is inconsistent.
        """
        report = self.show(submission_id=submission_id)
        reasons = reasons or []
        existing = _existing_decisions(store=self.store, report=report, outcome=outcome)
        if outcome == "accepted" and existing:
            report = _restore_decision_plan(
                store=self.store,
                report=report,
                decision=_load_object(content=existing[0][1], context=existing[0][0]),
                results_bucket=self.results_bucket,
            )
        if outcome == "accepted":
            if existing:
                _validate_predecessor_archives(
                    store=self.store,
                    report=report,
                    decision=_load_object(
                        content=existing[0][1], context=existing[0][0]
                    ),
                    results_bucket=self.results_bucket,
                )
            if any(item.conflict for item in report.canonical_observations):
                raise ReviewError("Canonical collision blocks acceptance")
            if (
                plan_fingerprint is not None
                and plan_fingerprint != report.plan_fingerprint
            ):
                raise ReviewError(
                    "Canonical plan changed or was not confirmed; run show and retry"
                )
            if any(item.replacement for item in report.canonical_observations) and (
                plan_fingerprint != report.plan_fingerprint
            ):
                raise ReviewError(
                    "Canonical plan changed or was not confirmed; run show and retry"
                )

        evidence = _promotion_evidence(report=report)
        if outcome == "accepted":
            canary_records = [
                json.loads(record.content)
                for record in report.records
                if _is_canary_identity(record.identity)
            ]
            if canary_records:
                _, _, _, canary_report = process_contamination_canaries(
                    records=canary_records
                )
                outcomes = canary_report.get("models")
                if _has_collected_canary(canary_records) and (
                    canary_report.get("status") != "scored"
                    or not isinstance(outcomes, list)
                    or any(
                        not isinstance(item, dict) or item.get("status") != "scored"
                        for item in outcomes
                    )
                ):
                    raise ReviewError(
                        "Collected contamination-canary evidence could not be "
                        "validated by private scoring"
                    )
            self._validate_canonical_records(report=report)
        known_digest = existing[0][2] if existing else None
        reservation = self._reserve_decision(
            report=report,
            submission_id=submission_id,
            outcome=outcome,
            reviewer=reviewer,
            evidence=evidence,
            known_digest=known_digest,
            existing=existing,
            publication=publication,
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
                predecessors=[
                    {
                        "canonical_path": observation.canonical_path,
                        "digest": observation.predecessor_digest,
                        "archive_path": _predecessor_path(
                            submission_id, observation.predecessor_digest
                        ),
                    }
                    for observation in report.canonical_observations
                    if observation.replacement
                ]
                if outcome == "accepted"
                else None,
                canonical_plan=[
                    {
                        "canonical_path": observation.canonical_path,
                        "identity": list(observation.identity),
                        "staged_digest": next(
                            record.digest
                            for record in report.records
                            if record.canonical_path == observation.canonical_path
                        ),
                        "predecessor_digest": observation.predecessor_digest,
                        "disposition": "replace"
                        if observation.replacement
                        else "write",
                    }
                    for observation in report.canonical_observations
                ]
                if outcome == "accepted"
                else None,
            )
        )
        decision_digest = _digest(decision_bytes)
        token = reservation.token
        self._bind_decision(
            report=report,
            submission_id=submission_id,
            outcome=outcome,
            token=token,
            decision_digest=decision_digest,
            evidence=evidence,
            publication=publication,
        )
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
                publication=publication,
            )
            self._promote_records(
                report=report,
                publication=publication,
                renew=lambda: self._renew_reservation(
                    report=report,
                    records=evidence,
                    token=token,
                    decision_digest=decision_digest,
                    publication=publication,
                ),
            )
        if self._broker_promoter:
            t.cast(
                c.Callable[
                    [
                        int,
                        str,
                        str,
                        str,
                        list[dict[str, str]],
                        str,
                        str | None,
                        str | None,
                    ],
                    None,
                ],
                self.promoter,
            )(
                report.issue_number,
                submission_id,
                outcome,
                token,
                evidence,
                decision_digest,
                publication.owner if outcome == "accepted" else None,
                publication.bucket if outcome == "accepted" else None,
            )
        else:
            t.cast(t.Callable[[int, str, str], None], self.promoter)(
                report.issue_number, submission_id, outcome
            )
        self._write_completion_receipt(
            submission_id=submission_id,
            outcome=outcome,
            decision_digest=decision_digest,
        )
        return report

    def _bind_decision(
        self,
        report: ReviewReport,
        submission_id: str,
        outcome: t.Literal["accepted", "rejected"],
        token: str,
        decision_digest: str,
        evidence: list[dict[str, str]],
        publication: PublicationSession | None,
    ) -> None:
        """Bind the immutable decision digest to its reservation.

        Args:
            report:
                Submission report being decided.
            submission_id:
                Submission identifier.
            outcome:
                Decision outcome.
            token:
                Reservation token to bind.
            decision_digest:
                Digest of the immutable decision bytes.
            evidence:
                Immutable promotion evidence for the submission.
            publication:
                Canonical publication session, when one is active.

        Raises:
            ReviewError:
                If the broker returns a different reservation token.
        """
        if not self.binder:
            return
        if self._broker_promoter:
            extended_binder = t.cast(
                c.Callable[
                    [
                        int,
                        str,
                        str,
                        str,
                        str,
                        list[dict[str, str]],
                        str | None,
                        str | None,
                    ],
                    str,
                ],
                self.binder,
            )
            bound = extended_binder(
                report.issue_number,
                submission_id,
                outcome,
                token,
                decision_digest,
                evidence,
                publication.owner if publication else None,
                publication.bucket if publication else None,
            )
        else:
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

    def _promote_records(
        self,
        report: ReviewReport,
        publication: PublicationSession | None = None,
        renew: c.Callable[[], None] | None = None,
    ) -> None:
        """Promote records.

        Args:
                    report:
                        Report.
                    publication (optional):
                        Active canonical publication session for verified writes.
                    renew (optional):
                        Renew.

        Raises:
        ReviewError: Raised when the operation cannot be completed.
                    An exception when the operation cannot be completed.
        """
        if publication is None:
            raise ReviewError("Canonical publication session is required")
        self._validate_canonical_records(report=report)
        for observation in report.canonical_observations:
            if not observation.replacement:
                continue
            predecessor = observation.predecessor_content
            digest = observation.predecessor_digest
            if predecessor is None or digest is None:
                raise ReviewError("Replacement plan has no predecessor bytes")
            archive_path = _predecessor_path(report.submission_id, digest)
            archived = self.store.read_optional(self.store.staging_bucket, archive_path)
            if archived is None:
                self.store.write_verified(
                    bucket=self.store.staging_bucket,
                    path=archive_path,
                    content=predecessor,
                )
            else:
                self.store.verify(
                    bucket=self.store.staging_bucket,
                    path=archive_path,
                    content=predecessor,
                )
        self._validate_canonical_records(report=report)
        for record in report.records:
            if _is_canary_identity(record.identity):
                continue
            if renew:
                renew()
            observation = next(
                item
                for item in report.canonical_observations
                if item.canonical_path == record.canonical_path
            )
            current = self.store.read_optional(
                self.results_bucket, record.canonical_path
            )
            if current == record.content:
                self.store.verify(
                    bucket=self.results_bucket,
                    path=record.canonical_path,
                    content=record.content,
                )
            elif current == observation.predecessor_content and (
                current is None or observation.replacement
            ):
                publication.write_verified(
                    path=record.canonical_path,
                    content=record.content,
                    expected=observation.predecessor_content,
                )
            else:
                raise ReviewError("Canonical destination changed during promotion")

    def _validate_canonical_records(self, report: ReviewReport) -> None:
        """Validate canonical records.

        Args:
                    report:
                        Report.

        Raises:
        ReviewError: Raised when the operation cannot be completed.
                    An exception when the operation cannot be completed.
        """
        _raise_on_identity_collisions(record.identity for record in report.records)
        existing: dict[str, bytes | None] = {
            record.canonical_path: self.store.read_optional(
                self.results_bucket, record.canonical_path
            )
            for record in report.records
            if not _is_canary_identity(record.identity)
        }
        for record in report.records:
            if _is_canary_identity(record.identity):
                continue
            try:
                decoded_record = json.loads(record.content)
                if not isinstance(decoded_record, dict) or not is_eee_record(
                    record=decoded_record
                ):
                    raise ReviewError(
                        f"{record.staged_path} is not an EEE-format record."
                    )
                raw_record = t.cast(JsonObject, decoded_record)
                model_info = raw_record["model_info"]
                details = model_info.get("additional_details")
                if "additional_details" in model_info and not isinstance(details, dict):
                    raise ReviewError(
                        f"{record.staged_path} model_info.additional_details "
                        "must be an object when supplied."
                    )
                if isinstance(details, dict):
                    non_bool = [
                        field
                        for field in REQUIRED_METADATA_FIELDS
                        if field in details and not isinstance(details[field], bool)
                    ]
                    if non_bool:
                        raise ReviewError(
                            f"{record.staged_path} has non-boolean precious "
                            "metadata field(s): " + ", ".join(non_bool)
                        )
            except (TypeError, ValueError) as error:
                raise ReviewError(str(error)) from error
            current = existing[record.canonical_path]
            observation = next(
                item
                for item in report.canonical_observations
                if item.canonical_path == record.canonical_path
            )
            if current not in {observation.predecessor_content, record.content}:
                raise ReviewError("Canonical plan changed after reviewer inspection")
            if (
                current is not None
                and current != record.content
                and not observation.replacement
            ):
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
        publication: PublicationSession | None = None,
    ) -> None:
        """Renew reservation.

        Args:
                    report:
                        Report.
                    records:
                        Records.
                    token:
                        Token.
                    decision_digest:
                        Decision digest.
                    publication (optional):
                        Active canonical publication session.

        Raises:
        ReviewError: Raised when the operation cannot be completed.
                    An exception when the operation cannot be completed.
        """
        if not self.renewer:
            return
        if self._broker_promoter:
            extended_renewer = t.cast(
                c.Callable[
                    [
                        int,
                        str,
                        str,
                        str,
                        list[dict[str, str]],
                        str,
                        str | None,
                        str | None,
                    ],
                    str,
                ],
                self.renewer,
            )
            renewed = extended_renewer(
                report.issue_number,
                report.submission_id,
                "accepted",
                token,
                records,
                decision_digest,
                publication.owner if publication else None,
                publication.bucket if publication else None,
            )
        else:
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

    def _reserve_decision(
        self,
        report: ReviewReport,
        submission_id: str,
        outcome: t.Literal["accepted", "rejected"],
        reviewer: str,
        evidence: list[dict[str, str]],
        known_digest: str | None,
        existing: list[tuple[str, bytes, str]],
        publication: PublicationSession | None,
    ) -> BrokerReservationResult:
        """Reserve the decision through the broker or locally.

        Args:
            report:
                Submission report being decided.
            submission_id:
                Submission identifier.
            outcome:
                Decision outcome.
            reviewer:
                Reviewer applying the decision.
            evidence:
                Immutable promotion evidence for the submission.
            known_digest:
                Digest of an existing immutable decision, if present.
            existing:
                Validated existing decision artifacts.
            publication:
                Canonical publication session, when one is active.

        Returns:
            Reservation metadata used to create the immutable decision.
        """
        if self.reserver and self._broker_promoter:
            extended_reserver = t.cast(
                c.Callable[
                    [
                        int,
                        str,
                        str,
                        str,
                        list[dict[str, str]],
                        str | None,
                        str | None,
                        str | None,
                    ],
                    BrokerReservationResult,
                ],
                self.reserver,
            )
            return extended_reserver(
                report.issue_number,
                submission_id,
                outcome,
                reviewer,
                evidence,
                known_digest,
                publication.owner if publication else None,
                publication.bucket if publication else None,
            )
        if self.reserver:
            return self.reserver(
                report.issue_number,
                submission_id,
                outcome,
                reviewer,
                evidence,
                known_digest,
            )
        return _local_reservation(
            reviewer=reviewer, existing=existing[0][1] if existing else None
        )

    def _write_completion_receipt(
        self, submission_id: str, outcome: str, decision_digest: str
    ) -> None:
        """Write and verify the private receipt for a completed decision.

        Args:
            submission_id: Submission whose decision completed.
            outcome: Durable decision outcome.
            decision_digest: Digest of the immutable decision bytes.

        Raises:
            ReviewError: If an existing receipt does not match exactly.
        """
        path = _completion_path(submission_id=submission_id, digest=decision_digest)
        content = _completion_bytes(
            submission_id=submission_id,
            outcome=outcome,
            decision_digest=decision_digest,
        )
        existing = self.store.read_optional(self.store.staging_bucket, path)
        if existing is not None:
            if existing != content:
                raise ReviewError(
                    "Completion receipt differs from the durable decision"
                )
            return
        self.store.write_verified(
            bucket=self.store.staging_bucket, path=path, content=content
        )

    def show(self, submission_id: str) -> ReviewReport:
        """Independently load and validate a staged submission.

        Args:
                    submission_id:
                        Submission id.

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
        report = _validate_manifest(
            manifest=manifest, store=self.store, scope_policy=self.scope_policy
        )
        observations = self._canonical_plan(report=report)
        fingerprint = _plan_fingerprint(report=report, observations=observations)
        return dataclasses.replace(
            report, canonical_observations=observations, plan_fingerprint=fingerprint
        )

    def _canonical_plan(self, report: ReviewReport) -> tuple[CanonicalObservation, ...]:
        """Inspect destinations and permit only strict newer same-identity records.

        Args:
            report: Validated staged result report.

        Returns:
            Immutable destination observations for reviewer confirmation.

        Raises:
            ValueError: If staged identities collide.
        """
        observations: list[CanonicalObservation] = []
        _raise_on_identity_collisions(record.identity for record in report.records)
        for record in report.records:
            if _is_canary_identity(record.identity):
                continue
            current = self.store.read_optional(
                self.results_bucket, record.canonical_path
            )
            replace = False
            if current is not None and current != record.content:
                try:
                    old = json.loads(current)
                    new = json.loads(record.content)
                    if not isinstance(old, dict) or not is_eee_record(record=old):
                        raise ValueError("existing canonical record is malformed")
                    if identity_from_eee_record(old) != record.identity:
                        raise ValueError(
                            "existing canonical record has different identity"
                        )
                    if identity_from_eee_record(new) != record.identity:
                        raise ValueError(
                            "staged record identity does not match its path"
                        )
                    if dedup_newer_record(old, new) is not new:
                        raise ValueError(
                            "staged record does not strictly outrank canonical"
                        )
                    replace = True
                except (TypeError, ValueError) as error:
                    conflict = str(error)
                else:
                    conflict = None
            else:
                conflict = None
            observations.append(
                CanonicalObservation(
                    identity=record.identity,
                    canonical_path=record.canonical_path,
                    predecessor_digest=_digest(current)
                    if current is not None
                    else None,
                    predecessor_content=current,
                    replacement=replace,
                    conflict=conflict,
                )
            )
        return tuple(observations)

    def _has_valid_completion_receipt(
        self,
        submission_id: str,
        outcome: t.Literal["accepted", "rejected"],
        decision_digest: str,
    ) -> bool:
        """Return whether a decision has its exact private completion receipt.

        Args:
            submission_id: Submission owning the decision.
            outcome: Durable decision outcome.
            decision_digest: Digest of the immutable decision bytes.

        Returns:
            Whether the receipt exists and matches the expected bytes exactly.
        """
        path = _completion_path(submission_id=submission_id, digest=decision_digest)
        existing = self.store.read_optional(self.store.staging_bucket, path)
        return existing == _completion_bytes(
            submission_id=submission_id,
            outcome=outcome,
            decision_digest=decision_digest,
        )

    def _inspect_decision(
        self, report: ReviewReport, outcome: str, decision_digest: str
    ) -> BrokerStatus:
        """Inspect broker evidence without reserving or publishing anything.

        Args:
            report: Independently validated submission evidence.
            outcome: Durable decision outcome.
            decision_digest: Digest of the immutable decision bytes.

        Returns:
            Matching broker status, or absent for offline custom promoters.
        """
        if self.status_checker is None:
            return "absent"
        return self.status_checker(
            report.issue_number,
            report.submission_id,
            outcome,
            _promotion_evidence(report=report),
            decision_digest,
        )

    @contextmanager
    def _publication_session(self) -> t.Iterator[PublicationSession]:
        """Open the configured canonical publication session.

        Yields:
            The configured canonical publication session.

        Raises:
            ReviewError:
                If canonical publication is not configured.
        """
        publisher = self.publisher
        if publisher is None and self.publisher_factory is not None:
            publisher = self.publisher_factory()
        if publisher is None:
            raise ReviewError("Canonical publication lock configuration is required")
        with publisher.session() as session:
            yield session

    def list_pending_submissions(self) -> list[tuple[str, int, str, str]]:
        """Return durable manifests without one terminal decision artifact.

        This is a read-only inventory operation. Malformed or conflicting decision
        objects remain pending so that the maintainer is directed to inspect them
        rather than allowing them to silently disappear from the queue.

        Returns:
            Submission ID, issue, contributor, and language summaries.
        """
        summaries = self._list_submission_summaries()
        return [
            summary
            for summary in summaries
            if not self._has_terminal_decision(
                submission_id=summary[0], issue_number=summary[1]
            )
        ]

    def _has_terminal_decision(self, submission_id: str, issue_number: int) -> bool:
        """Return whether decision objects resolve to one terminal outcome.

        Args:
            submission_id: Submission whose durable decision objects are checked.
            issue_number: Issue number expected in its validated manifest.

        Returns:
            ``True`` if validation succeeds and at least one consistent decision
            exists; ``False`` for missing or invalid terminal state.
        """
        try:
            report = self.show(submission_id=submission_id)
            decisions = _existing_decisions(
                store=self.store, report=report, outcome=None
            )
        except ReviewError as error:
            logger.warning(
                "Submission %s remains pending: terminal-state validation failed: %s",
                submission_id,
                error,
            )
            return False
        if report.issue_number != issue_number:
            logger.warning(
                "Submission %s remains pending: manifest issue differs from summary",
                submission_id,
            )
            return False
        if not decisions:
            return False
        decision = _load_object(content=decisions[0][1], context=decisions[0][0])
        if "canonical_plan" not in decision and (
            decision["outcome"] == "accepted" or self.status_checker is None
        ):
            return True
        outcome = t.cast(t.Literal["accepted", "rejected"], decision["outcome"])
        if self._has_valid_completion_receipt(
            submission_id=submission_id,
            outcome=outcome,
            decision_digest=decisions[0][2],
        ):
            return True
        try:
            terminal = (
                self._inspect_decision(
                    report=report, outcome=outcome, decision_digest=decisions[0][2]
                )
                == "terminal"
            )
            if terminal:
                _validate_predecessor_archives(
                    store=self.store,
                    report=report,
                    decision=decision,
                    results_bucket=self.results_bucket,
                    completed=True,
                )
            return terminal
        except (ReviewError, OSError) as error:
            logger.warning(
                "Submission %s remains pending: broker status validation failed: %s",
                submission_id,
                error,
            )
            return False

    def _list_submission_summaries(self) -> list[tuple[str, int, str, str]]:
        """Load summaries for every durable manifest.

        Returns:
            Submission ID, issue, contributor, and language summaries.
        """
        summaries: list[tuple[str, int, str, str]] = []
        for path in self.store.list_manifests():
            manifest = _load_object(
                content=self.store.read(self.store.staging_bucket, path), context=path
            )
            summaries.append(_manifest_summary(manifest))
        return summaries

    def list_submissions(self) -> list[tuple[str, int, str, str]]:
        """Return all durable manifest summaries, including historical ones.

        Returns:
            Manifest summaries as ``(submission_id, issue_number, contributor,
            language)`` tuples.
        """
        return self._list_submission_summaries()


def _completion_bytes(submission_id: str, outcome: str, decision_digest: str) -> bytes:
    """Serialise one deterministic private completion receipt.

    Args:
        submission_id: Submission whose decision completed.
        outcome: Durable decision outcome.
        decision_digest: Digest of the immutable decision bytes.

    Returns:
        Compact, key-sorted UTF-8 JSON bytes ending in a newline.
    """
    receipt = {
        "protocol": PROTOCOL_VERSION,
        "submission": submission_id,
        "outcome": outcome,
        "decision_digest": decision_digest,
    }
    return (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _completion_path(submission_id: str, digest: str) -> str:
    """Return the content-addressed private receipt path.

    Args:
        submission_id: Submission owning the receipt.
        digest: Digest of the immutable decision.

    Returns:
        ``volunteer/completions/<submission_id>/<digest>.json``.
    """
    return f"{_COMPLETION_PREFIX}/{submission_id}/{digest}.json"


def _decision_bytes(
    report: ReviewReport,
    outcome: str,
    reviewer: str,
    reasons: list[str],
    decided_at: str,
    predecessors: list[dict[str, str | None]] | None = None,
    canonical_plan: list[dict[str, object]] | None = None,
) -> bytes:
    """Serialize an immutable decision artifact as canonical UTF-8 JSON.

    Args:
        report: Validated submission evidence referenced by the decision.
        outcome: Terminal outcome, accepted or rejected.
        reviewer: Verified reviewer login.
        reasons: Reviewer-supplied rationale strings.
        decided_at: Broker-bound decision timestamp.
        predecessors (optional): Archived canonical predecessors for replacements.
        canonical_plan (optional): Immutable destination expectations.

    Returns:
        Compact, key-sorted UTF-8 JSON bytes ending in a newline.

    Raises:
        ReviewError: If the reviewer or timestamp is blank.
    """
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
    if predecessors is not None:
        decision["canonical_plan_fingerprint"] = report.plan_fingerprint
        decision["predecessors"] = predecessors
    if canonical_plan is not None:
        decision["canonical_plan"] = canonical_plan
    return (
        json.dumps(decision, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _decision_path(submission_id: str, digest: str) -> str:
    """Return the immutable content-addressed path for a decision.

    Args:
        submission_id: Submission owning the decision.
        digest: SHA-256 digest used as the filename.

    Returns:
        ``volunteer/decisions/<submission_id>/<digest>.json``.
    """
    return f"{_DECISION_PREFIX}/{submission_id}/{digest}.json"


def _existing_decisions(
    store: BucketStore, report: ReviewReport, outcome: str | None
) -> list[tuple[str, bytes, str]]:
    """Load and validate every durable decision for a submission.

    Args:
            store:
                Store.
            report:
                Report.
            outcome:
                Outcome.

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
    content: bytes, decision: JsonObject, report: ReviewReport, outcome: str | None
) -> None:
    """Validate existing decision.

    Args:
            content:
                Content.
            decision:
                Decision.
            report:
                Report.
            outcome:
                Outcome.

    Raises:
    ReviewError: Raised when the operation cannot be completed.
            An exception when the operation cannot be completed.
    """
    decision_outcome = decision.get("outcome")
    if decision_outcome not in {"accepted", "rejected"}:
        raise ReviewError("Existing decision artifact is malformed or inconsistent")
    if outcome is not None and decision_outcome != outcome:
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
    decision_warnings = decision.get("warnings")
    plan_fingerprint = decision.get("canonical_plan_fingerprint")
    predecessors = decision.get("predecessors")
    canonical_plan = decision.get("canonical_plan")
    if canonical_plan is not None:
        expected_by_path = {
            record.canonical_path: record
            for record in report.records
            if not _is_canary_identity(record.identity)
        }
        if not isinstance(canonical_plan, list) or len(canonical_plan) != len(
            expected_by_path
        ):
            raise ReviewError(
                "Existing decision artifact has an invalid canonical plan"
            )
        seen_plan: set[str] = set()
        planned_observations: list[CanonicalObservation] = []
        planned_predecessors: list[dict[str, str]] = []
        for item in canonical_plan:
            if not isinstance(item, dict):
                raise ReviewError(
                    "Existing decision artifact has an invalid canonical plan"
                )
            path = item.get("canonical_path")
            record = expected_by_path.get(path) if isinstance(path, str) else None
            predecessor_digest = item.get("predecessor_digest")
            if (
                record is None
                or path in seen_plan
                or item.get("identity") != list(record.identity)
                or item.get("staged_digest") != record.digest
                or item.get("disposition") not in {"replace", "write"}
                or (
                    predecessor_digest is not None
                    and (
                        not isinstance(predecessor_digest, str)
                        or len(predecessor_digest) != 64
                        or any(
                            char not in "0123456789abcdef"
                            for char in predecessor_digest
                        )
                    )
                )
                or (item.get("disposition") == "replace")
                != (
                    predecessor_digest is not None
                    and predecessor_digest != record.digest
                )
            ):
                raise ReviewError(
                    "Existing decision artifact has an invalid canonical plan"
                )
            seen_plan.add(path)
            replacement = item["disposition"] == "replace"
            planned_observations.append(
                CanonicalObservation(
                    identity=record.identity,
                    canonical_path=path,
                    predecessor_digest=predecessor_digest,
                    predecessor_content=None,
                    replacement=replacement,
                )
            )
            if replacement:
                planned_predecessors.append(
                    {
                        "canonical_path": path,
                        "digest": predecessor_digest,
                        "archive_path": _predecessor_path(
                            report.submission_id, predecessor_digest
                        ),
                    }
                )
        if (
            predecessors != planned_predecessors
            or plan_fingerprint
            != _plan_fingerprint(
                report=report, observations=tuple(planned_observations)
            )
        ):
            raise ReviewError(
                "Existing decision artifact has an inconsistent canonical plan"
            )
    if "canonical_plan_fingerprint" in decision and (
        not isinstance(plan_fingerprint, str)
        or len(plan_fingerprint) != 64
        or any(char not in "0123456789abcdef" for char in plan_fingerprint)
    ):
        raise ReviewError("Existing decision artifact is malformed or inconsistent")
    if predecessors is not None:
        expected_paths = {record.canonical_path for record in report.records}
        seen_paths: set[str] = set()
        if not isinstance(predecessors, list):
            raise ReviewError("Existing decision artifact is malformed or inconsistent")
        for predecessor in predecessors:
            if not isinstance(predecessor, dict):
                raise ReviewError(
                    "Existing decision artifact is malformed or inconsistent"
                )
            path = predecessor.get("canonical_path")
            digest = predecessor.get("digest")
            archive_path = predecessor.get("archive_path")
            if (
                not isinstance(path, str)
                or path not in expected_paths
                or path in seen_paths
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or archive_path != _predecessor_path(report.submission_id, digest)
            ):
                raise ReviewError(
                    "Existing decision artifact is malformed or inconsistent"
                )
            seen_paths.add(path)
    reviewer = decision.get("reviewer")
    decided_at = decision.get("decided_at")
    issue_number = decision.get("issue_number")
    if (
        decision.get("protocol_version") != PROTOCOL_VERSION
        or decision.get("artifact") != _DECISION_ARTIFACT
        or decision.get("immutable") is not True
        or decision.get("submission_id") != report.submission_id
        or not isinstance(issue_number, int)
        or isinstance(issue_number, bool)
        or issue_number <= 0
        or issue_number != report.issue_number
        or decision.get("records") != expected_records
        or not isinstance(decision_warnings, list)
        or any(not isinstance(warning, str) for warning in decision_warnings)
        or decision_warnings != list(report.warnings)
        or not isinstance(reviewer, str)
        or not reviewer.strip()
        or not isinstance(decided_at, str)
        or not decided_at.strip()
        or not isinstance(decision_reasons, list)
        or any(not isinstance(reason, str) for reason in decision_reasons)
    ):
        raise ReviewError("Existing decision artifact is malformed or inconsistent")
    decision_report = dataclasses.replace(
        report,
        plan_fingerprint=t.cast(
            str, decision.get("canonical_plan_fingerprint", report.plan_fingerprint)
        ),
    )
    expected_content = _decision_bytes(
        report=decision_report,
        outcome=t.cast(str, decision_outcome),
        reviewer=reviewer,
        reasons=t.cast(list[str], decision_reasons),
        decided_at=decided_at,
        predecessors=(
            t.cast(list[dict[str, str | None]], decision["predecessors"])
            if "predecessors" in decision
            else None
        ),
        canonical_plan=t.cast(list[dict[str, object]], canonical_plan)
        if canonical_plan is not None
        else None,
    )
    if content != expected_content:
        raise ReviewError("Existing decision artifact is malformed or inconsistent")


def _plan_fingerprint(
    report: ReviewReport, observations: tuple[CanonicalObservation, ...]
) -> str:
    """Bind the displayed approval plan to immutable staged evidence and observations.

    Args:
        report: Validated staged evidence.
        observations: Destination observations displayed for confirmation.

    Returns:
        SHA-256 fingerprint of the approval plan.
    """
    payload = {
        "submission_id": report.submission_id,
        "records": sorted(
            (record.canonical_path, record.digest) for record in report.records
        ),
        "destinations": sorted(
            (
                item.canonical_path,
                item.predecessor_digest,
                item.replacement,
                item.conflict,
            )
            for item in observations
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _predecessor_path(submission_id: str, digest: str | None) -> str:
    """Return the private content-addressed path for preserved canonical bytes.

    Args:
        submission_id: Submission owning the replacement.
        digest: SHA-256 digest of the predecessor.

    Returns:
        Private staging-bucket archive path.

    Raises:
        ReviewError: If the predecessor digest is missing.
    """
    if digest is None:
        raise ReviewError("Replacement predecessor digest is missing")
    return f"volunteer/predecessors/{submission_id}/{digest}.json"


def _local_reservation(
    reviewer: str, existing: bytes | None
) -> BrokerReservationResult:
    """Create a deterministic local reservation for tests or local use.

    Args:
        reviewer: Reviewer login to bind when no prior decision exists.
        existing: Existing decision bytes, if present.

    Returns:
        A local reservation token plus prior decision metadata when recoverable;
        otherwise the supplied reviewer and fixed epoch timestamp.
    """
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
                decision_digest=_digest(existing),
            )
    return BrokerReservationResult(
        token="local-test-reservation",
        decision_reviewer=reviewer,
        decision_created_at=_LOCAL_DECISION_CREATED_AT,
    )


def _promotion_evidence(report: ReviewReport) -> list[dict[str, str]]:
    """Serialise immutable submission evidence for broker comparisons.

    Args:
        report: Independently validated staged results.

    Returns:
        Identity-sorted canonical paths and digests, including private canaries.
    """
    return sorted(
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


def _restore_decision_plan(
    store: BucketStore, report: ReviewReport, decision: JsonObject, results_bucket: str
) -> ReviewReport:
    """Restore immutable destination expectations from a durable decision.

    Args:
        store: Durable bucket store.
        report: Current staged report.
        decision: Validated durable decision.
        results_bucket: Canonical results bucket.

    Returns:
        Report whose observations represent the original decision plan.

    Raises:
        ReviewError: If current canonical state violates the persisted plan.
    """
    plan = decision.get("canonical_plan")
    if not isinstance(plan, list):
        if any(
            item.replacement or item.conflict for item in report.canonical_observations
        ):
            raise ReviewError(
                "Legacy decision cannot adopt a changed canonical destination"
            )
        return report
    records = {record.canonical_path: record for record in report.records}
    observations: list[CanonicalObservation] = []
    for item in plan:
        if not isinstance(item, dict):
            raise ReviewError("Persisted canonical plan is malformed")
        path = item["canonical_path"]
        record = records[path]
        digest = item["predecessor_digest"]
        replacement = item["disposition"] == "replace"
        archive_path = (
            _predecessor_path(report.submission_id, digest) if replacement else None
        )
        archived = (
            store.read_optional(store.staging_bucket, archive_path)
            if archive_path
            else None
        )
        current = store.read_optional(results_bucket, path)
        if digest is None:
            predecessor = None
            if current not in {None, record.content}:
                raise ReviewError("Canonical destination changed during promotion")
        elif not replacement:
            predecessor = record.content
            if current != predecessor:
                raise ReviewError("Canonical destination changed during promotion")
        else:
            if archived is not None and _digest(archived) != digest:
                raise ReviewError("Archived canonical predecessor is corrupted")
            if archived is None and (current is None or _digest(current) != digest):
                raise ReviewError(
                    "Replacement archive is missing after canonical replacement"
                )
            predecessor = archived if archived is not None else current
            if current not in {predecessor, record.content}:
                raise ReviewError("Canonical destination changed during promotion")
        observations.append(
            CanonicalObservation(
                identity=record.identity,
                canonical_path=path,
                predecessor_digest=digest,
                predecessor_content=predecessor,
                replacement=replacement,
            )
        )
    return dataclasses.replace(
        report,
        canonical_observations=tuple(observations),
        plan_fingerprint=t.cast(
            str, decision.get("canonical_plan_fingerprint", report.plan_fingerprint)
        ),
    )


def _validate_predecessor_archives(
    store: BucketStore,
    report: ReviewReport,
    decision: JsonObject,
    results_bucket: str,
    completed: bool = False,
) -> None:
    """Verify durable replacement evidence against private archives and destinations.

    Args:
        store: Durable staging and canonical object store.
        report: Current validated submission evidence.
        decision: Previously persisted decision object.
        results_bucket: Canonical results bucket.
        completed (optional): Require archives independently of current destinations.

    Raises:
        ReviewError: If predecessor evidence or canonical state is inconsistent.
    """
    predecessors = decision.get("predecessors", [])
    if not isinstance(predecessors, list):
        raise ReviewError("Persisted replacement plan is malformed")
    records = {record.canonical_path: record for record in report.records}
    seen: set[str] = set()
    for item in predecessors:
        if not isinstance(item, dict):
            raise ReviewError("Persisted replacement plan is malformed")
        path = item.get("canonical_path")
        digest = item.get("digest")
        archive_path = item.get("archive_path")
        if (
            not isinstance(path, str)
            or path in seen
            or path not in records
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or archive_path != _predecessor_path(report.submission_id, digest)
        ):
            raise ReviewError("Persisted replacement plan is malformed")
        seen.add(path)
        archived = store.read_optional(store.staging_bucket, archive_path)
        if completed:
            if archived is None:
                raise ReviewError("Replacement archive is missing after completion")
            if _digest(archived) != digest:
                raise ReviewError("Archived canonical predecessor is corrupted")
            continue
        current = store.read_optional(results_bucket, path)
        if archived is not None:
            if _digest(archived) != digest:
                raise ReviewError("Archived canonical predecessor is corrupted")
            if current not in {archived, records[path].content}:
                raise ReviewError("Canonical destination changed during promotion")
        elif current is None or _digest(current) != digest:
            raise ReviewError(
                "Replacement archive is missing after canonical replacement"
            )


def bind_with_broker(
    issue_number: int,
    submission_id: str,
    outcome: str,
    reservation_token: str,
    decision_digest: str,
    records: list[dict[str, str]],
    publication_owner: str | None = None,
    publication_bucket: str | None = None,
) -> str:
    """Atomically bind a decision digest to an active broker reservation.

    Args:
            issue_number:
                Issue number.
            submission_id:
                Submission id.
            outcome:
                Outcome.
            reservation_token:
                Reservation token.
            decision_digest:
                Decision digest.
            records:
                Records.
            publication_owner (optional):
                Owner of the active canonical publication session.
            publication_bucket (optional):
                Bucket containing the active canonical publication session.

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
        publication_owner=publication_owner,
        publication_bucket=publication_bucket,
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
    publication_owner: str | None = None,
    publication_bucket: str | None = None,
) -> str | BrokerReservationResult:
    """Request a reservation, renewal, or decision-digest binding from the broker.

    Args:
        issue_number: GitHub issue being decided.
        submission_id: Submission being decided.
        outcome: Requested terminal outcome.
        records: Canonical identity/path/digest evidence sent to the broker.
        reservation_token (optional):
            Existing token for renewal or binding; omitted to reserve.
        reviewer (optional): Reviewer login included when creating a reservation.
        decision_digest (optional): Decision artifact digest to bind or renew.
        publication_owner (optional): Owner of the active canonical publication session.
        publication_bucket (optional): Bucket containing the active publication session.

    Returns:
        A reservation result for a new reservation, or the unchanged token string
        for a renewal/binding request.

    Raises:
        ReviewError: If credentials, broker response, or required metadata is invalid.
    """
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
    if outcome == "accepted":
        if not publication_owner or not publication_bucket:
            raise ReviewError(
                "Accepted reservations require publication session ownership"
            )
        request_body["publication_owner"] = publication_owner
        request_body["publication_bucket"] = publication_bucket
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
    response_digest = body.get("decision_digest") if isinstance(body, dict) else None
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
        decision_digest=response_digest if isinstance(response_digest, str) else None,
        status=body.get("status", "reserved"),
    )


def inspect_with_broker(
    issue_number: int,
    submission_id: str,
    outcome: str,
    records: list[dict[str, str]],
    decision_digest: str,
) -> BrokerStatus:
    """Read matching broker promotion status without changing its reservation.

    Args:
        issue_number: Evaluation request owning the submission.
        submission_id: Submission whose decision is inspected.
        outcome: Durable decision outcome.
        records: Immutable identity/path/digest evidence bound to the decision.
        decision_digest: Digest of the immutable decision bytes.

    Returns:
        Absent, reserved, or terminal status for the supplied evidence.

    Raises:
        ReviewError: If credentials, transport, protocol, or evidence are invalid.
    """
    secret = os.environ.get("VOLUNTEER_PROMOTION_SECRET")
    if not secret:
        raise ReviewError("VOLUNTEER_PROMOTION_SECRET is required")
    endpoint = os.environ.get(
        "VOLUNTEER_BROKER_PROMOTION_STATUS_URL",
        "https://euroeval.com/api/worker/promotion-status",
    )
    payload = json.dumps(
        {
            "protocol_version": PROTOCOL_VERSION,
            "issue_number": issue_number,
            "submission_id": submission_id,
            "outcome": outcome,
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
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body: object = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as error:
        raise ReviewError("Broker promotion status lookup failed") from error
    if not isinstance(body, dict) or body.get("protocol_version") != PROTOCOL_VERSION:
        raise ReviewError("Broker promotion status protocol differs")
    status = body.get("status")
    if status not in ("absent", "reserved", "terminal"):
        raise ReviewError("Broker returned an invalid promotion status")
    if status != "absent" and body.get("decision_digest") != decision_digest:
        raise ReviewError("Broker promotion status decision evidence differs")
    return t.cast(BrokerStatus, status)


def promote_with_broker(
    issue_number: int,
    submission_id: str,
    outcome: str,
    reservation_token: str,
    records: list[dict[str, str]],
    decision_digest: str,
    publication_owner: str | None = None,
    publication_bucket: str | None = None,
) -> None:
    """Call the authenticated broker promotion transition.

    Args:
            issue_number:
                Issue number.
            submission_id:
                Submission id.
            outcome:
                Outcome.
            reservation_token:
                Reservation token.
            records:
                Immutable canonical identity/path/digest evidence.
            decision_digest:
                Digest of the immutable decision artifact.
            publication_owner (optional):
                Owner of the active canonical publication session.
            publication_bucket (optional):
                Bucket containing the active canonical publication session.

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
            **(
                {
                    "publication_owner": publication_owner,
                    "publication_bucket": publication_bucket,
                }
                if outcome == "accepted" and publication_owner and publication_bucket
                else {}
            ),
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
    publication_owner: str | None = None,
    publication_bucket: str | None = None,
) -> str:
    """Renew a pending broker reservation without changing its token.

    Args:
            issue_number:
                Issue number.
            submission_id:
                Submission id.
            outcome:
                Outcome.
            reservation_token:
                Reservation token.
            records:
                Records.
            decision_digest (optional):
                Digest of the immutable decision artifact.
            publication_owner (optional):
                Owner of the active canonical publication session.
            publication_bucket (optional):
                Bucket containing the active publication session.

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
        publication_owner=publication_owner,
        publication_bucket=publication_bucket,
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
    decision_digest: str | None = None,
    publication_owner: str | None = None,
    publication_bucket: str | None = None,
) -> BrokerReservationResult:
    """Reserve one immutable maintainer outcome at the broker.

    Args:
            issue_number:
                Issue number.
            submission_id:
                Submission id.
            outcome:
                Outcome.
            reviewer:
                Reviewer.
            records:
                Records.
            decision_digest (optional):
                Digest of a previously stored immutable decision.
            publication_owner (optional):
                Owner of the active canonical publication session.
            publication_bucket (optional):
                Bucket containing the active publication session.

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
        decision_digest=decision_digest,
        publication_owner=publication_owner,
        publication_bucket=publication_bucket,
    )
    if not isinstance(reservation, BrokerReservationResult):
        raise ReviewError("Broker returned an invalid promotion reservation")
    return reservation
