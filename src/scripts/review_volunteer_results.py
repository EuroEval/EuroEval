"""Review and promote volunteer evaluation results from private staging."""

from __future__ import annotations

import argparse
import logging
import os
import typing as t
from collections import defaultdict

from dotenv import load_dotenv
from huggingface_hub import HfApi

from euroeval_worker.review import (
    BucketApi,
    BucketStore,
    ReviewError,
    ReviewReport,
    VolunteerReviewer,
)
from leaderboards.constants import HF_RESULTS_BUCKET

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the maintainer review command.

    Returns:
        Process exit status.

    Raises:
        ReviewError:
            If required review configuration is absent.
    """
    load_dotenv()
    arguments = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        reviewer = arguments.reviewer or os.environ.get("GITHUB_ACTOR", "")
        if arguments.command in {"approve", "reject"} and not reviewer:
            raise ReviewError("Set --reviewer or GITHUB_ACTOR to a verified login")
        service = _service()
        if arguments.command == "list":
            for (
                submission_id,
                issue,
                contributor,
                language,
            ) in service.list_submissions():
                logger.info(
                    "%s  issue=%s  contributor=%s  language=%s",
                    submission_id,
                    issue,
                    contributor,
                    language,
                )
            return 0
        if arguments.command == "show":
            _log_report(service.show(submission_id=arguments.submission_id))
            return 0
        outcome = "accepted" if arguments.command == "approve" else "rejected"
        report = service.decide(
            submission_id=arguments.submission_id,
            outcome=outcome,
            reviewer=reviewer,
            reasons=arguments.reason,
        )
        logger.info(
            "%s submission %s for issue %s",
            outcome.capitalize(),
            report.submission_id,
            report.issue_number,
        )
        return 0
    except ReviewError as error:
        logger.error("Review failed: %s", error)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review durable volunteer results without consulting Redis."
    )
    parser.add_argument(
        "--reviewer",
        help="Verified maintainer GitHub login (defaults to GITHUB_ACTOR).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List durable staged manifests.")
    show = subparsers.add_parser("show", help="Revalidate and summarise a submission.")
    show.add_argument("submission_id")
    for command in ("approve", "reject"):
        decision = subparsers.add_parser(
            command, help=f"Revalidate and {command} a submission."
        )
        decision.add_argument("submission_id")
        decision.add_argument(
            "--reason",
            action="append",
            default=[],
            help="Review reason; may be repeated.",
        )
    return parser


def _service() -> VolunteerReviewer:
    staging = os.environ.get("HF_STAGING_BUCKET")
    token = os.environ.get("HF_TOKEN")
    if not staging or not token:
        raise ReviewError("HF_STAGING_BUCKET and HF_TOKEN are required")
    return VolunteerReviewer(
        store=BucketStore(
            api=t.cast(BucketApi, HfApi(token=token)),
            token=token,
            staging_bucket=staging,
        ),
        results_bucket=os.environ.get("HF_RESULTS_BUCKET", HF_RESULTS_BUCKET),
    )


def _log_report(report: ReviewReport) -> None:
    logger.info("Submission: %s", report.submission_id)
    logger.info("Issue: %s", report.issue_number)
    logger.info("Verified contributor: %s", report.contributor)
    logger.info("Model: %s@%s", report.model_id, report.model_revision)
    logger.info("Language: %s", report.language)
    logger.info("EuroEval version: %s", report.euroeval_version)
    if report.provenance:
        logger.info("Provenance: %s", report.provenance)
    logger.info("Expected identities: %s", len(report.expected_identities))
    for identity in report.expected_identities:
        logger.info("  expected %s", identity)
    logger.info("Actual identities: %s", len(report.records))
    for record in report.records:
        logger.info("  actual   %s  sha256=%s", record.identity, record.digest)
    by_metric: dict[str, list[float]] = defaultdict(list)
    for record in report.records:
        for metric, score in record.scores:
            by_metric[metric].append(score)
    for metric, scores in sorted(by_metric.items()):
        logger.info("Score range %s: %g to %g", metric, min(scores), max(scores))
    logger.info("Checks: %s", "; ".join(report.checks))
    logger.info("Warnings: %s", "; ".join(report.warnings) or "none")


if __name__ == "__main__":
    raise SystemExit(main())
