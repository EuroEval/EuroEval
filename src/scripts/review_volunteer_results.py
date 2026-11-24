"""Review and promote volunteer evaluation results from private staging."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex

from dotenv import load_dotenv

from euroeval_worker.review import (
    ReviewError,
    ReviewReport,
    VolunteerReviewer,
    reviewer_from_environment,
)
from euroeval_worker.review_models import ValidatedRecord

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the maintainer review command.

    Args:
        argv (optional): Optional command-line arguments; defaults to ``None``, which
            uses ``sys.argv``.

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
            submissions = (
                service.list_submissions()
                if arguments.all
                else service.list_pending_submissions()
            )
            logger.info(
                "%s volunteer submissions:",
                "All durable" if arguments.all else "Pending",
            )
            if not submissions:
                logger.info(
                    "No %s volunteer submissions found.",
                    "durable" if arguments.all else "pending",
                )
            for submission_id, issue, contributor, language in submissions:
                logger.info(
                    "%s  issue=%s  contributor=%s  language=%s",
                    submission_id,
                    issue,
                    contributor,
                    language,
                )
            return 0
        if arguments.command == "show":
            report = service.show(submission_id=arguments.submission_id)
            _log_report(report, verbose=arguments.verbose)
            return 0
        outcome = "accepted" if arguments.command == "approve" else "rejected"
        report = service.decide(
            submission_id=arguments.submission_id,
            outcome=outcome,
            reviewer=reviewer,
            reasons=arguments.reason,
            plan_fingerprint=arguments.plan_fingerprint,
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


def _log_report(report: ReviewReport, *, verbose: bool = False) -> None:
    """Log a concise summary, or full validation detail when requested.

    Args:
        report: Validated volunteer submission report to display.
        verbose: Whether to include record-level checks and provenance.
    """
    logger.info("Submission: %s", report.submission_id)
    logger.info("Issue: %s", report.issue_number)
    logger.info("Verified contributor: %s", report.contributor)
    logger.info("Model: %s@%s", report.model_id, report.model_revision)
    logger.info("Language: %s", report.language)
    logger.info("EuroEval version: %s", report.euroeval_version)
    logger.info(
        "Benchmark records: %s expected, %s submitted",
        len(report.expected_identities),
        sum(not _is_canary(record) for record in report.records),
    )
    for record in report.records:
        if _is_canary(record):
            continue
        scores = ", ".join(f"{metric}={score:g}" for metric, score in record.scores)
        logger.info("Result %s  scores={%s}", record.identity, scores)
    if verbose:
        if report.provenance:
            logger.info("Provenance: %s", report.provenance)
        for check in report.checks:
            logger.info("Check: %s", check)
    for observation in report.canonical_observations:
        if observation.replacement or observation.conflict:
            staged = next(
                record
                for record in report.records
                if record.canonical_path == observation.canonical_path
            )
            old = json.loads(observation.predecessor_content or b"{}")
            library = old.get("eval_library", {})
            old_version = (
                library.get("version", "unknown")
                if isinstance(library, dict)
                else "unknown"
            )
            old_scores = {
                item.get("evaluation_name", "unknown"): item.get(
                    "score_details", {}
                ).get("score")
                for item in old.get("evaluation_results", [])
                if isinstance(item, dict)
                and isinstance(item.get("score_details"), dict)
            }
            logger.info(
                "%s %s: stored version=%s scores=%s; submitted version=%s scores=%s",
                "CONFLICT" if observation.conflict else "REPLACEMENT",
                observation.identity[1],
                old_version,
                old_scores,
                report.euroeval_version,
                dict(staged.scores),
            )
    blocked = [item for item in report.canonical_observations if item.conflict]
    for observation in blocked:
        logger.error(
            "BLOCKED %s: stored canonical result differs from submitted result; "
            "there is no automatic winner and approval cannot override this with y.",
            observation.canonical_path,
        )
    command = ["uv", "run", "python", "src/scripts/review_volunteer_results.py"]
    if blocked:
        logger.info(
            "Status: BLOCKED (%s conflicts). Left pending; nothing approved.",
            len(blocked),
        )
        logger.info(
            "Inspect full evidence: %s",
            shlex.join(command + ["show", report.submission_id, "--verbose"]),
        )
        logger.info(
            "To keep existing results and discard this ENTIRE submission, review "
            "the evidence first, then explicitly reject it: %s",
            shlex.join(
                command
                + [
                    "--reviewer",
                    os.environ.get("GITHUB_ACTOR") or "<github-login>",
                    "reject",
                    report.submission_id,
                    "--reason",
                    "keep existing canonical results",
                ]
            ),
        )
        logger.info(
            "Replacing existing results with this submission is not supported by "
            "the review CLI. Leave it pending if you want the submitted results; "
            "do not edit the bucket manually."
        )
    else:
        logger.info(
            "Status: no approval blockers. Approval still requires confirmation."
        )
    if verbose:
        logger.info("Approval plan fingerprint: %s", report.plan_fingerprint)
    warnings = set(report.warnings)
    for record in report.records:
        warnings.update(record.warnings)
    if warnings:
        logger.info("Non-blocking notices:")
        for warning in sorted(warnings):
            logger.info("  %s", warning)
    if verbose:
        for record in report.records:
            logger.info("Record %s sha256=%s", record.identity, record.digest)
    if not blocked and any(item.replacement for item in report.canonical_observations):
        logger.info("Existing results selected for replacement will be archived.")


def _is_canary(record: ValidatedRecord) -> bool:
    """Return whether a staged record is an auxiliary canary result.

    Args:
        record: Validated staged record.

    Returns:
        Whether the record is a canary.
    """
    dataset = record.identity[1]
    return dataset == "contamination-canary" or dataset.startswith(
        "contamination-canary-"
    )


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser for listing and reviewing submissions.

    Returns:
        Parser with list, show, approve, and reject subcommands.
    """
    parser = argparse.ArgumentParser(
        description="Review durable volunteer results without consulting Redis."
    )
    parser.add_argument(
        "--reviewer",
        help="Verified maintainer GitHub login (defaults to GITHUB_ACTOR).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    listing = subparsers.add_parser(
        "list", help="List pending staged manifests (use --all for history)."
    )
    listing.add_argument(
        "--all",
        action="store_true",
        help="Include submissions with terminal decisions.",
    )
    show = subparsers.add_parser("show", help="Revalidate and summarise a submission.")
    show.add_argument("submission_id")
    show.add_argument("--verbose", action="store_true", help="Show validation details.")
    for command in ("approve", "reject"):
        decision = subparsers.add_parser(
            command, help=f"Revalidate and {command} a submission."
        )
        decision.add_argument("submission_id")
        if command == "approve":
            decision.add_argument(
                "--plan-fingerprint",
                help="Fingerprint displayed by show; required for acceptance.",
            )
        else:
            decision.set_defaults(plan_fingerprint=None)
        decision.add_argument(
            "--reason",
            action="append",
            default=[],
            help="Review reason; may be repeated.",
        )
    return parser


def _service() -> VolunteerReviewer:
    """Return the reviewer service configured from the environment.

    Returns:
        A reviewer service backed by the configured durable submission store.
    """
    return reviewer_from_environment()


def show_and_render(
    submission_id: str,
    reviewer: VolunteerReviewer | None = None,
    *,
    verbose: bool = False,
) -> ReviewReport:
    """Load, render, and return the exact report shown to a reviewer.

    Args:
        submission_id: Submission to display.
        reviewer (optional): Reviewer adapter; defaults to environment configuration.
        verbose: Whether to include record-level validation details.

    Returns:
        The rendered report, including its confirmation fingerprint.
    """
    report = (reviewer or _service()).show(submission_id=submission_id)
    _log_report(report, verbose=verbose)
    return report


if __name__ == "__main__":
    raise SystemExit(main())
