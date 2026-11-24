"""Review and promote volunteer evaluation results from private staging."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys

import click
from dotenv import load_dotenv

from euroeval_worker.review import (
    ReviewError,
    ReviewReport,
    VolunteerReviewer,
    reviewer_from_environment,
)
from euroeval_worker.review_models import ValidatedRecord
from leaderboards.record_fields import get_num_failed_instances
from leaderboards.score_extraction import get_scored_count

logger = logging.getLogger(__name__)
_USE_COLOUR = False


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
    configure_colours(no_colour=arguments.no_colour)
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
            report = service.show(
                submission_id=arguments.submission_id,
                replace_existing=arguments.replace_existing,
            )
            _log_report(report, verbose=arguments.verbose)
            return 0
        outcome = "accepted" if arguments.command == "approve" else "rejected"
        report = service.decide(
            submission_id=arguments.submission_id,
            outcome=outcome,
            reviewer=reviewer,
            reasons=arguments.reason,
            plan_fingerprint=arguments.plan_fingerprint,
            replace_existing=arguments.replace_existing,
        )
        if not arguments.quiet:
            logger.info(
                "%s %s (%s)",
                outcome.capitalize(),
                report.model_id,
                report.language,
                extra={"review_success": True},
            )
        return 0
    except ReviewError as error:
        logger.error("Review failed: %s", error)
        return 1


def _log_report(report: ReviewReport, *, verbose: bool = False) -> None:
    """Render a five-line summary, with audit evidence only on request.

    Args:
        report: Validated submission evidence.
        verbose: Include full evidence outside the compact summary.
    """
    render_model_review(reports=[report])
    if not verbose:
        return
    records = [record for record in report.records if not _is_canary(record)]
    warnings = {w.removesuffix(" (informational)") for w in report.warnings}
    warnings.update(
        w.removesuffix(" (informational)") for r in report.records for w in r.warnings
    )
    logger.info("Submission: %s", report.submission_id)
    logger.info("Model revision: %s", report.model_revision)
    logger.info("Provenance: %s", report.provenance)
    for record in records:
        logger.info("%s: %s", record.identity[1], dict(record.scores))
    for item in report.canonical_observations:
        if not item.conflict and not item.replacement:
            continue
        old = json.loads(item.predecessor_content or b"{}")
        logger.info("Stored %s: %s", item.identity[1], old.get("evaluation_results"))
        if item.conflict:
            logger.info("Needs review: %s", item.conflict)
    for warning in sorted(warnings):
        logger.info("Notice: %s", warning)
    for check in report.checks:
        logger.info("Check: %s", check)
    for record in report.records:
        logger.info("Record %s sha256=%s", record.identity, record.digest)
    logger.info("Approval plan fingerprint: %s", report.plan_fingerprint)


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


def render_model_review(reports: list[ReviewReport]) -> None:
    """Show all submitted languages of one model in a five-line box.

    Args:
        reports: Validated submissions for one model.
    """
    records = [r for report in reports for r in report.records if not _is_canary(r)]
    flagged: set[str] = set()
    unknown: set[str] = set()
    for record in records:
        data = json.loads(record.content)
        failed = get_num_failed_instances(record=data)
        if failed is None:
            unknown.add(record.identity[1])
        elif failed > 0:
            total = get_scored_count(record=data, dataset=record.identity[1])
            if total is None or total <= 0:
                unknown.add(record.identity[1])
            elif failed / total >= 0.1:
                flagged.add(record.identity[1])
    differences = len(
        {
            o.canonical_path
            for report in reports
            for o in report.canonical_observations
            if o.conflict or o.replacement
        }
    )
    languages = ", ".join(sorted({r.language for r in reports}))
    contributors = ", ".join(sorted({r.contributor for r in reports}))
    versions = ", ".join(sorted({r.euroeval_version for r in reports}))
    expected = sum(len(report.expected_identities) for report in reports)
    dataset_text = f"{len(records)}/{expected} dataset results"
    failure_text = f"{len(flagged)} datasets with ≥10% failures"
    if unknown:
        failure_text += f" ({len(unknown)} unknown)"
    existing = (
        f"Approve replaces {differences} existing results (archived)"
        if differences
        else "Approve adds submitted results"
    )
    lines = [
        f"{reports[0].model_id} | Languages: {languages}",
        f"{contributors} | EuroEval {versions} | {dataset_text}",
        f"{failure_text} | {existing}",
    ]
    width = max(36, min(120, shutil.get_terminal_size().columns)) - 4
    logger.info("┌%s┐", "─" * (width + 2))
    for line in lines:
        clipped = line if len(line) <= width else line[: width - 1] + "…"
        padded = f"{clipped:<{width}}"
        if dataset_text in padded:
            padded = padded.replace(
                dataset_text,
                style_status(dataset_text, success=len(records) == expected),
            )
        if failure_text in padded:
            padded = padded.replace(
                failure_text, style_status(failure_text, success=not flagged)
            )
        logger.info("│ %s │", padded)
    logger.info("└%s┘", "─" * (width + 2))


def style_status(text: str, *, success: bool) -> str:
    """Apply muted green or red when terminal colours are enabled.

    Args:
        text: Visible status or action text.
        success: Use green for success, red for failure.

    Returns:
        Styled text, or unchanged text when colours are disabled.
    """
    if not _USE_COLOUR:
        return text
    return click.style(text, fg="green" if success else "red", dim=True)


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser for listing and reviewing submissions.

    Returns:
        Parser with list, show, approve, and reject subcommands.
    """
    parser = argparse.ArgumentParser(
        description="Review durable volunteer results without consulting Redis."
    )
    parser.add_argument(
        "--no-colour", action="store_true", help="Disable coloured output."
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
    show.add_argument(
        "--replace-existing",
        action="store_true",
        help="Preview selecting submitted results over differing stored records.",
    )
    for command in ("approve", "reject"):
        decision = subparsers.add_parser(
            command, help=f"Revalidate and {command} a submission."
        )
        decision.add_argument("submission_id")
        decision.add_argument(
            "--quiet", action="store_true", help="Suppress decision receipt."
        )
        if command == "approve":
            decision.add_argument(
                "--plan-fingerprint",
                help="Fingerprint displayed by show; required for acceptance.",
            )
            decision.add_argument(
                "--replace-existing",
                action="store_true",
                help="Select submitted results; requires matching preview fingerprint.",
            )
        else:
            decision.set_defaults(plan_fingerprint=None, replace_existing=False)
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


def configure_colours(*, no_colour: bool = False) -> None:
    """Configure muted status colours for terminal output only.

    Args:
        no_colour: Disable colours, including downstream subprocess output.
    """
    global _USE_COLOUR
    if no_colour:
        os.environ["NO_COLOR"] = "1"
    _USE_COLOUR = not no_colour and "NO_COLOR" not in os.environ and sys.stderr.isatty()
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(_ReviewFormatter("%(message)s"))


class _ReviewFormatter(logging.Formatter):
    """Colour errors red and completed decisions green, without bright colours."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a status message, leaving ordinary output neutral.

        Args:
            record: Event emitted by the collector or reviewer.

        Returns:
            Terminal text with optional muted status colour.
        """
        message = super().format(record)
        if not _USE_COLOUR:
            return message
        if record.levelno >= logging.ERROR:
            return style_status(message, success=False)
        if getattr(record, "review_success", False):
            return style_status(message, success=True)
        return message


def show_and_render(
    submission_id: str,
    reviewer: VolunteerReviewer | None = None,
    *,
    verbose: bool = False,
    replace_existing: bool = False,
) -> ReviewReport:
    """Load, render, and return the exact report shown to a reviewer.

    Args:
        submission_id: Submission to display.
        reviewer (optional): Reviewer adapter; defaults to environment configuration.
        verbose: Whether to include record-level validation details.
        replace_existing: Preview the submitted replacement plan.

    Returns:
        The rendered report, including its confirmation fingerprint.
    """
    service = reviewer or _service()
    report = (
        service.show(submission_id=submission_id, replace_existing=True)
        if replace_existing
        else service.show(submission_id=submission_id)
    )
    _log_report(report, verbose=verbose)
    return report


if __name__ == "__main__":
    raise SystemExit(main())
