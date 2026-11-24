"""Tests for the public volunteer review presentation and collector workflow."""

import json
import shlex
from unittest.mock import Mock

import pytest

from euroeval_worker.review import VolunteerReviewer
from euroeval_worker.review_models import (
    CanonicalObservation,
    ReviewReport,
    ValidatedRecord,
)
from src.scripts import collect_evaluation_results, review_volunteer_results


@pytest.fixture
def report() -> ReviewReport:
    """Return a submission with a benchmark collision and auxiliary canary.

    Returns:
        A byte-backed report suitable for presentation without network access.
    """
    identity = ("model", "task", True, True)
    path = "model/task__val__fewshot.json"
    stored = json.dumps(
        {
            "eval_library": {"version": "1.0"},
            "evaluation_results": [
                {"evaluation_name": "test_f1", "score_details": {"score": 40.0}}
            ],
        }
    ).encode()
    return ReviewReport(
        submission_id="submission with spaces",
        issue_number=12,
        contributor="alice",
        model_id="model",
        model_revision="revision",
        language="da",
        euroeval_version="2.0",
        provenance={"worker_version": "worker-version"},
        expected_identities=(identity,),
        records=(
            ValidatedRecord(
                identity=identity,
                digest="submitted-digest",
                staged_path="staged",
                canonical_path=path,
                content=b"{}",
                scores=(("test_f1", 50.0),),
                warnings=("Missing metadata",),
            ),
            ValidatedRecord(
                identity=("model", "contamination-canary-da", None, None),
                digest="canary-digest",
                staged_path="canary",
                canonical_path="canary",
                content=b"{}",
                scores=(("test_collection_success", 0.0),),
                warnings=("Missing metadata",),
            ),
        ),
        canonical_observations=(
            CanonicalObservation(
                identity=identity,
                canonical_path=path,
                predecessor_digest="stored-digest",
                predecessor_content=stored,
                replacement=False,
                conflict=(
                    "Result conflict requires an explicit choice: unresolved_difference"
                ),
            ),
        ),
        plan_fingerprint="fingerprint",
        warnings=("Missing metadata",),
        checks=("schema validated",),
    )


def test_blocked_preflight_skips_without_approval_or_collection_error(
    report: ReviewReport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A collision stays pending without an impossible approval prompt."""
    reviewer = Mock(spec=VolunteerReviewer)
    reviewer.list_pending_submissions.return_value = [
        (report.submission_id, report.issue_number, report.contributor, report.language)
    ]
    reviewer.show.return_value = report
    monkeypatch.setenv("HF_STAGING_BUCKET", "private/staging")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_ACTOR", "maintainer")
    monkeypatch.setattr(
        collect_evaluation_results, "reviewer_from_environment", lambda: reviewer
    )
    monkeypatch.setattr(
        collect_evaluation_results, "_stdin_is_interactive", lambda: True
    )
    prompt = Mock(side_effect=["y"])
    monkeypatch.setattr("builtins.input", prompt)
    decision = Mock()
    monkeypatch.setattr(collect_evaluation_results, "run_review_command", decision)

    assert collect_evaluation_results.preflight_volunteer_review() is True
    assert prompt.call_count == 1
    decision.assert_not_called()
    reviewer.decide.assert_not_called()


def test_summary_explains_conflicts_and_supported_actions(
    report: ReviewReport, caplog: pytest.LogCaptureFixture
) -> None:
    """Default presentation compares evidence and supplies valid shell commands."""
    reviewer = Mock(spec=VolunteerReviewer)
    reviewer.show.return_value = report
    caplog.set_level("INFO")

    assert (
        review_volunteer_results.show_and_render(
            submission_id=report.submission_id, reviewer=reviewer
        )
        == report
    )

    output = caplog.text
    assert "40.0" in output and "50" in output
    assert "1.0" in output and "2.0" in output
    assert output.count("Missing metadata") == 1
    assert "1 expected, 1 submitted" in output
    assert "BLOCKED" in output
    assert "ENTIRE submission" in output
    assert "not supported" in output
    assert "submitted-digest" not in output
    commands = [
        shlex.split(record.getMessage().split(": ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("Inspect full evidence:")
    ]
    assert commands == [
        [
            "uv",
            "run",
            "python",
            "src/scripts/review_volunteer_results.py",
            "show",
            report.submission_id,
            "--verbose",
        ]
    ]
    reviewer.decide.assert_not_called()


def test_verbose_cli_retains_audit_evidence(
    report: ReviewReport,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The public CLI exposes digests, checks and provenance on request."""
    reviewer = Mock(spec=VolunteerReviewer)
    reviewer.show.return_value = report
    monkeypatch.setattr(
        review_volunteer_results, "reviewer_from_environment", lambda: reviewer
    )
    caplog.set_level("INFO")

    assert (
        review_volunteer_results.main(["show", report.submission_id, "--verbose"]) == 0
    )
    assert "submitted-digest" in caplog.text
    assert "canary-digest" in caplog.text
    assert "schema validated" in caplog.text
    assert "worker-version" in caplog.text
    assert report.plan_fingerprint in caplog.text
    reviewer.decide.assert_not_called()
