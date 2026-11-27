"""Tests for the public volunteer review presentation and collector workflow."""

import dataclasses
import io
import json
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest
from click import style
from click.testing import CliRunner

from euroeval_worker.review import VolunteerReviewer
from euroeval_worker.review_models import (
    CanonicalObservation,
    ReviewReport,
    ValidatedRecord,
)
from leaderboards import score_extraction
from src.scripts import collect_evaluation_results, review_volunteer_results


@pytest.fixture(autouse=True)
def isolate_staging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep review approvals out of the working copy's staging file."""
    monkeypatch.setattr(
        collect_evaluation_results, "NEW_RESULTS_PATH", tmp_path / "new_results.jsonl"
    )


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
                content=json.dumps({"model": "model", "dataset": "task"}).encode(),
                scores=(("test_f1", 50.0),),
                warnings=("Missing metadata",),
            ),
            ValidatedRecord(
                identity=("model", "contamination-canary-da", None, None),
                digest="canary-digest",
                staged_path="canary",
                canonical_path="canary",
                content=json.dumps(
                    {
                        "eval_library": {
                            "additional_details": {"dataset": "contamination-canary-da"}
                        }
                    }
                ).encode(),
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


def test_collector_accepts_no_colour_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The collector CLI accepts --no-colour before doing any collection work."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(
        collect_evaluation_results, "check_required_env_vars", lambda: None
    )
    monkeypatch.setattr(
        collect_evaluation_results, "preflight_volunteer_review", lambda: False
    )
    result = CliRunner().invoke(collect_evaluation_results.main, ["--no-colour"])
    assert result.exit_code == 0
    assert "\x1b[" not in result.output


def test_conflicting_preflight_can_be_left_pending(
    report: ReviewReport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conflicts offer a review action rather than silently skipping."""
    reviewer = Mock(spec=VolunteerReviewer)
    reviewer.group_submissions_by_model.side_effect = lambda submission_ids: [
        submission_ids
    ]
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
    prompt = Mock(side_effect=["y", "s"])
    monkeypatch.setattr("builtins.input", prompt)
    decision = Mock()
    monkeypatch.setattr(collect_evaluation_results, "run_review_command", decision)

    assert collect_evaluation_results.preflight_volunteer_review() is True
    assert prompt.call_count == 2
    decision.assert_not_called()
    reviewer.decide.assert_not_called()


@pytest.mark.parametrize(("failed", "flagged"), [(9, 0), (10, 1), (11, 1)])
def test_failure_summary_matches_leaderboard_threshold(
    report: ReviewReport,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failed: int,
    flagged: int,
) -> None:
    """The red-asterisk criterion is at least 10%, using the leaderboard denominator."""
    data = {
        "eval_library": {"additional_details": {"raw_results": "[{}]"}},
        "evaluation_results": [
            {"score_details": {"details": {"num_failed_instances": str(failed)}}}
        ],
    }
    record = dataclasses.replace(report.records[0], content=json.dumps(data).encode())
    report = dataclasses.replace(report, records=(record,))
    monkeypatch.setattr(score_extraction, "dataset_sources", lambda: {"task": "source"})
    monkeypatch.setattr(
        score_extraction, "get_split_sizes", lambda source: {"test": 100, "val": 100}
    )
    caplog.set_level("INFO")
    review_volunteer_results.render_model_review(reports=[report])
    assert f"{flagged} datasets with ≥10% failures" in caplog.text
    assert "unknown" not in caplog.text


@pytest.mark.parametrize("action", ["a", "r"])
def test_model_actions_cover_all_languages(
    report: ReviewReport,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    action: str,
) -> None:
    """One model decision covers every pending language, with no IDs in the box."""
    first = dataclasses.replace(report, canonical_observations=())
    second = dataclasses.replace(first, submission_id="second-language", language="ro")
    reports = {r.submission_id: r for r in [first, second]}
    reviewer = Mock(spec=VolunteerReviewer)
    reviewer.group_submissions_by_model.side_effect = lambda submission_ids: [
        submission_ids
    ]
    reviewer.show.side_effect = lambda **kw: reports[kw["submission_id"]]
    reviewer.list_pending_submissions.return_value = [
        (r.submission_id, 12, "alice", r.language) for r in reports.values()
    ]
    monkeypatch.setenv("HF_STAGING_BUCKET", "private/staging")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_ACTOR", "maintainer")
    monkeypatch.setattr(
        collect_evaluation_results, "reviewer_from_environment", lambda: reviewer
    )
    monkeypatch.setattr(
        collect_evaluation_results, "_stdin_is_interactive", lambda: True
    )
    answers = ["y", "a"] if action == "a" else ["y", "r", "reason", "y"]
    monkeypatch.setattr("builtins.input", Mock(side_effect=answers))
    decision = Mock(return_value=0)
    monkeypatch.setattr(collect_evaluation_results, "run_review_command", decision)
    caplog.set_level("INFO")
    assert collect_evaluation_results.preflight_volunteer_review()
    commands = [c.args[0] for c in decision.call_args_list]
    if action == "a":
        assert commands == []
        staged = collect_evaluation_results.NEW_RESULTS_PATH.read_text().splitlines()
        assert [json.loads(line) for line in staged] == [
            json.loads(first.records[0].content),
            json.loads(second.records[0].content),
        ]
    else:
        assert {c[3] for c in commands} == set(reports)
        assert all("reject" in c and "--quiet" not in c for c in commands)
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith(("┌", "│", "└"))
    ]
    assert len(lines) == 5
    assert "da, ro" in "\n".join(lines)
    assert "submission with spaces" not in "\n".join(lines)
    assert "#12" not in "\n".join(lines)
    progress_output = capsys.readouterr().err
    assert "Loading model" in progress_output
    assert "1/2" in progress_output
    assert "2/2" in progress_output


def test_muted_status_colours_and_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Success/failure use dim green/red; opting out removes terminal escapes."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(review_volunteer_results.sys.stderr, "isatty", lambda: True)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    root.addHandler(handler)
    output = logging.getLogger("review-colour-test")
    output.setLevel(logging.INFO)
    try:
        review_volunteer_results.configure_colours()
        output.info("Completed", extra={"review_success": True})
        output.error("Failed")
        assert "\x1b[32m" in stream.getvalue()
        assert "\x1b[31m" in stream.getvalue()
        assert "\x1b[2m" in stream.getvalue()
        assert "\x1b[92m" not in stream.getvalue()
        assert "\x1b[91m" not in stream.getvalue()
        stream.seek(0)
        stream.truncate()
        review_volunteer_results.configure_colours(no_colour=True)
        output.info("Completed", extra={"review_success": True})
        output.error("Failed")
        assert "\x1b[" not in stream.getvalue()
    finally:
        root.removeHandler(handler)


@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        (["a"], "approve"),
        (["r", "keep existing", "y"], "reject"),
        (["d", "s"], None),
        (["r", "", "s"], None),
        (["r", "reason", "n", "s"], None),
        (["q"], "stop"),
    ],
)
def test_review_actions(
    report: ReviewReport,
    monkeypatch: pytest.MonkeyPatch,
    actions: list[str],
    expected: str | None,
) -> None:
    """Only confirmed decisions issue commands; details and skip are read-only."""
    selected = dataclasses.replace(
        report,
        canonical_observations=tuple(
            dataclasses.replace(o, replacement=True, conflict=None)
            for o in report.canonical_observations
        ),
        plan_fingerprint="selected-fingerprint",
    )
    reviewer = Mock(spec=VolunteerReviewer)
    reviewer.group_submissions_by_model.side_effect = lambda submission_ids: [
        submission_ids
    ]
    reviewer.list_pending_submissions.return_value = [
        (report.submission_id, 12, "alice", "da")
    ]
    reviewer.show.side_effect = lambda **kw: (
        selected if kw.get("replace_existing") else report
    )
    monkeypatch.setenv("HF_STAGING_BUCKET", "private/staging")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_ACTOR", "maintainer")
    monkeypatch.setattr(
        collect_evaluation_results, "reviewer_from_environment", lambda: reviewer
    )
    monkeypatch.setattr(
        collect_evaluation_results, "_stdin_is_interactive", lambda: True
    )
    prompt = Mock(side_effect=["y", *actions])
    monkeypatch.setattr("builtins.input", prompt)
    decision = Mock(return_value=0)
    monkeypatch.setattr(collect_evaluation_results, "run_review_command", decision)
    assert collect_evaluation_results.preflight_volunteer_review() is (
        expected != "stop"
    )
    if expected == "approve":
        decision.assert_not_called()
        staged = collect_evaluation_results.NEW_RESULTS_PATH.read_text().splitlines()
        assert [json.loads(line) for line in staged] == [
            json.loads(report.records[0].content)
        ]
    elif expected == "reject":
        command = decision.call_args.args[0]
        assert "reject" in command
        assert report.submission_id in command
        assert "keep existing" in command
        assert "--quiet" not in command
    else:
        decision.assert_not_called()
        assert not collect_evaluation_results.NEW_RESULTS_PATH.exists()


@pytest.mark.parametrize("no_colour", [False, True])
@pytest.mark.parametrize("expected_count", [1, 2])
@pytest.mark.parametrize("failed", [0, 10])
def test_review_colours_are_visible_before_deciding(
    report: ReviewReport,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    no_colour: bool,
    failed: int,
    expected_count: int,
) -> None:
    """The summary and action menu show muted statuses before any decision occurs."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(collect_evaluation_results.sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr(score_extraction, "dataset_sources", lambda: {"task": "source"})
    monkeypatch.setattr(
        score_extraction, "get_split_sizes", lambda source: {"test": 100, "val": 100}
    )
    data = {
        "eval_library": {"additional_details": {"raw_results": "[{}]"}},
        "evaluation_results": [
            {"score_details": {"details": {"num_failed_instances": str(failed)}}}
        ],
    }
    record = dataclasses.replace(report.records[0], content=json.dumps(data).encode())
    report = dataclasses.replace(report, records=(record,), canonical_observations=())
    if expected_count == 2:
        report = dataclasses.replace(
            report,
            expected_identities=(
                *report.expected_identities,
                ("model", "other-task", True, True),
            ),
        )
    reviewer = Mock(spec=VolunteerReviewer)
    reviewer.list_pending_submissions.return_value = [
        (report.submission_id, 12, "alice", "da")
    ]
    reviewer.group_submissions_by_model.return_value = [[report.submission_id]]
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
    prompt = Mock(side_effect=["y", "s"])
    monkeypatch.setattr("builtins.input", prompt)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    root.addHandler(handler)
    caplog.set_level("INFO")
    try:
        collect_evaluation_results.configure_colours(no_colour=no_colour)
        assert collect_evaluation_results.preflight_volunteer_review()
        menu = prompt.call_args.args[0]
        if no_colour:
            assert "\x1b[" not in stream.getvalue()
            assert "\x1b[" not in menu
        else:
            assert ("\x1b[31m" if failed else "\x1b[32m") in stream.getvalue()
            assert (
                style(
                    f"1/{expected_count} dataset results",
                    fg="green" if expected_count == 1 else "red",
                    dim=True,
                )
                in stream.getvalue()
            )
            assert (
                style(
                    f"{int(failed > 0)} datasets with ≥10% failures",
                    fg="red" if failed else "green",
                    dim=True,
                )
                in stream.getvalue()
            )
            assert "\x1b[32m" in menu and "\x1b[31m" in menu
            assert "\x1b[2m" in menu
            assert "\x1b[91m" not in menu and "\x1b[92m" not in menu
        reviewer.decide.assert_not_called()
    finally:
        root.removeHandler(handler)
        collect_evaluation_results.configure_colours(no_colour=True)


def test_stopping_first_model_does_not_inspect_later_models(
    report: ReviewReport,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Stopping review leaves a later model's unavailable evidence untouched."""
    reviewer = Mock(spec=VolunteerReviewer)
    reviewer.list_pending_submissions.return_value = [
        (report.submission_id, 12, "alice", "da"),
        ("unavailable-model", 13, "alice", "ro"),
    ]
    reviewer.group_submissions_by_model.return_value = [
        [report.submission_id],
        ["unavailable-model"],
    ]
    reviewer.show.side_effect = [report, RuntimeError("Evidence unavailable")]
    monkeypatch.setenv("HF_STAGING_BUCKET", "private/staging")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_ACTOR", "maintainer")
    monkeypatch.setattr(
        collect_evaluation_results, "reviewer_from_environment", lambda: reviewer
    )
    monkeypatch.setattr(
        collect_evaluation_results, "_stdin_is_interactive", lambda: True
    )
    monkeypatch.setattr("builtins.input", Mock(side_effect=["y", "q"]))
    decision = Mock()
    monkeypatch.setattr(collect_evaluation_results, "run_review_command", decision)
    assert collect_evaluation_results.preflight_volunteer_review() is False
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)
    decision.assert_not_called()


@pytest.mark.parametrize("long_values", [False, True])
def test_summary_fits_five_lines(
    report: ReviewReport, caplog: pytest.LogCaptureFixture, long_values: bool
) -> None:
    """The default summary is readable and bounded even for long identifiers."""
    if long_values:
        report = dataclasses.replace(report, model_id="model" * 100)
    reviewer = Mock(spec=VolunteerReviewer)
    reviewer.group_submissions_by_model.side_effect = lambda submission_ids: [
        submission_ids
    ]
    reviewer.show.return_value = report
    caplog.set_level("INFO")
    review_volunteer_results.show_and_render(
        submission_id=report.submission_id, reviewer=reviewer
    )
    lines = [record.getMessage() for record in caplog.records]
    assert len(lines) == 5
    assert all(len(line) <= 120 for line in lines)
    assert "1/1 dataset results" in caplog.text
    assert "Approve replaces 1" in caplog.text
    assert "submitted-digest" not in caplog.text
    assert "test_f1" not in caplog.text
    reviewer.decide.assert_not_called()


def test_verbose_cli_retains_audit_evidence(
    report: ReviewReport,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The public CLI exposes digests, checks and provenance on request."""
    reviewer = Mock(spec=VolunteerReviewer)
    reviewer.group_submissions_by_model.side_effect = lambda submission_ids: [
        submission_ids
    ]
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
