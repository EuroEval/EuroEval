"""Tests for the process_evaluation_queue script orchestration."""

import pytest

from src.scripts import process_evaluation_queue


def test_process_issue_fails_when_official_results_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The issue should be marked errored when official results are incomplete."""
    comments: list[str] = []
    unassigned: list[int] = []
    assigned: list[int] = []

    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="issue_is_still_claimable",
        value=lambda number: True,
    )

    lines_per_read = iter([["before"], ["before", '{"foo":"bar"}', '{"baz":"qux"}']])

    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="assign_issue",
        value=lambda number, assignee: assigned.append(number),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="unassign_issue",
        value=lambda number, assignee: unassigned.append(number),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="read_jsonl_lines",
        value=lambda path: next(lines_per_read, ["before"]),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="run_euroeval",
        value=lambda **kwargs: (0, "all good"),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="missing_official_dataset_language_pairs",
        value=lambda lines, requested_languages: {("danish_ner", "da")},
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="num_errored_benchmarks",
        value=lambda output: 0,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="num_skipped_benchmarks",
        value=lambda output: 0,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue, name="__version__", value="99.0.0"
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="comment_on_issue",
        value=lambda number, body: comments.append(body),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="cached_model_summary",
        value=lambda model_id: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="set_vm_marker",
        value=lambda number, vm_id: True,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="clear_vm_marker",
        value=lambda number, vm_id: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="vm_marker_matches",
        value=lambda number, vm_id: True,
    )

    def fake_release(number: int, vm_id: str, assignee: str) -> bool:
        unassigned.append(number)
        return True

    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="release_issue_if_owned",
        value=fake_release,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="add_failed_label",
        value=lambda number: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="add_gated_label",
        value=lambda number: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="remove_gated_label",
        value=lambda number: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="issue_has_matching_error_comment",
        value=lambda number, reason: False,
    )

    process_evaluation_queue.process_issue(
        issue={"number": 17, "body": "body"},
        model_id="foo/bar",
        groups=[
            "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)"
        ],
    )

    assert assigned == [17]
    assert unassigned == [17]
    assert len(comments) == 1
    assert "missing official dataset-language pair(s)" in comments[0]
    assert "danish_ner/da" in comments[0]


def test_process_issue_does_not_special_case_oom_anymore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OOM output should still be treated as a normal failure and commented."""
    comments: list[str] = []
    unassigned: list[int] = []

    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="issue_is_still_claimable",
        value=lambda number: True,
    )

    lines_per_read = iter([["before"], ["before"]])

    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="assign_issue",
        value=lambda number, assignee: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="unassign_issue",
        value=lambda number, assignee: unassigned.append(number),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="read_jsonl_lines",
        value=lambda path: next(lines_per_read, ["before"]),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="run_euroeval",
        value=lambda **kwargs: (1, "RuntimeError: CUDA out of memory"),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="missing_official_dataset_language_pairs",
        value=lambda lines, requested_languages: set(),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="num_errored_benchmarks",
        value=lambda output: 0,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="num_skipped_benchmarks",
        value=lambda output: 0,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue, name="__version__", value="99.0.0"
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="comment_on_issue",
        value=lambda number, body: comments.append(body),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="cached_model_summary",
        value=lambda model_id: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="set_vm_marker",
        value=lambda number, vm_id: True,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="clear_vm_marker",
        value=lambda number, vm_id: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="vm_marker_matches",
        value=lambda number, vm_id: True,
    )

    def fake_release(number: int, vm_id: str, assignee: str) -> bool:
        unassigned.append(number)
        return True

    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="release_issue_if_owned",
        value=fake_release,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="add_failed_label",
        value=lambda number: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="add_gated_label",
        value=lambda number: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="remove_gated_label",
        value=lambda number: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="issue_has_matching_error_comment",
        value=lambda number, reason: False,
    )

    process_evaluation_queue.process_issue(
        issue={"number": 42, "body": ""}, model_id="foo/bar", groups=["Greek"]
    )

    assert len(comments) == 1
    assert "euroeval exited with code 1" in comments[0]
    assert unassigned == [42]
