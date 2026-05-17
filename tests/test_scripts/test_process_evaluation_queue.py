"""Tests for scripts/process_evaluation_queue.py."""

from pytest import MonkeyPatch

from scripts import process_evaluation_queue as queue


def test_process_issue_fails_when_official_results_are_missing(
    monkeypatch: MonkeyPatch,
) -> None:
    """The issue should be marked errored when official results are incomplete."""
    comments: list[str] = []
    marker_versions: list[str] = []
    unassigned: list[int] = []
    assigned: list[int] = []

    lines_per_read = iter([["before"], ["before", '{"foo":"bar"}']])

    monkeypatch.setattr(
        target=queue, name="assign_issue", value=lambda number: assigned.append(number)
    )
    monkeypatch.setattr(
        target=queue,
        name="unassign_issue",
        value=lambda number: unassigned.append(number),
    )
    monkeypatch.setattr(
        target=queue,
        name="read_jsonl_lines",
        value=lambda path: next(lines_per_read, ["before"]),
    )
    monkeypatch.setattr(
        target=queue,
        name="run_euroeval",
        value=lambda model_id, languages: (0, "all good"),
    )
    monkeypatch.setattr(
        target=queue,
        name="missing_official_dataset_language_pairs",
        value=lambda lines, requested_languages: {("danish_ner", "da")},
    )
    monkeypatch.setattr(
        target=queue, name="num_errored_benchmarks", value=lambda output: 0
    )
    monkeypatch.setattr(target=queue, name="euroeval_version", value=lambda: "99.0.0")
    monkeypatch.setattr(
        target=queue,
        name="comment_on_issue",
        value=lambda number, comment: comments.append(comment),
    )
    monkeypatch.setattr(
        target=queue,
        name="set_errored_marker",
        value=lambda number, body, version: marker_versions.append(version),
    )

    queue.process_issue(
        issue={"number": 17, "body": "body"},
        model_id="foo/bar",
        groups=[
            "Scandinavian languages (Danish, Faroese, Icelandic, Norwegian, Swedish)"
        ],
    )

    assert assigned == [17]
    assert unassigned == [17]
    assert marker_versions == ["99.0.0"]
    assert len(comments) == 1
    assert "missing results for 1 official dataset-language pair(s)" in comments[0]
    assert "danish_ner/da" in comments[0]


def test_process_issue_does_not_special_case_oom_anymore(
    monkeypatch: MonkeyPatch,
) -> None:
    """OOM output should still be treated as a normal failure and commented."""
    comments: list[str] = []
    marker_versions: list[str] = []
    unassigned: list[int] = []

    lines_per_read = iter([["before"], ["before"]])

    monkeypatch.setattr(target=queue, name="assign_issue", value=lambda number: None)
    monkeypatch.setattr(
        target=queue,
        name="unassign_issue",
        value=lambda number: unassigned.append(number),
    )
    monkeypatch.setattr(
        target=queue,
        name="read_jsonl_lines",
        value=lambda path: next(lines_per_read, ["before"]),
    )
    monkeypatch.setattr(
        target=queue,
        name="run_euroeval",
        value=lambda model_id, languages: (1, "RuntimeError: CUDA out of memory"),
    )
    monkeypatch.setattr(
        target=queue,
        name="missing_official_dataset_language_pairs",
        value=lambda lines, requested_languages: set(),
    )
    monkeypatch.setattr(
        target=queue, name="num_errored_benchmarks", value=lambda output: 0
    )
    monkeypatch.setattr(target=queue, name="euroeval_version", value=lambda: "99.0.0")
    monkeypatch.setattr(
        target=queue,
        name="comment_on_issue",
        value=lambda number, comment: comments.append(comment),
    )
    monkeypatch.setattr(
        target=queue,
        name="set_errored_marker",
        value=lambda number, body, version: marker_versions.append(version),
    )

    queue.process_issue(
        issue={"number": 42, "body": ""}, model_id="foo/bar", groups=["Greek"]
    )

    assert len(comments) == 1
    assert "euroeval exited with code 1" in comments[0]
    assert marker_versions == ["99.0.0"]
    assert unassigned == [42]
