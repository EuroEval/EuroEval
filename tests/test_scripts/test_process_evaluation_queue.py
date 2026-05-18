"""Tests for process_evaluation_queue utilities."""

from types import SimpleNamespace

import pytest

from src.scripts import process_evaluation_queue


def test_process_issue_fails_when_official_results_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The issue should be marked errored when official results are incomplete."""
    comments: list[str] = []
    marker_versions: list[str] = []
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
        value=lambda number: assigned.append(number),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="unassign_issue",
        value=lambda number: unassigned.append(number),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="read_jsonl_lines",
        value=lambda path: next(lines_per_read, ["before"]),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="run_euroeval",
        value=lambda model_id, languages: (0, "all good"),
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
        target=process_evaluation_queue, name="euroeval_version", value=lambda: "99.0.0"
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="comment_on_issue",
        value=lambda number, comment: comments.append(comment),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="set_errored_marker",
        value=lambda number, body, version: marker_versions.append(version),
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
    assert marker_versions == ["99.0.0"]
    assert len(comments) == 1
    assert "missing results for 1 official dataset-language pair(s)" in comments[0]
    assert "danish_ner/da" in comments[0]


def test_process_issue_does_not_special_case_oom_anymore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OOM output should still be treated as a normal failure and commented."""
    comments: list[str] = []
    marker_versions: list[str] = []
    unassigned: list[int] = []

    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="issue_is_still_claimable",
        value=lambda number: True,
    )

    lines_per_read = iter([["before"], ["before"]])

    monkeypatch.setattr(
        target=process_evaluation_queue, name="assign_issue", value=lambda number: None
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="unassign_issue",
        value=lambda number: unassigned.append(number),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="read_jsonl_lines",
        value=lambda path: next(lines_per_read, ["before"]),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="run_euroeval",
        value=lambda model_id, languages: (1, "RuntimeError: CUDA out of memory"),
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
        target=process_evaluation_queue, name="euroeval_version", value=lambda: "99.0.0"
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="comment_on_issue",
        value=lambda number, comment: comments.append(comment),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="set_errored_marker",
        value=lambda number, body, version: marker_versions.append(version),
    )

    process_evaluation_queue.process_issue(
        issue={"number": 42, "body": ""}, model_id="foo/bar", groups=["Greek"]
    )

    assert len(comments) == 1
    assert "euroeval exited with code 1" in comments[0]
    assert marker_versions == ["99.0.0"]
    assert unassigned == [42]


@pytest.mark.parametrize(
    argnames=["dtype", "count", "expected_bytes"],
    argvalues=[
        ("INT4", 3, 2),
        ("custom_8", 9, 9),
        ("something16_else", 1, 2),
        ("unknown_dtype", 5, 10),
    ],
)
def test_estimated_model_bytes_dtype_fallback(
    monkeypatch: pytest.MonkeyPatch, dtype: str, count: int, expected_bytes: int
) -> None:
    """Unknown dtypes should use numeric fallback, then BF16 fallback."""

    def fake_get_safetensors_metadata(
        *, repo_id: str, revision: str | None = None, token: str | None = None
    ) -> SimpleNamespace:
        _ = repo_id, revision, token
        return SimpleNamespace(parameter_count={dtype: count})

    monkeypatch.setattr(
        process_evaluation_queue,
        "get_safetensors_metadata",
        fake_get_safetensors_metadata,
    )
    bytes_needed = process_evaluation_queue.estimated_model_bytes(
        model_id="org/model-name"
    )
    assert bytes_needed == expected_bytes


@pytest.mark.parametrize(
    argnames=["model_id"], argvalues=[("org/model#low@rev",), ("org/model@rev#low",)]
)
def test_estimated_model_bytes_handles_model_id_extras(
    monkeypatch: pytest.MonkeyPatch, model_id: str
) -> None:
    """Model id parsing should handle # and @ in either order."""
    calls: list[dict[str, str | None]] = []

    def fake_get_safetensors_metadata(
        *, repo_id: str, revision: str | None = None, token: str | None = None
    ) -> SimpleNamespace:
        calls.append({"repo_id": repo_id, "revision": revision, "token": token})
        return SimpleNamespace(parameter_count={"BF16": 1})

    monkeypatch.setattr(
        process_evaluation_queue,
        "get_safetensors_metadata",
        fake_get_safetensors_metadata,
    )
    assert process_evaluation_queue.estimated_model_bytes(model_id=model_id) == 2
    assert calls[0]["repo_id"] == "org/model"
    assert calls[0]["revision"] == "rev"
