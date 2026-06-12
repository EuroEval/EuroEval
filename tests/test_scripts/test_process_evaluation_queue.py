"""Tests for the process_evaluation_queue script orchestration."""

from types import SimpleNamespace

import pytest
from huggingface_hub import ModelInfo

import leaderboards.queue_progress as queue_progress
from leaderboards import evaluation_common
from leaderboards.queue_hf_cache import is_gguf_model
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
        target=process_evaluation_queue, name="euroeval_version", value=lambda: "99.0.0"
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
        value=lambda number, vm_id: None,
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
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="find_progress_comment",
        value=lambda number: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="post_or_update_progress_comment",
        value=lambda **kwargs: None,
    )
    monkeypatch.setattr(
        target=queue_progress, name="upload_results_gist", value=lambda **kwargs: None
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
        target=process_evaluation_queue, name="euroeval_version", value=lambda: "99.0.0"
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
        value=lambda number, vm_id: None,
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
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="find_progress_comment",
        value=lambda number: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="post_or_update_progress_comment",
        value=lambda **kwargs: None,
    )
    monkeypatch.setattr(
        target=queue_progress, name="upload_results_gist", value=lambda **kwargs: None
    )

    process_evaluation_queue.process_issue(
        issue={"number": 42, "body": ""}, model_id="foo/bar", groups=["Greek"]
    )

    assert len(comments) == 1
    assert "euroeval exited with code 1" in comments[0]
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
        evaluation_common, "get_safetensors_metadata", fake_get_safetensors_metadata
    )
    bytes_needed = evaluation_common.estimated_model_bytes(model_id="org/model-name")
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
        evaluation_common, "get_safetensors_metadata", fake_get_safetensors_metadata
    )
    assert evaluation_common.estimated_model_bytes(model_id=model_id) == 2
    assert calls[0]["repo_id"] == "org/model"
    assert calls[0]["revision"] == "rev"


@pytest.mark.parametrize(
    argnames=["info", "expected"],
    argvalues=[
        # The "gguf" tag alone is enough, regardless of file layout.
        (ModelInfo(id="org/m", tags=["gguf", "qwen"]), True),
        # Tag matching is case-insensitive.
        (ModelInfo(id="org/m", tags=["GGUF"]), True),
        # library_name set to gguf.
        (ModelInfo(id="org/m", library_name="gguf"), True),
        # A .gguf file nested in a per-quant subfolder, no tag.
        (
            ModelInfo(
                id="org/m", siblings=[{"rfilename": "Q4_K_M/model-00001-of-2.gguf"}]
            ),
            True,
        ),
        # A plain safetensors model is not GGUF.
        (
            ModelInfo(
                id="org/m",
                tags=["text-generation"],
                library_name="transformers",
                siblings=[{"rfilename": "model.safetensors"}],
            ),
            False,
        ),
        # Missing/None attributes must not raise.
        (ModelInfo(id="org/m"), False),
    ],
)
def test_is_gguf_model(info: ModelInfo, expected: bool) -> None:
    """GGUF repos are detected via tag, library_name, or any .gguf sibling."""
    assert is_gguf_model(info=info) is expected
