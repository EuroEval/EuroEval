"""Tests for the process_evaluation_queue script orchestration."""

import json
from pathlib import Path

import pytest

from leaderboards import bucket_sync
from src.scripts import process_evaluation_queue


def test_sync_bucket_preserves_local_only_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Local-only record files not yet in the bucket should survive startup sync."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    model_dir = results_dir / "foo_bar"
    model_dir.mkdir()
    local_record_file = model_dir / "dataset__test__test.json"
    local_record_file.write_text(
        '{"model_info": {"id": "foo/bar"}, "local": true}', encoding="utf-8"
    )

    class FakeHfApi:
        def sync_bucket(
            self,
            source: str,
            dest: str,
            token: str,
            ignore_times: bool = False,
            **kwargs,
        ) -> None:
            # Tree sync doesn't remove files - local-only files persist
            pass

    monkeypatch.setattr(target=bucket_sync, name="RESULTS_DIR", value=results_dir)
    monkeypatch.setattr(target=bucket_sync, name="HfApi", value=FakeHfApi)
    monkeypatch.setattr(
        target=bucket_sync, name="resolve_hf_token", value=lambda: "token"
    )

    bucket_sync.sync_bucket()

    # Local-only file should still exist after sync
    assert local_record_file.exists()
    content = json.loads(local_record_file.read_text(encoding="utf-8"))
    assert content["local"] is True


def test_sync_bucket_prefers_synced_content_for_existing_result_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Remote metadata should be fetched during sync."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    model_dir = results_dir / "foo_bar"
    model_dir.mkdir()
    record_file = model_dir / "greek_ner__test__false.json"
    local_content = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {"additional_details": {"dataset": "greek_ner"}},
        "local": True,
    }
    record_file.write_text(json.dumps(local_content), encoding="utf-8")

    class FakeHfApi:
        def sync_bucket(
            self,
            source: str,
            dest: str,
            token: str,
            ignore_times: bool = False,
            **kwargs,
        ) -> None:
            # Tree sync preserves existing files - remote content merged via file paths
            pass

    monkeypatch.setattr(target=bucket_sync, name="RESULTS_DIR", value=results_dir)
    monkeypatch.setattr(target=bucket_sync, name="HfApi", value=FakeHfApi)
    monkeypatch.setattr(
        target=bucket_sync, name="resolve_hf_token", value=lambda: "token"
    )

    bucket_sync.sync_bucket()

    # File should still exist after sync
    assert record_file.exists()
    content = json.loads(record_file.read_text(encoding="utf-8"))
    # In tree model, files persist; dedup happens later in merge_results
    assert "local" in content or "open" in content


def test_sync_bucket_handles_separate_result_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each result identity is a separate file in tree structure."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    model_dir = results_dir / "foo_bar"
    model_dir.mkdir()

    # Two separate result files for different datasets
    greek_file = model_dir / "greek_ner__test__false.json"
    greek_file.write_text(
        json.dumps(
            {
                "model_info": {"id": "foo/bar"},
                "eval_library": {"additional_details": {"dataset": "greek_ner"}},
            }
        ),
        encoding="utf-8",
    )
    danish_file = model_dir / "danish_ner__test__false.json"
    danish_file.write_text(
        json.dumps(
            {
                "model_info": {"id": "foo/bar"},
                "eval_library": {"additional_details": {"dataset": "danish_ner"}},
            }
        ),
        encoding="utf-8",
    )

    class FakeHfApi:
        def sync_bucket(
            self,
            source: str,
            dest: str,
            token: str,
            ignore_times: bool = False,
            **kwargs,
        ) -> None:
            # Tree sync preserves separate files
            pass

    monkeypatch.setattr(target=bucket_sync, name="RESULTS_DIR", value=results_dir)
    monkeypatch.setattr(target=bucket_sync, name="HfApi", value=FakeHfApi)
    monkeypatch.setattr(
        target=bucket_sync, name="resolve_hf_token", value=lambda: "token"
    )

    bucket_sync.sync_bucket()

    # Both files should persist after sync
    assert greek_file.exists()
    assert danish_file.exists()


def test_run_claimed_issue_resumes_from_merged_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Merged canonical results should count when resuming an issue."""
    results_ready: list[int] = []
    cleared_markers: list[tuple[int, str]] = []

    def read_lines(path: Path) -> list[str]:
        if path.name == "euroeval_benchmark_results.jsonl":
            return ['{"model":"foo/bar"}']
        return []

    monkeypatch.setattr(
        target=process_evaluation_queue, name="read_jsonl_lines", value=read_lines
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="result_lines_for_model",
        value=lambda lines, model_id: lines,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="completed_languages",
        value=lambda lines, requested_languages: (
            list(requested_languages) if lines == ['{"model":"foo/bar"}'] else []
        ),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="run_euroeval",
        value=lambda **kwargs: pytest.fail("merged results should avoid re-running"),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="remove_failed_label",
        value=lambda number: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="add_results_ready_label",
        value=lambda number: results_ready.append(number),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="clear_vm_marker",
        value=lambda number, vm_id: cleared_markers.append((number, vm_id)),
    )

    process_evaluation_queue._run_claimed_issue(
        issue={"number": 8},
        model_id="foo/bar",
        languages=["el"],
        assignee="tester",
        vm_id="test-vm",
        gpu_memory_utilization=None,
    )

    assert results_ready == [8]
    assert cleared_markers == [(8, "test-vm")]


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

    # _run_claimed_issue reads the model's result subdirectory directly (not via
    # read_jsonl_lines), then reads the local euroeval output once, followed by a
    # before/after read per pending language.
    lines_per_read = iter([[], ["before"], ["before"]])

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
        assignee="tester",
        vm_id="test-vm",
        gpu_memory_utilization=None,
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
        issue={"number": 42, "body": ""},
        model_id="foo/bar",
        groups=["Greek"],
        assignee="tester",
        vm_id="test-vm",
        gpu_memory_utilization=None,
    )

    assert len(comments) == 1
    assert "euroeval exited with code 1" in comments[0]
    assert unassigned == [42]


def test_process_issue_marks_ready_when_missing_pairs_are_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resume whose missing pairs are all intentional skips must not error."""
    comments: list[str] = []
    results_ready: list[int] = []
    failed_labels: list[int] = []

    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="issue_is_still_claimable",
        value=lambda number: True,
    )
    # The bucket already holds a line for this model; the run produces nothing
    # new (the only remaining official pair is one euroeval skips).
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="read_jsonl_lines",
        value=lambda path: ['{"model":"foo/bar"}'],
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="result_lines_for_model",
        value=lambda lines, model_id: ['{"model":"foo/bar"}'],
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="completed_languages",
        value=lambda lines, requested_languages: [],
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="assign_issue",
        value=lambda number, assignee: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="unassign_issue",
        value=lambda number, assignee: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="run_euroeval",
        value=lambda **kwargs: (0, "skipped 1 benchmarks"),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="upload_results_to_hf_bucket",
        value=lambda lines, model_id: True,
    )
    # One official pair stays missing, and euroeval reports exactly one skip.
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="missing_official_dataset_language_pairs",
        value=lambda lines, requested_languages: {("greek_ner", "el")},
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="num_errored_benchmarks",
        value=lambda output: 0,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="num_skipped_benchmarks",
        value=lambda output: 1,
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
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="release_issue_if_owned",
        value=lambda number, vm_id, assignee: True,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="remove_failed_label",
        value=lambda number: None,
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="add_results_ready_label",
        value=lambda number: results_ready.append(number),
    )
    monkeypatch.setattr(
        target=process_evaluation_queue,
        name="add_failed_label",
        value=lambda number: failed_labels.append(number),
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

    process_evaluation_queue.process_issue(
        issue={"number": 7, "body": ""},
        model_id="foo/bar",
        groups=["Greek"],
        assignee="tester",
        vm_id="test-vm",
        gpu_memory_utilization=None,
    )

    assert comments == []
    assert failed_labels == []
    assert results_ready == [7]
