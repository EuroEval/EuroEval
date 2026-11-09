"""Benchmark-result JSONL consumers preserve quoted content without remote effects."""

import json
from pathlib import Path

import pytest

from leaderboards.queue_parsing import read_jsonl_lines
from src.scripts import collect_evaluation_results as collect
from src.scripts import swap_leaderboard_dataset as swap


@pytest.fixture(params=["\n", "\r\n"])
def newline(request: pytest.FixtureRequest) -> str:
    """Return physical LF or CRLF input framing.

    Args:
        request:
            The selected physical newline.

    Returns:
        A record delimiter to write as exact bytes.
    """
    return request.param


@pytest.fixture(params=["left}{right", "x\u0085y", "x\u2028y", "x\u2029y"])
def record(request: pytest.FixtureRequest) -> dict[str, object]:
    """Return a synthetic identity-bearing result with quoted separator content.

    Args:
        request:
            The selected string content.

    Returns:
        A local-only benchmark-result record.
    """
    return {
        "model_info": {"id": "synthetic/model", "note": request.param},
        "eval_library": {
            "additional_details": {
                "dataset": "synthetic",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
    }


def test_collector_upload_preserves_valid_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: dict[str, object],
    newline: str,
) -> None:
    """Parse a real local input and preserve it in the result tree.

    Args:
        tmp_path:
            Temporary local directory.
        monkeypatch:
            Replace only the output directory and remote upload.
        record:
            Synthetic benchmark record.
        newline:
            Physical LF or CRLF input delimiter.
    """
    path = tmp_path / "input.jsonl"
    path.write_bytes((json.dumps(record, ensure_ascii=False) + newline).encode("utf-8"))
    results_dir = tmp_path / "results"
    monkeypatch.setattr(collect, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(collect, "_upload_to_bucket", lambda paths: True)

    assert collect.upload_results_to_hf(new_results_path=path) is True
    written = list(results_dir.rglob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text(encoding="utf-8")) == record


def test_manual_collection_does_not_rewrite_quoted_separators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: dict[str, object],
    newline: str,
) -> None:
    """Exercise the public command's manual-input path and stop before deployment.

    Args:
        tmp_path:
            Temporary local directory.
        monkeypatch:
            Disable remote work and stop at regeneration, before deployment.
        record:
            Synthetic benchmark record.
        newline:
            Physical LF or CRLF input delimiter.
    """
    path = tmp_path / "new_results.jsonl"
    path.write_bytes((json.dumps(record, ensure_ascii=False) + newline).encode("utf-8"))
    monkeypatch.setattr(collect, "NEW_RESULTS_PATH", path)
    monkeypatch.setattr(collect, "check_required_env_vars", lambda: None)
    monkeypatch.setattr(collect, "preflight_volunteer_review", lambda: True)
    monkeypatch.setattr(collect, "_fetch_issues", lambda: [])
    monkeypatch.setattr(collect, "_harvest_results", lambda issues: [])
    monkeypatch.setattr(collect, "_upload_results", lambda: None)
    monkeypatch.setattr(collect, "regenerate_leaderboards", lambda **kwargs: False)
    callback = collect.main.callback
    assert callback is not None

    with pytest.raises(SystemExit) as caught:
        callback(force=False)

    assert caught.value.code == 1
    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_queue_read_preserves_physical_jsonl_records(
    tmp_path: Path, record: dict[str, object], newline: str
) -> None:
    """Keep quoted Unicode separators within their original record.

    Args:
        tmp_path:
            Temporary local directory.
        record:
            Synthetic benchmark record.
        newline:
            Physical LF or CRLF input delimiter.
    """
    path = tmp_path / "results.jsonl"
    path.write_bytes((json.dumps(record, ensure_ascii=False) + newline).encode("utf-8"))

    lines = read_jsonl_lines(path=path)

    assert len(lines) == 1
    assert "\r" not in lines[0]
    assert json.loads(lines[0]) == record


def test_sync_accepts_valid_jsonl_and_retains_exact_line_deduplication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: dict[str, object],
    newline: str,
) -> None:
    """Test consolidation with a valid UTF-8 JSONL merger output, not a bucket call.

    Args:
        tmp_path:
            Temporary local directory.
        monkeypatch:
            Replace the bucket reader and merger output with synthetic JSONL.
        record:
            Synthetic benchmark record.
        newline:
            Physical LF or CRLF input delimiter.
    """
    line = json.dumps(record, ensure_ascii=False)
    new_record = {**record, "new_record": True}
    new_line = json.dumps(new_record, ensure_ascii=False)
    path = tmp_path / "new_results.jsonl"
    path.write_bytes((line + newline).encode("utf-8"))
    monkeypatch.setattr(swap, "NEW_RESULTS_PATH", path)
    monkeypatch.setattr(swap, "result_sync_dataset_ids", lambda **kwargs: {"synthetic"})
    monkeypatch.setattr(swap, "download_bucket_files_for_datasets", lambda **kwargs: 0)

    def merge(results_file: Path) -> int:
        """Supply one valid existing record without any bucket access.

        Args:
            results_file:
                Temporary merger output path.

        Returns:
            Number of synthetic records.
        """
        merged_newline = "\n" if newline == "\r\n" else "\r\n"
        results_file.write_bytes(
            (line + merged_newline + new_line + merged_newline).encode("utf-8")
        )
        return 2

    monkeypatch.setattr(swap, "merge_results", merge)
    swap.sync_results_from_bucket(
        old_dataset=None,
        new_datasets=("synthetic",),
        swapped_task="knowledge",
        target_codes={"da"},
    )

    assert path.read_text(encoding="utf-8") == line + "\n" + new_line + "\n"
