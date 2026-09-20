"""Tests for the report-only contamination-canary scoring script."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from canary import score_contamination_canaries as script

from euroeval.canary_evidence import CANARY_PRIVATE_DIR_ENV, CANARY_RESULT_DATASET
from leaderboards.contamination_canary import CANARY_KEY_ENV, CANARY_REPORT_PATH_ENV


def test_default_paths_configure_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the documented file and private-scoring defaults."""
    monkeypatch.chdir(tmp_path)
    Path("euroeval_benchmark_results.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        script,
        "score_canary_records",
        lambda *, records: {"status": "missing", "models": []},
    )

    assert script.main([]) == 0

    assert os.environ[CANARY_KEY_ENV] == str(script.DEFAULT_KEY_PATH.expanduser())
    assert os.environ[CANARY_PRIVATE_DIR_ENV] == str(
        script.DEFAULT_PRIVATE_DIR.expanduser()
    )
    assert os.environ[CANARY_REPORT_PATH_ENV] == str(
        script.DEFAULT_REPORT_PATH.expanduser()
    )


def test_custom_arguments_configure_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replace every environment-backed scorer path with a CLI argument."""
    results_path = tmp_path / "results.jsonl"
    results_path.write_text("", encoding="utf-8")
    key_path = tmp_path / "audit.key"
    private_dir = tmp_path / "private"
    report_path = tmp_path / "reports" / "audit.json"
    monkeypatch.setattr(
        script,
        "score_canary_records",
        lambda *, records: {"status": "missing", "models": []},
    )

    assert (
        script.main(
            [
                str(results_path),
                "--key",
                str(key_path),
                "--private-dir",
                str(private_dir),
                "--report-path",
                str(report_path),
            ]
        )
        == 0
    )

    assert os.environ[CANARY_KEY_ENV] == str(key_path)
    assert os.environ[CANARY_PRIVATE_DIR_ENV] == str(private_dir)
    assert os.environ[CANARY_REPORT_PATH_ENV] == str(report_path)


def test_all_and_only_canary_records_are_scored_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass every canary variant, and no ordinary result, in one scorer call."""
    ordinary = _record(dataset="sentiment")
    canary = _record(dataset=CANARY_RESULT_DATASET)
    canary_variant = _record(dataset=f"{CANARY_RESULT_DATASET}-worker")
    results_path = _write_jsonl(
        tmp_path=tmp_path, records=[ordinary, canary, canary_variant]
    )
    scored_batches: list[list[dict[str, object]]] = []

    def fake_score(*, records: list[dict[str, object]]) -> dict[str, object]:
        scored_batches.append(records)
        return {"status": "scored", "models": []}

    monkeypatch.setattr(script, "score_canary_records", fake_score)

    assert script.main([str(results_path)]) == 0

    assert scored_batches == [[canary, canary_variant]]


def test_complete_report_is_stable_json_on_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Write only sorted, indented report JSON to normal standard output."""
    results_path = _write_jsonl(
        tmp_path=tmp_path, records=[_record(dataset=CANARY_RESULT_DATASET)]
    )
    report: dict[str, object] = {
        "status": "unavailable",
        "models": [],
        "reason": "FileNotFoundError",
    }
    monkeypatch.setattr(script, "score_canary_records", lambda *, records: report)

    assert script.main([str(results_path)]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == report
    assert captured.out == json.dumps(report, indent=2, sort_keys=True) + "\n"
    assert captured.err == ""


def test_no_canary_input_preserves_missing_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Let the scorer report its meaningful missing status for an empty selection."""
    results_path = _write_jsonl(
        tmp_path=tmp_path, records=[_record(dataset="sentiment")]
    )
    observed: list[list[dict[str, object]]] = []

    def fake_score(*, records: list[dict[str, object]]) -> dict[str, object]:
        observed.append(records)
        return {"schema_version": "test/v1", "status": "missing", "models": []}

    monkeypatch.setattr(script, "score_canary_records", fake_score)

    assert script.main([str(results_path)]) == 0

    assert observed == [[]]
    assert json.loads(capsys.readouterr().out)["status"] == "missing"


@pytest.mark.parametrize(
    ("contents", "expected_error"),
    [(None, "results file does not exist"), ("not-json\n", "valid JSONL")],
)
def test_missing_or_invalid_input_is_a_clear_cli_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    contents: str | None,
    expected_error: str,
) -> None:
    """Reject unreadable benchmark input with a non-zero stderr diagnostic."""
    results_path = tmp_path / "results.jsonl"
    if contents is not None:
        results_path.write_text(contents, encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        script.main([str(results_path)])

    captured = capsys.readouterr()
    assert exc_info.value.code != 0
    assert captured.out == ""
    assert expected_error in captured.err
    assert str(results_path) in captured.err


def test_malformed_input_does_not_leak_completion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep malformed completion data out of the CLI diagnostics."""
    secret_completion = "UNIQUE_SECRET_COMPLETION_MARKER"
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        f'{{"normalised_completion": "{secret_completion}"\n', encoding="utf-8"
    )

    with pytest.raises(SystemExit) as exc_info:
        script.main([str(results_path)])

    captured = capsys.readouterr()
    assert exc_info.value.code != 0
    assert secret_completion not in captured.out
    assert secret_completion not in captured.err


def _record(*, dataset: str) -> dict[str, object]:
    """Create the minimum record shape used by canary selection.

    Returns:
        A benchmark result with the requested dataset name.
    """
    return {
        "eval_library": {
            "additional_details": {
                "dataset": dataset,
                "contamination_canary_evidence": {
                    "observations": [{"normalised_completion": "private words"}]
                },
            }
        }
    }


def _write_jsonl(*, tmp_path: Path, records: list[dict[str, object]]) -> Path:
    """Write benchmark records to a temporary JSONL file.

    Returns:
        The temporary JSONL path.
    """
    path = tmp_path / "results.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path
