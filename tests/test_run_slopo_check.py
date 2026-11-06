"""Tests for the isolated Slopo check runner."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.run_slopo_check import run_slopo_check


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (
            "Similarity ratio (including exact copies): 0.00% "
            "(0/10 units flagged as similar)",
            0,
        ),
        (
            "Similarity ratio (including exact copies): 20.00% "
            "(2/10 units flagged as similar)",
            1,
        ),
    ],
)
def test_duplicate_counts_control_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, summary: str, expected: int
) -> None:
    """Valid duplicate counts pass only when the count is zero."""
    _write_configuration(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=summary if args[0][-1] == "analyze" else "", stderr=""
        ),
    )

    assert run_slopo_check(root=tmp_path) == expected


def _write_configuration(root: Path) -> None:
    """Create the minimal Slopo configuration used by a runner test."""
    (root / ".slopo").mkdir()
    (root / ".slopo" / "slopo.ignore.txt").write_text("# ignored clusters\n")
    (root / "src").mkdir()
    (root / "slopo.conf.yaml").write_text(
        yaml.safe_dump(
            {
                "source_dir": "src",
                "db_file": ".slopo/slopo.db",
                "report_dir": ".slopo/report",
                "ignore_file": ".slopo/slopo.ignore.txt",
            }
        )
    )


@pytest.mark.parametrize("summary", ["not an analysis", "Similarity ratio: 0%"])
def test_malformed_analysis_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, summary: str
) -> None:
    """An analysis without one valid summary is not treated as success."""
    _write_configuration(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=summary if args[0][-1] == "analyze" else "", stderr=""
        ),
    )

    assert run_slopo_check(root=tmp_path) == 1


@pytest.mark.parametrize("failed_stage", ["index", "embed", "analyze"])
def test_stage_failures_stop_the_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failed_stage: str,
) -> None:
    """A failed Slopo stage fails the check and prevents later stages."""
    _write_configuration(tmp_path)
    called: list[str] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        stage = command[-1]
        called.append(stage)
        return subprocess.CompletedProcess(
            args=command,
            returncode=2 if stage == failed_stage else 0,
            stdout="stage output\n",
            stderr="failure details\n" if stage == failed_stage else "",
        )

    monkeypatch.setattr(subprocess, "run", run)

    assert run_slopo_check(root=tmp_path) == 1
    assert called == ["index", "embed", "analyze"][: called.index(failed_stage) + 1]
    assert "failure details" in capsys.readouterr().out


def test_uses_isolated_database_and_report_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured shared database and report are never passed to Slopo."""
    _write_configuration(tmp_path)
    shared_database = tmp_path / ".slopo" / "slopo.db"
    shared_database.symlink_to(tmp_path / "primary-cache.db")
    configurations: list[dict[str, object]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        config_path = Path(command[4])
        config = yaml.safe_load(config_path.read_text())
        configurations.append(config)
        output = (
            "Similarity ratio (including exact copies): 0.00% "
            "(0/10 units flagged as similar)"
            if command[-1] == "analyze"
            else ""
        )
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout=output, stderr=""
        )

    monkeypatch.setattr(subprocess, "run", run)

    assert run_slopo_check(root=tmp_path) == 0
    assert len(configurations) == 3
    assert all(config["db_file"] != str(shared_database) for config in configurations)
    assert all(
        config["report_dir"] != str(tmp_path / ".slopo" / "report")
        for config in configurations
    )
    assert shared_database.is_symlink()
