"""Tests for resilient prebuilt Vercel deployment."""

import json
import os
import stat
from pathlib import Path

import pytest
from deploy_vercel_frontend import main


def test_known_upload_error_recovers_with_exact_deploy_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry the known failure and preserve the production deploy arguments."""
    calls_path = _fake_vercel(tmp_path=tmp_path, monkeypatch=monkeypatch)
    monkeypatch.setenv("STATUSES", "1,0")
    monkeypatch.setenv(
        "OUTPUT", "Unexpected token 'I', \"Internal S\"... is not valid JSON"
    )
    delays = []
    monkeypatch.setattr("deploy_vercel_frontend.time.sleep", delays.append)

    assert main([]) == 0
    assert json.loads(calls_path.read_text()) == [
        ["deploy", "--prebuilt", "--prod", "--yes", "--non-interactive"],
        ["deploy", "--prebuilt", "--prod", "--yes", "--non-interactive"],
    ]
    assert delays == [2]


def _fake_vercel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install a controllable fake Vercel executable for an interface test.

    Args:
        tmp_path:
            Temporary directory for the executable and call record.
        monkeypatch:
            Pytest environment fixture.

    Returns:
        Path to the fake CLI's recorded calls.
    """
    executable = tmp_path / "vercel"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "path = os.environ['CALLS']\n"
        "try: calls = json.load(open(path))\n"
        "except FileNotFoundError: calls = []\n"
        "calls.append(sys.argv[1:])\n"
        "json.dump(calls, open(path, 'w'))\n"
        "statuses = os.environ.get('STATUSES', '0').split(',')\n"
        "status = int(statuses[min(len(calls) - 1, len(statuses) - 1)])\n"
        "if status:\n"
        " print(os.environ.get('OUTPUT', 'deployment failed'), flush=True)\n"
        "sys.exit(status)\n"
        "print('deployment complete', flush=True)\n"
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("CALLS", str(tmp_path / "calls.json"))
    return tmp_path / "calls.json"


def test_persistent_upload_failure_stops_after_three_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bound repeated upload failures and return the CLI failure status."""
    calls_path = _fake_vercel(tmp_path=tmp_path, monkeypatch=monkeypatch)
    monkeypatch.setenv("STATUSES", "7")
    monkeypatch.setenv(
        "OUTPUT", "Unexpected token 'I', \"Internal S\"... is not valid JSON"
    )
    delays = []
    monkeypatch.setattr("deploy_vercel_frontend.time.sleep", delays.append)

    assert main([]) == 7
    assert len(json.loads(calls_path.read_text())) == 3
    assert delays == [2, 4]


def test_retry_does_not_run_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry only the deploy command without invoking a build."""
    calls_path = _fake_vercel(tmp_path=tmp_path, monkeypatch=monkeypatch)
    monkeypatch.setenv("STATUSES", "1,0")
    monkeypatch.setenv(
        "OUTPUT", "Unexpected token 'I', \"Internal S\"... is not valid JSON"
    )
    monkeypatch.setattr("deploy_vercel_frontend.time.sleep", lambda _: None)

    assert main([]) == 0
    assert all(args[0] == "deploy" for args in json.loads(calls_path.read_text()))


def test_success_runs_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not repeat a successful deployment."""
    calls_path = _fake_vercel(tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert main([]) == 0
    assert len(json.loads(calls_path.read_text())) == 1


def test_unrelated_failure_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave unrelated deployment errors to the CLI without a retry."""
    calls_path = _fake_vercel(tmp_path=tmp_path, monkeypatch=monkeypatch)
    monkeypatch.setenv("STATUSES", "9")
    monkeypatch.setenv("OUTPUT", "Invalid JSON response")

    assert main([]) == 9
    assert len(json.loads(calls_path.read_text())) == 1
