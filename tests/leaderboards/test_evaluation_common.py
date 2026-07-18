"""Tests for the `leaderboards.evaluation_common` module."""

import os
import pty
import select
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from leaderboards import evaluation_common
from leaderboards.evaluation_common import _killed_by_signal_note


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


@pytest.mark.parametrize(argnames=["returncode"], argvalues=[(None,), (0,)])
def test_killed_by_signal_note_returns_none_when_not_signal_death(
    returncode: int | None,
) -> None:
    """A non-negative or absent returncode is not a signal death."""
    assert _killed_by_signal_note(returncode=returncode) is None


def test_killed_by_signal_note_ignores_shell_exit_code_convention() -> None:
    """``Popen.returncode`` is negative for signal deaths, never 128+N.

    The shell convention (e.g. 130 = 128 + SIGINT) only appears in ``$?``;
    ``subprocess.Popen.returncode`` reports ``-N`` instead, so a positive 130
    is an ordinary exit code and must not be misread as a signal death.
    """
    assert _killed_by_signal_note(returncode=130) is None


@pytest.mark.parametrize(
    argnames=["returncode", "must_contain"],
    argvalues=[
        (-2, ["signal 2", "SIGINT"]),
        (-9, ["signal 9 (SIGKILL)", "out of memory"]),
    ],
)
def test_killed_by_signal_note_describes_signal_death(
    returncode: int, must_contain: list[str]
) -> None:
    """A negative returncode yields a diagnostic naming the signal."""
    note = _killed_by_signal_note(returncode=returncode)
    assert note is not None
    for needle in must_contain:
        assert needle in note


def test_pty_drain_handles_exited_parent_with_open_grandchild(tmp_path: Path) -> None:
    """Regression test for the PTY drain hang.

    When a subprocess spawns a grandchild that inherits the PTY slave fd,
    the grandchild can keep the PTY open after the parent exits. The old
    blocking os.read() drain would hang forever waiting for EOF. This test
    verifies the select-driven drain with a counter completes quickly and
    returns the correct non-zero return code.

    The parent script spawns a background sleep (grandchild) that inherits
    stdio and keeps the PTY open, then exits with code 42. The drain should
    complete after a few empty reads, not wait for the sleep to finish.
    """
    # Parent spawns background child that keeps PTY open, then exits non-zero
    parent_script = tmp_path / "parent.sh"
    parent_script.write_text(
        "sleep 10 &\n"  # Grandchild keeps PTY slave open
        "echo 'parent exiting'\n"
        "exit 42\n"
    )
    parent_script.chmod(0o755)

    parent_fd, child_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            ["bash", str(parent_script)],
            stdin=child_fd,
            stdout=child_fd,
            stderr=child_fd,
            close_fds=True,
        )
        os.close(child_fd)

        # Drain loop - mirrors the fixed code in run_euroeval
        captured: list[bytes] = []
        drained_empty_reads = 0
        max_empty_reads_after_exit = 5
        start_time = time.time()

        while True:
            ready, _, _ = select.select([parent_fd], [], [], 0.1)
            try:
                chunk = os.read(parent_fd, 4096) if ready else b""
            except OSError:
                break

            if chunk:
                drained_empty_reads = 0
                captured.append(chunk)
            elif proc.poll() is not None:
                # Parent exited with grandchild still holding PTY open.
                # Break after N empty reads to avoid hanging.
                drained_empty_reads += 1
                if drained_empty_reads > max_empty_reads_after_exit:
                    break
            # else: process still running, continue waiting

        elapsed = time.time() - start_time
        proc.wait()  # Reap the parent (grandchild may still be running)

        # Assertions
        assert proc.returncode == 42, "Parent should exit with code 42"
        assert elapsed < 2.0, f"Drain should complete quickly, took {elapsed:.2f}s"
        output = b"".join(captured).decode("utf-8", errors="replace")
        assert "parent exiting" in output, "Should capture parent output"
    finally:
        os.close(parent_fd)
        # Clean up background sleep process
        try:
            subprocess.run(
                ["pkill", "-f", "sleep 10"], capture_output=True, timeout=1, check=False
            )
        except subprocess.TimeoutExpired:
            pass
