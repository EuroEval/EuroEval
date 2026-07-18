"""Tests for the `leaderboards.evaluation_common` module."""

import collections.abc as c
import os
import signal
import subprocess
import time
import typing as t
from contextlib import contextmanager
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


def test_pty_drain_handles_exited_parent_with_open_grandchild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the PTY drain hang.

    When a subprocess spawns a grandchild that inherits the PTY slave fd,
    the grandchild can keep the PTY open after the parent exits. The drain
    loop must bound the post-parent-exit wait with a fixed deadline to avoid
    hanging indefinitely on chatty grandchildren.

    This test exercises ``run_euroeval()`` directly by monkeypatching
    ``subprocess.Popen`` to spawn a shell script that creates a background
    grandchild keeping the PTY open.
    """
    parent_script = tmp_path / "parent.sh"
    parent_script.write_text(
        "# Background grandchild inherits PTY and keeps it open\n"
        "sleep 60 &\n"
        "# Parent exits quickly\n"
        "echo 'parent output'\n"
        "exit 42\n"
    )
    parent_script.chmod(0o755)

    # Mock slow/optional dependencies to avoid heavy imports
    @contextmanager
    def _mock_no_terminal_output(disable: bool = True) -> c.Generator[None, None, None]:
        yield

    def _mock_resolve_hf_token() -> None:
        return None

    # Patch before calling run_euroeval
    monkeypatch.setattr(
        "euroeval.logging_utils.no_terminal_output",
        _mock_no_terminal_output,
        raising=False,
    )
    monkeypatch.setattr(
        "leaderboards.evaluation_common.resolve_hf_token", _mock_resolve_hf_token
    )

    # Track and cleanup spawned processes via process group
    spawned_pgid: int | None = None
    original_popen = subprocess.Popen

    def mock_popen(
        cmd: list[str],
        *args: t.Any,  # noqa: ANN002, ANN401
        **kwargs: t.Any,  # noqa: ANN003, ANN401
    ) -> subprocess.Popen:
        nonlocal spawned_pgid
        if cmd and cmd[0] == "euroeval":
            cmd = ["bash", str(parent_script)]
            # Start in new process group for scoped cleanup
            kwargs["preexec_fn"] = os.setsid
        proc = original_popen(cmd, *args, **kwargs)
        if cmd and cmd[0] == "bash":
            spawned_pgid = os.getpgid(proc.pid)
        return proc

    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    start_time = time.time()

    # Call run_euroeval directly
    returncode, output = evaluation_common.run_euroeval(
        model_id="test-model", languages=["en"], stream_output=False
    )

    elapsed = time.time() - start_time

    # Scoped cleanup of process group (no broad pkill)
    if spawned_pgid is not None:
        try:
            os.killpg(spawned_pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

    # Assertions
    assert returncode == 42, "Should return the parent's exit code"
    assert elapsed < 3.0, f"Drain should complete within deadline, took {elapsed:.2f}s"
    assert "parent output" in output, "Should capture parent's output"
