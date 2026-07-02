"""Tests for the `leaderboards.evaluation_common` module."""

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
