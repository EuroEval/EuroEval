"""Tests for the `leaderboards.evaluation_common` module."""

from types import SimpleNamespace

import pytest

from leaderboards import evaluation_common


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
