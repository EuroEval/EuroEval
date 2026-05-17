"""Tests for process_evaluation_queue utilities."""

from types import SimpleNamespace

import pytest

from src.scripts import process_evaluation_queue


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
    monkeypatch: pytest.MonkeyPatch,
    dtype: str,
    count: int,
    expected_bytes: int,
) -> None:
    """Unknown dtypes should use numeric fallback, then BF16 fallback."""

    def fake_get_safetensors_metadata(
        *, repo_id: str, revision: str | None = None, token: str | None = None
    ) -> SimpleNamespace:
        _ = repo_id, revision, token
        return SimpleNamespace(parameter_count={dtype: count})

    monkeypatch.setattr(
        process_evaluation_queue,
        "get_safetensors_metadata",
        fake_get_safetensors_metadata,
    )
    bytes_needed = process_evaluation_queue.estimated_model_bytes(
        model_id="org/model-name"
    )
    assert bytes_needed == expected_bytes


@pytest.mark.parametrize(
    argnames=["model_id"],
    argvalues=[
        ("org/model#low@rev",),
        ("org/model@rev#low",),
    ],
)
def test_estimated_model_bytes_handles_model_id_extras(
    monkeypatch: pytest.MonkeyPatch, model_id: str
) -> None:
    """Model id parsing should handle # and @ in either order."""
    calls: list[dict[str, str | None]] = []

    def fake_get_safetensors_metadata(
        *, repo_id: str, revision: str | None = None, token: str | None = None
    ) -> SimpleNamespace:
        calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "token": token,
            }
        )
        return SimpleNamespace(parameter_count={"BF16": 1})

    monkeypatch.setattr(
        process_evaluation_queue,
        "get_safetensors_metadata",
        fake_get_safetensors_metadata,
    )
    assert process_evaluation_queue.estimated_model_bytes(model_id=model_id) == 2
    assert calls[0]["repo_id"] == "org/model"
    assert calls[0]["revision"] == "rev"
