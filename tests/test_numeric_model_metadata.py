"""Tests for safe recovery of Hugging Face numeric model metadata."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from euroeval.numeric_model_metadata import (
    positive_context_length,
    positive_integer,
    recover_numeric_metadata,
)


def test_context_length_rejects_tokenizer_sentinel() -> None:
    """Treat enormous tokenizer model_max_length values as unavailable."""
    assert positive_context_length(1024.0) == 1024
    assert positive_context_length(10**30) is None


def test_failed_hub_lookup_returns_empty_metadata() -> None:
    """Treat transient Hub failures as unknown rather than caching sentinels."""
    with patch(
        "euroeval.numeric_model_metadata.HfApi",
        side_effect=OSError("temporary network failure"),
    ):
        with patch(
            "euroeval.numeric_model_metadata.hf_hub_download",
            side_effect=OSError("temporary network failure"),
        ):
            assert recover_numeric_metadata(model_id="org/model") == {}


def test_positive_integer_accepts_integral_numeric_representations() -> None:
    """Preserve positive integral floats and decimal strings without truncation."""
    assert positive_integer(1024.0) == 1024
    assert positive_integer("1024.0") == 1024
    assert positive_integer("1.024e3") == 1024
    assert positive_integer(1.5) is None
    assert positive_integer("1.5") is None
    assert positive_integer(True) is None
    assert positive_integer(float("inf")) is None
    assert positive_integer("nan") is None
    assert positive_integer("invalid") is None


def test_recovers_nested_config_and_safetensors_counts(tmp_path: Path) -> None:
    """Read nested text config and safetensors counts at an immutable revision."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {"text_config": {"vocab_size": 248320, "max_position_embeddings": 262144}}
        ),
        encoding="utf-8",
    )
    api = MagicMock()
    api.model_info.return_value.safetensors = {"total": 1107265600}

    with (
        patch("euroeval.numeric_model_metadata.HfApi", return_value=api),
        patch(
            "euroeval.numeric_model_metadata.hf_hub_download",
            return_value=str(config_path),
        ) as download,
    ):
        result = recover_numeric_metadata(model_id="org/model@revision")

    assert result == {
        "num_model_parameters": 1107265600,
        "vocabulary_size": 248320,
        "max_sequence_length": 262144,
    }
    api.model_info.assert_called_once_with(
        repo_id="org/model", revision="revision", expand=["safetensors"], token=None
    )
    download.assert_called_once_with(
        repo_id="org/model",
        filename="config.json",
        revision="revision",
        token=None,
        cache_dir=None,
    )


def test_recovery_uses_explicit_token_and_cache_directory(tmp_path: Path) -> None:
    """Use configured credentials and cache paths for all Hub requests."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"vocab_size": 32000}), encoding="utf-8")
    api = MagicMock()
    api.model_info.return_value.safetensors = None
    with (
        patch("euroeval.numeric_model_metadata.HfApi", return_value=api) as api_cls,
        patch(
            "euroeval.numeric_model_metadata.hf_hub_download",
            return_value=str(config_path),
        ) as download,
    ):
        recover_numeric_metadata(
            model_id="org/model", token="secret-token", cache_dir="/cache"
        )
    api_cls.assert_called_once_with(token="secret-token")
    api.model_info.assert_called_once_with(
        repo_id="org/model",
        revision="main",
        expand=["safetensors"],
        token="secret-token",
    )
    assert download.call_count == 3
    assert all(
        call.kwargs["token"] == "secret-token" and call.kwargs["cache_dir"] == "/cache"
        for call in download.call_args_list
    )


def test_tokenizer_context_sentinel_does_not_hide_fallback(tmp_path: Path) -> None:
    """Reject tokenizer sentinels and recover context from tokenizer-only config."""
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(
        json.dumps({"model_max_length": 10**30, "vocab_size": 32000}), encoding="utf-8"
    )
    api = MagicMock()
    api.model_info.return_value.safetensors = None
    with (
        patch("euroeval.numeric_model_metadata.HfApi", return_value=api),
        patch(
            "euroeval.numeric_model_metadata.hf_hub_download",
            side_effect=[FileNotFoundError, FileNotFoundError, str(tokenizer_config)],
        ),
    ):
        result = recover_numeric_metadata(model_id="org/model")
    assert result == {"vocabulary_size": 32000}
