"""Tests for safe recovery of Hugging Face numeric model metadata."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from euroeval.numeric_model_metadata import recover_numeric_metadata


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
        repo_id="org/model", revision="revision", expand=["safetensors"]
    )
    download.assert_called_once_with(
        repo_id="org/model", filename="config.json", revision="revision"
    )
