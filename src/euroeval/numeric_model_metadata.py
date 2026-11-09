"""Recover numeric model metadata from safe Hugging Face metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .string_utils import split_model_id

logger = logging.getLogger(__name__)

_NUMERIC_KEYS = {
    "num_model_parameters": ("parameters",),
    "vocabulary_size": ("vocab_size",),
    "max_sequence_length": (
        "max_position_embeddings",
        "model_max_length",
        "n_positions",
        "max_seq_len",
    ),
}
_CONFIG_FILES = ("config.json", "encoder/config.json", "tokenizer_config.json")


def positive_integer(value: object) -> int | None:
    """Return a positive integer value, treating sentinels as missing.

    Args:
        value:
            Candidate numeric metadata.

    Returns:
        A positive integer, or None when the value is missing or invalid.
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def recover_numeric_metadata(model_id: str) -> dict[str, int]:
    """Read parameter, vocabulary, and context metadata without loading model code.

    Args:
        model_id:
            Hugging Face repository ID, optionally followed by a revision.

    Returns:
        Numeric metadata fields found in Hub safetensors and JSON config metadata.
    """
    components = split_model_id(model_id=model_id)
    recovered: dict[str, int] = {}
    try:
        info = HfApi().model_info(
            repo_id=components.model_id,
            revision=components.revision,
            expand=["safetensors"],
        )
        safetensors = info.safetensors
        if safetensors is not None:
            total = (
                safetensors.get("total")
                if isinstance(safetensors, dict)
                else getattr(safetensors, "total", None)
            )
            if isinstance(total, int) and total > 0:
                recovered["num_model_parameters"] = total
    except Exception as error:  # noqa: BLE001 - Hub metadata is best effort
        logger.debug("Could not recover parameter count for %s: %s", model_id, error)

    for filename in _CONFIG_FILES:
        try:
            config_path = hf_hub_download(
                repo_id=components.model_id,
                filename=filename,
                revision=components.revision,
            )
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - absent/inaccessible configs are expected
            continue
        if not isinstance(config, dict):
            continue
        candidates = [config]
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            candidates.insert(0, text_config)
        for field, keys in _NUMERIC_KEYS.items():
            if field in recovered:
                continue
            for candidate in candidates:
                for key in keys:
                    value = candidate.get(key)
                    if (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value > 0
                    ):
                        recovered[field] = value
                        break
                if field in recovered:
                    break
        if all(field in recovered for field in _NUMERIC_KEYS):
            break

    return recovered
