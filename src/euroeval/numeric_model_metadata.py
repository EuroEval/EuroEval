"""Recover numeric model metadata from safe Hugging Face metadata."""

from __future__ import annotations

import json
import logging
from decimal import Decimal, InvalidOperation
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
_MAX_CONTEXT_LENGTH = 1_000_000_000


def recover_numeric_metadata(
    model_id: str, *, token: str | None = None, cache_dir: str | None = None
) -> dict[str, int]:
    """Read parameter, vocabulary, and context metadata without loading model code.

    Args:
        model_id:
            Hugging Face repository ID, optionally followed by a revision.
        token (optional):
            Hugging Face access token for gated repositories.
        cache_dir (optional):
            Directory for downloaded config files.

    Returns:
        Numeric metadata fields found in Hub safetensors and JSON config metadata.
    """
    components = split_model_id(model_id=model_id)
    recovered: dict[str, int] = {}
    try:
        info = HfApi(token=token).model_info(
            repo_id=components.model_id,
            revision=components.revision,
            expand=["safetensors"],
            token=token,
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
                token=token,
                cache_dir=cache_dir,
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
                    parsed = (
                        positive_context_length(value)
                        if field == "max_sequence_length"
                        else positive_integer(value)
                    )
                    if parsed is not None:
                        recovered[field] = parsed
                        break
                if field in recovered:
                    break
        if all(field in recovered for field in _NUMERIC_KEYS):
            break

    return recovered


def positive_context_length(value: object) -> int | None:
    """Return a plausible positive context length, rejecting tokenizer sentinels.

    Args:
        value:
            Candidate context length.

    Returns:
        A plausible context length, or None when missing, invalid, or a sentinel.
    """
    parsed = positive_integer(value)
    return parsed if parsed is not None and parsed <= _MAX_CONTEXT_LENGTH else None


def positive_integer(value: object) -> int | None:
    """Return a positive integral numeric value, treating sentinels as missing.

    Args:
        value:
            Candidate numeric metadata.

    Returns:
        A positive integer, or None when the value is missing or invalid.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite() or parsed <= 0 or parsed != parsed.to_integral_value():
        return None
    try:
        return int(parsed)
    except ValueError:
        return None
