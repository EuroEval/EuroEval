"""Resolve public Hugging Face model metadata safely."""

from __future__ import annotations

import asyncio
import json
import math
import re
import typing as t
from urllib.parse import quote

import httpx
from fastapi import HTTPException

from .dependencies import fetch_with_retry
from .policy import ModelType


class ModelMetadataEvidence(t.TypedDict):
    """Metadata evidence used to classify a resolved model."""

    pipeline_tag: str
    architectures: list[str]
    model_type: ModelType
    is_encoder_decoder: bool | None


class ResolvedModel(t.TypedDict):
    """Validated immutable model metadata and repository size estimates."""

    id: str
    revision: str
    config: dict[str, object]
    weight_bytes: int
    repo_bytes: int
    model_type: ModelType
    model_metadata: ModelMetadataEvidence


async def model_may_fit(
    http: httpx.AsyncClient,
    model_id: str,
    hardware: dict[str, object],
    selected: dict[str, object] | None = None,
) -> bool:
    """Estimate whether a model fits available hardware without resolving its config.

    Args:
        http:
            Shared asynchronous HTTP client.
        model_id:
            Requested Hugging Face model identifier.
        hardware:
            Hardware record containing disk capacity and GPU utilisation.
        selected (optional):
            Selected GPU record, if any. Defaults to None.

    Returns:
        Whether the model appears safe and small enough for the hardware.

    Raises:
        HTTPException:
            If Hugging Face cannot supply complete metadata or file sizes.
    """
    encoded = "/".join(quote(part, safe="") for part in model_id.split("/"))
    info = await _get_json(
        http=http,
        url=f"https://huggingface.co/api/models/{encoded}?blobs=true",
        failure=f"Hugging Face model {model_id} could not be inspected.",
    )
    returned_id, private, gated = info.get("id"), info.get("private"), info.get("gated")
    siblings = info.get("siblings")
    if (
        not isinstance(returned_id, str)
        or returned_id.lower() != model_id.lower()
        or not isinstance(private, bool)
        or not (isinstance(gated, bool) or gated == "false")
        or not isinstance(siblings, list)
        or not siblings
    ):
        raise HTTPException(status_code=422, detail="Incomplete model metadata.")
    if private or gated is True:
        return False

    files: list[tuple[str, int]] = []
    for item in siblings:
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=422, detail="Incomplete model file metadata."
            )
        name, size = item.get("rfilename"), item.get("size")
        if not isinstance(name, str) or not name.strip():
            raise HTTPException(
                status_code=422, detail="Incomplete model file metadata."
            )
        lower = name.lower()
        if lower.endswith(".py") or re.search(
            r"\.(bin|pt|pth|ckpt|gguf|onnx|h5|msgpack)$", lower
        ):
            return False
        if (
            isinstance(size, bool)
            or not isinstance(size, (int, float))
            or not math.isfinite(size)
            or (isinstance(size, float) and not size.is_integer())
        ):
            raise HTTPException(status_code=422, detail="Incomplete model file sizes.")
        if size < 0 or size > 9_007_199_254_740_991:
            raise HTTPException(status_code=422, detail="Incomplete model file sizes.")
        files.append((name, int(size)))

    weights = [size for name, size in files if name.lower().endswith(".safetensors")]
    weight_bytes, repo_bytes = sum(weights), sum(size for _, size in files)
    if not weights or not weight_bytes or not repo_bytes:
        return False
    free_disk, utilisation = (
        hardware.get("free_disk_bytes"),
        hardware.get("gpu_memory_utilisation"),
    )
    free_memory = selected.get("free_memory_bytes") if selected else 0
    return (
        isinstance(free_disk, (int, float))
        and not isinstance(free_disk, bool)
        and free_disk >= repo_bytes
        and isinstance(utilisation, (int, float))
        and not isinstance(utilisation, bool)
        and isinstance(free_memory, (int, float))
        and not isinstance(free_memory, bool)
        and weight_bytes * 1.35 <= free_memory * utilisation
    )


async def _get_json(
    *, http: httpx.AsyncClient, url: str, failure: str
) -> dict[str, object]:
    """Fetch and decode a metadata object within a total request deadline.

    Args:
        http:
            Shared asynchronous HTTP client.
        url:
            Fully encoded Hugging Face metadata URL.
        failure:
            Description for an unsuccessful HTTP status.

    Returns:
        Decoded JSON object.

    Raises:
        ValueError:
            If the response is not a JSON object.
        HTTPException:
            If Hugging Face returns an unsuccessful status.
    """
    response = await asyncio.wait_for(fetch_with_retry(http=http, url=url), timeout=2.5)
    if response.status_code >= 400:
        status = (
            503 if response.status_code >= 500 or response.status_code == 429 else 422
        )
        raise HTTPException(status_code=status, detail=failure)
    try:
        value = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Hugging Face returned invalid JSON metadata.") from error
    if not isinstance(value, dict):
        raise ValueError("Hugging Face returned incomplete metadata.")
    return value


async def resolve_model(http: httpx.AsyncClient, model_id: str) -> ResolvedModel:
    """Resolve a public, safe model at an immutable revision.

    Args:
        http:
            Shared asynchronous HTTP client.
        model_id:
            Requested Hugging Face model identifier.

    Returns:
        Validated model metadata with byte estimates.

    Raises:
        ValueError:
            If metadata is incomplete, unsafe, or contradictory.
    """
    encoded = "/".join(quote(part, safe="") for part in model_id.split("/"))
    info = await _get_json(
        http=http,
        url=f"https://huggingface.co/api/models/{encoded}?blobs=true",
        failure="Hugging Face model metadata could not be resolved.",
    )
    files = _model_info(info=info, model_id=model_id)
    pipeline = info.get("pipeline_tag")
    if not isinstance(pipeline, str) or not pipeline.strip():
        raise ValueError("Hugging Face omitted an unambiguous pipeline tag.")
    revision = info.get("sha")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Hugging Face did not provide an immutable model revision.")

    config = await _get_json(
        http=http,
        url=(
            f"https://huggingface.co/{encoded}/raw/"
            f"{quote(revision, safe='')}/config.json"
        ),
        failure="The model config could not be resolved at its immutable revision.",
    )
    if any(
        _js_truthy(config.get(key))
        for key in ("auto_map", "custom_code", "trust_remote_code")
    ):
        raise ValueError("Models with auto_map or custom code are not supported.")
    model_name = config.get("model_type")
    architectures = config.get("architectures")
    encoder_decoder = config.get("is_encoder_decoder")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("The model config omitted model_type.")
    if (
        not isinstance(architectures, list)
        or not architectures
        or any(not isinstance(item, str) or not item for item in architectures)
    ):
        raise ValueError("The model config has no unambiguous architectures.")
    if "is_encoder_decoder" in config and not isinstance(encoder_decoder, bool):
        raise ValueError("The model config has an invalid encoder-decoder flag.")
    model_type = _model_type_for(pipeline, encoder_decoder)
    if model_type is None:
        raise ValueError("The model has contradictory capability metadata.")

    weights = [
        file for file in files if file["rfilename"].lower().endswith(".safetensors")
    ]
    weight_bytes = sum(file["size"] for file in weights)
    repo_bytes = sum(file["size"] for file in files)
    if not weight_bytes or not repo_bytes:
        raise ValueError("Model files must have positive complete sizes.")
    evidence: ModelMetadataEvidence = {
        "pipeline_tag": pipeline,
        "architectures": architectures,
        "model_type": model_type,
        "is_encoder_decoder": encoder_decoder
        if isinstance(encoder_decoder, bool)
        else None,
    }
    return {
        "id": model_id,
        "revision": revision,
        "config": config,
        "weight_bytes": weight_bytes,
        "repo_bytes": repo_bytes,
        "model_type": model_type,
        "model_metadata": evidence,
    }


def _js_truthy(value: object) -> bool:
    """Apply JavaScript truthiness for JSON values.

    Args:
        value:
            JSON value to inspect.

    Returns:
        Whether JavaScript treats the value as truthy.
    """
    return value is not None and value is not False and value != 0 and value != ""


def _model_info(*, info: dict[str, object], model_id: str) -> list[dict[str, object]]:
    """Validate model identity, public access, and sized repository files.

    Args:
        info:
            Raw Hugging Face model metadata.
        model_id:
            Requested model identifier.

    Returns:
        Validated file entries.

    Raises:
        ValueError:
            If identity, access, or file metadata is invalid.
    """
    returned_id = info.get("id")
    if not isinstance(returned_id, str) or returned_id.lower() != model_id.lower():
        raise ValueError("Hugging Face returned incomplete model metadata.")
    gated = info.get("gated")
    if not isinstance(info.get("private"), bool) or not (
        isinstance(gated, bool) or gated == "false"
    ):
        raise ValueError("Hugging Face omitted public access metadata.")
    if info["private"] or gated is True:
        raise ValueError("The model must be public and ungated.")
    siblings = info.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        raise ValueError("Hugging Face response omitted repository files.")
    files: list[dict[str, object]] = []
    for item in siblings:
        if not isinstance(item, dict):
            raise ValueError("Hugging Face response omitted repository files.")
        name, size = item.get("rfilename"), item.get("size")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Hugging Face response omitted repository files.")
        if (
            isinstance(size, bool)
            or not isinstance(size, (int, float))
            or not math.isfinite(size)
            or int(size) != size
            or size < 0
            or size > 9_007_199_254_740_991
        ):
            raise ValueError(
                "Hugging Face did not provide complete sizes for a fit estimate."
            )
        lower = name.lower()
        if lower.endswith(".py") or re.search(
            r"\.(bin|pt|pth|ckpt|gguf|onnx|h5|msgpack)$", lower
        ):
            raise ValueError(
                "The model must use safetensors and contain no custom Python."
            )
        files.append({"rfilename": name, "size": int(size)})
    if not any(file["rfilename"].lower().endswith(".safetensors") for file in files):
        raise ValueError("The model must contain safetensors weights.")
    return files


def _model_type_for(pipeline_tag: str, is_encoder_decoder: object) -> ModelType | None:
    """Classify a pipeline tag while rejecting contradictory config evidence.

    Args:
        pipeline_tag:
            Hugging Face pipeline tag.
        is_encoder_decoder:
            Optional encoder-decoder config value.

    Returns:
        Model type, or None when evidence contradicts.
    """
    generative_tags = {
        "text-generation",
        "text2text-generation",
        "conversational",
        "image-text-to-text",
        "text-to-audio",
        "automatic-speech-recognition",
    }
    if pipeline_tag in generative_tags:
        return "generative"
    return None if is_encoder_decoder is True else "encoder"
