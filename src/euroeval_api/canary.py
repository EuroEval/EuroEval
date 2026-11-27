"""Private completion-canary protocol and corpus access."""

import hashlib
import os
import re
import typing as t
import unicodedata

import httpx
from fastapi import HTTPException

from .dependencies import fetch_with_retry

CANARY_PROTOCOL_VERSION = "private-completion-canary/v1"
CANARY_CORPUS_ID = "EuroEval/watermark-audit"
CANARY_CORPUS_REVISION = "16d468bbacc284c912a8598a392239af2387ca53"
CANARY_CORPUS_SHA256 = (
    "37258fb324cf400bba4bd57cda430a73393928f5e09506594c3678adc38ff324"
)
CANARY_EVIDENCE_SCHEMA = "contamination-canary-evidence/v1"
CANARY_NORMALISER_VERSION = "first-two-words-nfc/v1"
CANARY_GENERATION_VERSION = "greedy-continuation/v1"
CANARY_RESULT_DATASET = "contamination-canary"


class CanaryLease(t.TypedDict, total=False):
    """A lease describing whether canary generation is required."""

    protocol_version: str
    corpus_revision: str
    corpus_sha256: str
    status: t.Literal["required"]
    reason: t.Literal["encoder"]


_corpus: str | None = None
_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "normaliser_version",
        "generation_version",
        "model_id",
        "requested_revision",
        "resolved_revision",
        "identity_kind",
        "backend",
        "corpus_id",
        "corpus_revision",
        "corpus_sha256",
        "row_count",
        "status",
        "reason",
        "observations",
    }
)


async def fetch_canary_corpus(http: httpx.AsyncClient) -> str:
    """Fetch and verify the immutable private canary corpus.

    Args:
        http:
            Shared outbound HTTP client.

    Returns:
        Verified UTF-8 corpus text.

    Raises:
        HTTPException:
            If credentials are missing, the provider fails, or content is invalid.
    """
    global _corpus
    if _corpus is not None:
        return _corpus
    token = os.environ.get("HF_CANARY_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise HTTPException(status_code=503, detail="Canary corpus is unavailable.")
    url = (
        f"https://huggingface.co/datasets/{CANARY_CORPUS_ID}/resolve/"
        f"{CANARY_CORPUS_REVISION}/test.jsonl"
    )
    try:
        response = await fetch_with_retry(
            http=http,
            url=url,
            headers={
                "authorization": f"Bearer {token}",
                "accept": "application/x-ndjson",
            },
        )
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502, detail="Canary corpus provider failed."
        ) from error
    if not response.is_success:
        raise HTTPException(status_code=502, detail="Canary corpus provider failed.")
    try:
        content = response.content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=502, detail="Canary corpus provider returned invalid text."
        ) from error
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != CANARY_CORPUS_SHA256:
        raise HTTPException(
            status_code=503, detail="Canary corpus does not match its frozen digest."
        )
    if len(content.rstrip().split("\n")) != 256:
        raise HTTPException(
            status_code=503, detail="Canary corpus must contain 256 rows."
        )
    _corpus = content
    return content


def reserve_canary(model_type: str) -> CanaryLease:
    """Reserve a canary lease based on model type.

    Args:
        model_type:
            Model type, with ``generative`` denoting a decoder.

    Returns:
        A protocol-pinned required lease.
    """
    lease: CanaryLease = {
        "protocol_version": CANARY_PROTOCOL_VERSION,
        "corpus_revision": CANARY_CORPUS_REVISION,
        "corpus_sha256": CANARY_CORPUS_SHA256,
        "status": "required",
    }
    if model_type != "generative":
        lease["reason"] = "encoder"
    return lease


def validate_canary_evidence(value: object, model_id: str, revision: str) -> None:
    """Validate canary evidence against the frozen protocol.

    Args:
        value:
            Untrusted evidence payload.
        model_id:
            Expected model identifier.
        revision:
            Expected immutable model revision.
    """
    if not isinstance(value, dict):
        _invalid("Canary evidence must be an object.")
    evidence = t.cast(dict[str, object], value)
    if set(evidence) != _EVIDENCE_FIELDS:
        _invalid("Canary evidence contains undeclared or missing fields.")
    if (
        evidence["schema_version"] != CANARY_EVIDENCE_SCHEMA
        or evidence["protocol_version"] != CANARY_PROTOCOL_VERSION
        or evidence["normaliser_version"] != CANARY_NORMALISER_VERSION
        or evidence["generation_version"] != CANARY_GENERATION_VERSION
        or evidence["model_id"] != model_id
        or evidence["requested_revision"] != revision
        or evidence["resolved_revision"] != revision
        or evidence["identity_kind"] != "immutable"
        or evidence["corpus_id"] != CANARY_CORPUS_ID
        or evidence["corpus_revision"] != CANARY_CORPUS_REVISION
        or evidence["corpus_sha256"] != CANARY_CORPUS_SHA256
        or type(evidence["row_count"]) is not int
        or evidence["row_count"] != 256
        or evidence["status"]
        not in {"collected", "not_applicable", "unsupported", "failed"}
        or not isinstance(evidence["backend"], str)
        or not evidence["backend"]
    ):
        _invalid("Canary evidence identity or protocol is invalid.")
    observations = evidence["observations"]
    if not isinstance(observations, list):
        _invalid("Canary observations must be an array.")
    status, reason = evidence["status"], evidence["reason"]
    if status != "collected":
        valid = (
            (status == "not_applicable" and reason == "encoder")
            or (status == "unsupported" and reason == "backend_unsupported")
            or (
                status == "failed"
                and reason
                in {"corpus_unavailable", "generation_failed", "incomplete_generation"}
            )
        )
        if observations or not valid:
            _invalid("Non-collected canary evidence is malformed.")
        return
    if reason is not None or len(observations) != 256:
        _invalid("Collected canary evidence must contain 256 rows.")
    row_ids: set[str] = set()
    for item in observations:
        if not isinstance(item, dict) or set(item) != {
            "row_id",
            "prompt_sha256",
            "normalised_completion",
        }:
            _invalid("Canary observation is invalid.")
        row = t.cast(dict[str, object], item)
        row_id, digest, completion = (
            row["row_id"],
            row["prompt_sha256"],
            row["normalised_completion"],
        )
        if (
            not isinstance(row_id, str)
            or not row_id
            or row_id in row_ids
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(completion, str)
            or len(completion.encode("utf-8")) > 256
            or not _valid_completion(completion)
        ):
            _invalid("Canary observation is invalid.")
        row_ids.add(row_id)


def _invalid(detail: str) -> t.NoReturn:
    raise HTTPException(status_code=422, detail=detail)


def _valid_completion(text: str) -> bool:
    """Check the Unicode-letter first-two-words normalisation form.

    Args:
        text:
            Normalised completion.

    Returns:
        Whether the completion matches the protocol.
    """
    if text == "":
        return True
    words = text.split(" ")
    return len(words) <= 2 and all(
        word
        and all(
            char in "-'" or unicodedata.category(char).startswith("L") for char in word
        )
        and word[0] not in "-'"
        and word[-1] not in "-'"
        and "--" not in word
        and "''" not in word
        and "-'" not in word
        and "'-" not in word
        for word in words
    )
