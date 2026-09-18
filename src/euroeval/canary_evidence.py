"""Versioned, plaintext-free production evidence for contamination canaries."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import typing as t
import unicodedata
from pathlib import Path

from huggingface_hub import hf_hub_download

CANARY_EVIDENCE_SCHEMA = "contamination-canary-evidence/v1"
CANARY_COLLECTION_PROTOCOL = "private-completion-canary/v1"
CANARY_NORMALISER_VERSION = "first-two-words-nfc/v1"
CANARY_GENERATION_VERSION = "greedy-continuation/v1"
CANARY_DATASET_ID = "EuroEval/watermark-audit"
CANARY_DATASET_REVISION = "16d468bbacc284c912a8598a392239af2387ca53"
CANARY_DATASET_FILENAME = "test.jsonl"
CANARY_CORPUS_SHA256 = (
    "37258fb324cf400bba4bd57cda430a73393928f5e09506594c3678adc38ff324"
)
CANARY_ROW_COUNT = 256
CANARY_GROUP_COUNT = 32
CANARY_RESULT_DATASET = "contamination-canary"
CANARY_RESULT_TASK = "contamination-detection"
CANARY_CORPUS_PATH_ENV = "EUROEVAL_CANARY_CORPUS_PATH"
CANARY_PRIVATE_DIR_ENV = "EUROEVAL_CANARY_PRIVATE_DIR"

_PROMPT_RE = re.compile(
    r"^(?P<prompt>[\s\S]+ referred to) (?P<first>[a-z]+) (?P<second>[a-z]+)\.$"
)
_WORD_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", flags=re.UNICODE)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_STATUSES = {"collected", "not_applicable", "unsupported", "failed"}
_ALLOWED_REASONS = {
    "encoder",
    "corpus_unavailable",
    "backend_unsupported",
    "generation_failed",
    "incomplete_generation",
}


@dataclasses.dataclass(frozen=True)
class CanaryPrompt:
    """One validated production prompt derived from the frozen augmented corpus."""

    row_id: str
    prompt: str
    prompt_sha256: str


@dataclasses.dataclass(frozen=True)
class CanaryObservation:
    """One plaintext-free canary completion observation."""

    row_id: str
    prompt_sha256: str
    normalised_completion: str


@dataclasses.dataclass(frozen=True)
class CanaryEvidence:
    """One complete model-level canary collection attempt."""

    model_id: str
    requested_revision: str
    resolved_revision: str
    identity_kind: t.Literal["immutable", "mutable"]
    backend: str
    status: t.Literal["collected", "not_applicable", "unsupported", "failed"]
    reason: str | None = None
    observations: tuple[CanaryObservation, ...] = ()
    schema_version: str = CANARY_EVIDENCE_SCHEMA
    protocol_version: str = CANARY_COLLECTION_PROTOCOL
    normaliser_version: str = CANARY_NORMALISER_VERSION
    generation_version: str = CANARY_GENERATION_VERSION
    corpus_id: str = CANARY_DATASET_ID
    corpus_revision: str = CANARY_DATASET_REVISION
    corpus_sha256: str = CANARY_CORPUS_SHA256
    row_count: int = CANARY_ROW_COUNT

    @property
    def identity(self) -> str:
        """The deterministic cache/storage identity for this evidence."""
        value = {
            "model_id": self.model_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "identity_kind": self.identity_kind,
            "backend": self.backend,
            "protocol_version": self.protocol_version,
            "normaliser_version": self.normaliser_version,
            "generation_version": self.generation_version,
            "corpus_revision": self.corpus_revision,
            "corpus_sha256": self.corpus_sha256,
        }
        return hashlib.sha256(canonical_json(value).encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Serialise and validate the evidence as its strict public contract.

        Returns:
            The JSON-compatible evidence object.
        """
        validate_evidence(self)
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "normaliser_version": self.normaliser_version,
            "generation_version": self.generation_version,
            "model_id": self.model_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "identity_kind": self.identity_kind,
            "backend": self.backend,
            "corpus_id": self.corpus_id,
            "corpus_revision": self.corpus_revision,
            "corpus_sha256": self.corpus_sha256,
            "row_count": self.row_count,
            "status": self.status,
            "reason": self.reason,
            "observations": [dataclasses.asdict(item) for item in self.observations],
        }


def canonical_json(value: object) -> str:
    """Encode JSON deterministically for hashing and exact-byte retries.

    Returns:
        The canonical compact JSON string.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def normalise_completion(value: str) -> str:
    """Extract the first two lexical words from a bounded model continuation.

    Returns:
        At most the first two NFC-normalised lexical words.

    Raises:
        TypeError:
            If the completion is not text.
    """
    if not isinstance(value, str):
        raise TypeError("canary completion must be text")
    normalised = unicodedata.normalize("NFC", value[:4096])
    words = _WORD_RE.findall(normalised)
    return " ".join(words[:2])


def load_canary_prompts(
    *, cache_dir: str | Path, token: str | None = None, corpus_path: Path | None = None
) -> tuple[CanaryPrompt, ...]:
    """Load and validate the frozen private corpus, then derive completion prompts.

    Returns:
        The ordered prompt records derived from the frozen corpus.

    Raises:
        ValueError:
            If the corpus digest, schema, template, identities, or row count is invalid.
    """
    configured = corpus_path or _configured_corpus_path()
    if configured is None:
        downloaded = hf_hub_download(
            repo_id=CANARY_DATASET_ID,
            repo_type="dataset",
            revision=CANARY_DATASET_REVISION,
            filename=CANARY_DATASET_FILENAME,
            cache_dir=str(cache_dir),
            token=token,
        )
        configured = Path(downloaded)
    payload = configured.read_bytes()
    if hashlib.sha256(payload).hexdigest() != CANARY_CORPUS_SHA256:
        raise ValueError("canary corpus does not match its frozen SHA-256")
    prompts: list[CanaryPrompt] = []
    row_ids: set[str] = set()
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid canary JSON on line {line_number}") from error
        if not isinstance(row, dict) or set(row) != {"row_id", "text"}:
            raise ValueError("canary rows must contain only row_id and text")
        row_id = row["row_id"]
        text = row["text"]
        if (
            not isinstance(row_id, str)
            or not row_id
            or row_id in row_ids
            or not isinstance(text, str)
        ):
            raise ValueError("canary row identity or text is invalid")
        match = _PROMPT_RE.fullmatch(text)
        if match is None:
            raise ValueError(
                "canary text does not match the frozen completion template"
            )
        prompt = match.group("prompt")
        row_ids.add(row_id)
        prompts.append(
            CanaryPrompt(
                row_id=row_id,
                prompt=prompt,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            )
        )
    if len(prompts) != CANARY_ROW_COUNT:
        raise ValueError("canary corpus must contain exactly 256 rows")
    return tuple(prompts)


def collected_evidence(
    *,
    model_id: str,
    requested_revision: str,
    resolved_revision: str,
    backend: str,
    prompts: t.Sequence[CanaryPrompt],
    completions: t.Sequence[str],
) -> CanaryEvidence:
    """Build complete evidence from model continuations without retaining prompts.

    Returns:
        Validated collected evidence.

    Raises:
        ValueError:
            If the number or contents of prompts and completions are invalid.
    """
    if len(prompts) != CANARY_ROW_COUNT or len(completions) != CANARY_ROW_COUNT:
        raise ValueError("collected canary evidence requires exactly 256 completions")
    evidence = CanaryEvidence(
        model_id=model_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        identity_kind=(
            "immutable" if _COMMIT_RE.fullmatch(resolved_revision) else "mutable"
        ),
        backend=backend,
        status="collected",
        observations=tuple(
            CanaryObservation(
                row_id=prompt.row_id,
                prompt_sha256=prompt.prompt_sha256,
                normalised_completion=normalise_completion(completion),
            )
            for prompt, completion in zip(prompts, completions, strict=True)
        ),
    )
    validate_evidence(evidence)
    return evidence


def status_evidence(
    *,
    model_id: str,
    requested_revision: str,
    resolved_revision: str,
    backend: str,
    status: t.Literal["not_applicable", "unsupported", "failed"],
    reason: str,
) -> CanaryEvidence:
    """Build a typed non-collected evidence record.

    Returns:
        Validated evidence with no observations.
    """
    evidence = CanaryEvidence(
        model_id=model_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        identity_kind=(
            "immutable" if _COMMIT_RE.fullmatch(resolved_revision) else "mutable"
        ),
        backend=backend,
        status=status,
        reason=reason,
    )
    validate_evidence(evidence)
    return evidence


def validate_evidence(evidence: CanaryEvidence) -> None:
    """Validate evidence strictly enough for persistence and server submission.

    Raises:
        ValueError:
            If any identity, protocol, status, or observation field is invalid.
    """
    if (
        evidence.schema_version != CANARY_EVIDENCE_SCHEMA
        or evidence.protocol_version != CANARY_COLLECTION_PROTOCOL
        or evidence.normaliser_version != CANARY_NORMALISER_VERSION
        or evidence.generation_version != CANARY_GENERATION_VERSION
        or evidence.corpus_id != CANARY_DATASET_ID
        or evidence.corpus_revision != CANARY_DATASET_REVISION
        or evidence.corpus_sha256 != CANARY_CORPUS_SHA256
        or evidence.row_count != CANARY_ROW_COUNT
    ):
        raise ValueError("canary evidence contract does not match the frozen protocol")
    if (
        not evidence.model_id.strip()
        or not evidence.requested_revision.strip()
        or not evidence.resolved_revision.strip()
        or evidence.identity_kind not in {"immutable", "mutable"}
        or not evidence.backend.strip()
        or evidence.status not in _ALLOWED_STATUSES
    ):
        raise ValueError("canary evidence identity or status is invalid")
    if evidence.identity_kind == "immutable" and not _COMMIT_RE.fullmatch(
        evidence.resolved_revision
    ):
        raise ValueError("immutable canary evidence requires a commit revision")
    if evidence.status == "collected":
        if (
            evidence.reason is not None
            or len(evidence.observations) != CANARY_ROW_COUNT
        ):
            raise ValueError("collected evidence must contain 256 observations")
        row_ids: set[str] = set()
        for observation in evidence.observations:
            if (
                not observation.row_id
                or observation.row_id in row_ids
                or not _SHA256_RE.fullmatch(observation.prompt_sha256)
                or len(observation.normalised_completion.encode("utf-8")) > 256
                or "\n" in observation.normalised_completion
            ):
                raise ValueError("canary observation is invalid")
            row_ids.add(observation.row_id)
    elif evidence.reason not in _ALLOWED_REASONS or evidence.observations:
        raise ValueError("non-collected evidence requires a typed reason and no rows")


def evidence_from_dict(value: object) -> CanaryEvidence:
    """Decode the strict evidence contract and reject undeclared fields.

    Returns:
        The validated evidence record.

    Raises:
        ValueError:
            If the object does not conform to the strict evidence contract.
    """
    if not isinstance(value, dict):
        raise ValueError("canary evidence must be an object")
    fields = {
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
    if set(value) != fields or not isinstance(value.get("observations"), list):
        raise ValueError("canary evidence fields are invalid")
    observations: list[CanaryObservation] = []
    for item in value["observations"]:
        if not isinstance(item, dict) or set(item) != {
            "row_id",
            "prompt_sha256",
            "normalised_completion",
        }:
            raise ValueError("canary observation fields are invalid")
        observations.append(
            CanaryObservation(
                row_id=_required_string(item, "row_id"),
                prompt_sha256=_required_string(item, "prompt_sha256"),
                normalised_completion=_string(item, "normalised_completion"),
            )
        )
    identity_kind = value.get("identity_kind")
    status = value.get("status")
    if identity_kind not in {"immutable", "mutable"} or status not in _ALLOWED_STATUSES:
        raise ValueError("canary evidence enum value is invalid")
    reason = value.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("canary evidence reason is invalid")
    row_count = value.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        raise ValueError("canary evidence row count is invalid")
    evidence = CanaryEvidence(
        schema_version=_required_string(value, "schema_version"),
        protocol_version=_required_string(value, "protocol_version"),
        normaliser_version=_required_string(value, "normaliser_version"),
        generation_version=_required_string(value, "generation_version"),
        model_id=_required_string(value, "model_id"),
        requested_revision=_required_string(value, "requested_revision"),
        resolved_revision=_required_string(value, "resolved_revision"),
        identity_kind=t.cast(t.Literal["immutable", "mutable"], identity_kind),
        backend=_required_string(value, "backend"),
        corpus_id=_required_string(value, "corpus_id"),
        corpus_revision=_required_string(value, "corpus_revision"),
        corpus_sha256=_required_string(value, "corpus_sha256"),
        row_count=row_count,
        status=t.cast(
            t.Literal["collected", "not_applicable", "unsupported", "failed"], status
        ),
        reason=reason,
        observations=tuple(observations),
    )
    validate_evidence(evidence)
    return evidence


def _configured_corpus_path() -> Path | None:
    value = os.getenv(CANARY_CORPUS_PATH_ENV)
    return Path(value).expanduser() if value else None


def _required_string(value: dict[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"canary evidence field {key!r} is invalid")
    return item


def _string(value: dict[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise ValueError(f"canary evidence field {key!r} is invalid")
    return item
