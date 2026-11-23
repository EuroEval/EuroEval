"""External inference-server configuration for native volunteer workers."""

import collections.abc as c
import dataclasses
import hashlib
import ipaddress
import json
import os
import typing as t
import urllib.parse
from pathlib import Path

from .types import (
    ExternalExecution,
    GgufArtifact,
    canonical_json,
    gguf_artifact_from_dict,
)

GGUF_MAGIC = b"GGUF"
GENERATIVE_TYPES = ("instruction_tuned", "reasoning")
PREFLIGHT_MAX_TOKENS = 32
PREFLIGHT_TIMEOUT_SECONDS = 30.0
_CHUNK_SIZE = 1024 * 1024
_LOOPBACK_NAMES = {"localhost"}


def _litellm_completion(**kwargs: object) -> object:
    """Call LiteLLM's completion endpoint, imported lazily for hermetic tests.

    Args:
        kwargs:
            LiteLLM completion keyword arguments.

    Returns:
        The LiteLLM model response.
    """
    import litellm  # noqa: PLC0415

    return litellm.completion(**t.cast(t.Any, kwargs))


@dataclasses.dataclass(frozen=True)
class ExternalServerConfig:
    """Frozen request settings and parsed execution for one external server.

    Attributes:
        api_base:
            Validated base URL of the local OpenAI-compatible endpoint.
        served_model:
            Model alias the endpoint serves; sent as the request model.
        gguf_path:
            Local GGUF file the contributor loaded into the server.
        max_context_length:
            Context-length override passed to every benchmark call.
        generative_type:
            Declared generative type; only instruction-tuned and reasoning
            models are supported in the pilot.
        execution:
            The parsed external execution discriminator binding the approved
            GGUF artifact.
        api_key (optional):
            Endpoint API key read from the environment; never persisted,
            logged, or included in the dataclass repr, and excluded from the
            fingerprint.
    """

    api_base: str
    served_model: str
    gguf_path: Path
    max_context_length: int
    generative_type: t.Literal["instruction_tuned", "reasoning"]
    execution: ExternalExecution
    api_key: str | None = dataclasses.field(default=None, repr=False)

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 of endpoint, alias, artifact, context and type.

        Returns:
            The hex digest; contains no secret material.
        """
        return execution_fingerprint(
            api_base=self.api_base,
            served_model=self.served_model,
            artifact=self.execution.artifact,
            max_context_length=self.max_context_length,
            generative_type=self.generative_type,
        )

    def request_kwargs(self) -> dict[str, object]:
        """Benchmark request settings shared by every external call.

        Returns:
            The endpoint, credential, alias-adjacent context and generative
            type settings for one benchmark invocation.
        """
        return {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "max_context_length": self.max_context_length,
            "generative_type": self.generative_type,
        }


def external_server_config(
    *,
    api_base: str,
    served_model: str,
    gguf_file: Path,
    gguf_binding: Path,
    max_context_length: int,
    generative_type: str,
    api_key_env: str | None = None,
) -> ExternalServerConfig:
    """Validate external-mode inputs and build one frozen config.

    Args:
        api_base:
            Base URL of the external inference endpoint.
        served_model:
            Model alias the endpoint serves.
        gguf_file:
            Local GGUF file loaded by the endpoint.
        gguf_binding:
            JSON file with either one six-field artifact descriptor or the
            ``{"version": 1, "artifacts": [...]}`` envelope.
        max_context_length:
            Context-length override for the served model.
        generative_type:
            ``instruction_tuned`` or ``reasoning``; base is not in the pilot.
        api_key_env (optional):
            Name of an environment variable holding the endpoint API key.

    Returns:
        The validated frozen external-server configuration.

    Raises:
        ValueError:
            If any input is malformed, the GGUF file does not exactly match
            one binding, or the named API-key variable is unset.
    """
    base = validate_api_base(api_base=api_base)
    if not isinstance(served_model, str) or not served_model.strip():
        raise ValueError("served model alias must be a non-empty string")
    if any(character.isspace() for character in served_model):
        raise ValueError("served model alias must not contain whitespace")
    if generative_type not in GENERATIVE_TYPES:
        raise ValueError(
            "generative type must be one of "
            + ", ".join(GENERATIVE_TYPES)
            + "; base models are not supported in the pilot"
        )
    if (
        isinstance(max_context_length, bool)
        or not isinstance(max_context_length, int)
        or max_context_length <= 0
    ):
        raise ValueError("max context length must be a positive integer")
    artifact = resolve_gguf_artifact(gguf_file=gguf_file, gguf_binding=gguf_binding)
    api_key = None
    if api_key_env is not None:
        api_key = os.environ.get(api_key_env)
        if api_key is None:
            raise ValueError(f"API key environment variable {api_key_env!r} is not set")
    return ExternalServerConfig(
        api_base=base,
        served_model=served_model,
        gguf_path=gguf_file,
        max_context_length=max_context_length,
        generative_type=generative_type,
        execution=ExternalExecution(kind="external_server", artifact=artifact),
        api_key=api_key,
    )


def execution_fingerprint(
    *,
    api_base: str,
    served_model: str,
    artifact: GgufArtifact,
    max_context_length: int,
    generative_type: str,
) -> str:
    """Compute the stable configuration fingerprint without secrets.

    Args:
        api_base:
            Validated endpoint base URL.
        served_model:
            Served model alias.
        artifact:
            Approved GGUF artifact descriptor.
        max_context_length:
            Context-length override.
        generative_type:
            Declared generative type.

    Returns:
        The hex SHA-256 digest of the canonical fingerprint payload.
    """
    payload = canonical_json(
        {
            "api_base": api_base,
            "served_model": served_model,
            "artifact": dataclasses.asdict(artifact),
            "max_context_length": max_context_length,
            "generative_type": generative_type,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_gguf_artifact(*, gguf_file: Path, gguf_binding: Path) -> GgufArtifact:
    """Select the single approved artifact matching the local GGUF file.

    Args:
        gguf_file:
            Local GGUF file whose magic, filename and streamed SHA-256 are
            verified.
        gguf_binding:
            Binding JSON file with the approved artifact candidates.

    Returns:
        The exactly matching approved artifact.

    Raises:
        ValueError:
            If the file is not GGUF, or filename/hash match zero or several
            candidates.
    """
    digest = streamed_gguf_sha256(gguf_file=gguf_file)
    candidates = load_gguf_binding(gguf_binding=gguf_binding)
    matches = [
        artifact
        for artifact in candidates
        if artifact.filename == gguf_file.name and artifact.sha256 == digest
    ]
    if len(matches) != 1:
        raise ValueError(
            "the GGUF file must match exactly one approved binding by filename "
            "and SHA-256"
        )
    return matches[0]


def load_gguf_binding(*, gguf_binding: Path) -> list[GgufArtifact]:
    """Load approved artifact candidates from a binding JSON file.

    Args:
        gguf_binding:
            Path to a JSON file containing either one six-field artifact
            descriptor or the ``{"version": 1, "artifacts": [...]}`` envelope.

    Returns:
        The strictly decoded candidate artifacts.

    Raises:
        ValueError:
            If the document has neither supported shape or an artifact is
            malformed.
    """
    raw = json.loads(gguf_binding.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and set(raw) == {"version", "artifacts"}:
        artifacts = raw["artifacts"]
        if raw["version"] != 1 or not isinstance(artifacts, list) or not artifacts:
            raise ValueError("gguf binding envelope must be version 1 with artifacts")
        return [gguf_artifact_from_dict(data=item) for item in artifacts]
    return [gguf_artifact_from_dict(data=raw)]


def streamed_gguf_sha256(*, gguf_file: Path) -> str:
    """Check the GGUF magic and hash a file in bounded chunks.

    Args:
        gguf_file:
            Local file to verify.

    Returns:
        The hex SHA-256 digest of the whole file.

    Raises:
        ValueError:
            If the file does not start with the GGUF magic bytes.
    """
    digest = hashlib.sha256()
    with gguf_file.open("rb") as handle:
        magic = handle.read(len(GGUF_MAGIC))
        if magic != GGUF_MAGIC:
            raise ValueError(f"{gguf_file} is not a GGUF file (bad magic bytes)")
        digest.update(magic)
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def validate_api_base(*, api_base: str) -> str:
    """Validate a local endpoint base URL without embedded secrets.

    Userinfo, query and fragment components are rejected. Plain HTTP is only
    allowed for loopback hosts; HTTPS is allowed for any host.

    Args:
        api_base:
            Candidate base URL.

    Returns:
        The validated URL text.

    Raises:
        ValueError:
            If the URL is not HTTP(S), carries userinfo/query/fragment, or uses
            plain HTTP for a non-loopback host.
    """
    if not isinstance(api_base, str) or not api_base.strip():
        raise ValueError("api base must be a non-empty URL string")
    parsed = urllib.parse.urlsplit(api_base)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("api base must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("api base must not embed credentials in the URL")
    if parsed.query or parsed.fragment:
        raise ValueError("api base must not contain a query or fragment")
    host = parsed.hostname
    if not host:
        raise ValueError("api base must contain a host")
    if parsed.scheme == "http" and not _is_loopback(host):
        raise ValueError("plain HTTP api base is only allowed for loopback hosts")
    return api_base


def _is_loopback(host: str) -> bool:
    """Return whether a host name or address is loopback.

    Args:
        host:
            Host name or IP address from a parsed URL.

    Returns:
        Whether the host resolves to a loopback address or is localhost.
    """
    if host.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def preflight_external_server(
    config: ExternalServerConfig, *, completion: c.Callable[..., object] | None = None
) -> str:
    """Run one bounded visible chat completion against the served alias.

    A single attempt is made through the existing LiteLLM/OpenAI-compatible
    path; there is no retry loop. The completion is capped at
    ``PREFLIGHT_MAX_TOKENS`` tokens with a ``PREFLIGHT_TIMEOUT_SECONDS`` timeout.
    The served alias is contributor-declared identity evidence, not remote
    weight attestation.

    Args:
        config:
            The frozen external-server configuration.
        completion (optional):
            Completion callable used for the probe; defaults to LiteLLM.

    Returns:
        The visible completion content.

    Raises:
        ValueError:
            If the endpoint returns no visible content.
    """
    run = completion if completion is not None else _litellm_completion
    response = run(
        model=f"openai/{config.served_model}",
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=PREFLIGHT_MAX_TOKENS,
        timeout=PREFLIGHT_TIMEOUT_SECONDS,
        api_base=config.api_base,
        api_key=config.api_key or "EMPTY",
    )
    content = _visible_content(response=response)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("external server preflight returned no visible completion")
    reported_model = getattr(response, "model", None)
    if reported_model is None and isinstance(response, dict):
        reported_model = response.get("model")
    if reported_model != config.served_model:
        raise ValueError("external server reported a different served model alias")
    return content


def _visible_content(*, response: object) -> str | None:
    """Extract the visible message content from a completion response.

    Args:
        response:
            LiteLLM-style response object or mapping.

    Returns:
        The first choice's message content, or ``None`` when absent.
    """
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return None
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return content if isinstance(content, str) else None


def verify_gguf_artifact(*, gguf_file: Path, artifact: GgufArtifact) -> str:
    """Re-verify the local file against an approved artifact descriptor.

    Args:
        gguf_file:
            Local GGUF file to re-hash.
        artifact:
            The approved artifact the file must match.

    Returns:
        The verified hex SHA-256 digest.

    Raises:
        ValueError:
            If the magic, filename or streamed SHA-256 does not match.
    """
    digest = streamed_gguf_sha256(gguf_file=gguf_file)
    if gguf_file.name != artifact.filename:
        raise ValueError(
            f"GGUF filename {gguf_file.name!r} does not match the approved "
            f"artifact filename {artifact.filename!r}"
        )
    if digest != artifact.sha256:
        raise ValueError(
            "GGUF file SHA-256 does not match the approved artifact digest"
        )
    return digest
