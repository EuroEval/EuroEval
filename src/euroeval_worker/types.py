"""Typed protocol objects shared by the worker and its broker."""

import dataclasses
import hashlib
import json
import typing as t

PROTOCOL_VERSION = "volunteer-worker/v1"

JsonValue: t.TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: t.TypeAlias = dict[str, JsonValue]


@dataclasses.dataclass(frozen=True)
class AuthStart:
    """Device-flow details returned by the broker."""

    session_id: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


@dataclasses.dataclass(frozen=True)
class AuthPoll:
    """Result of one device-flow poll."""

    pending: bool
    credential: str | None = None
    github_login: str | None = None
    retry_after: int | None = None


@dataclasses.dataclass(frozen=True)
class HardwareReport:
    """Hardware and software facts sent when claiming work."""

    architecture: str
    ram_bytes: int
    free_disk_bytes: int
    driver_version: str | None
    cuda_version: str | None
    pytorch_version: str | None
    gpus: tuple["Gpu", ...]
    gpu_memory_utilisation: float = 0.8
    selected_gpu_index: int | None = None
    selected_gpu_uuid: str | None = None


@dataclasses.dataclass(frozen=True)
class Gpu:
    """A single NVIDIA GPU discovered through nvidia-smi."""

    name: str
    uuid: str
    free_memory_bytes: int
    total_memory_bytes: int
    compute_capability: str | None
    index: int = 0


@dataclasses.dataclass(frozen=True)
class Lease:
    """One broker-issued evaluation lease."""

    lease_id: str
    issue_number: int
    model_id: str
    model_revision: str
    language: str
    euroeval_version: str
    image_digest: str
    expires_at: str
    worker_version: str = "legacy-worker"
    gpu_memory_utilisation: float = 0.8
    model_profile: str | None = None
    selected_gpu_uuid: str | None = None
    selected_gpu_index: int | None = None


@dataclasses.dataclass(frozen=True, init=False)
class EEERecord:
    """An exact EEE JSON object and its deterministic digest.

    ``record`` and ``sha256`` keyword arguments are accepted for compatibility
    with older evaluator adapters. New code should use ``record_json`` and
    ``digest`` so that the original bytes cannot be accidentally re-encoded.
    """

    record_json: str
    digest: str

    def __init__(
        self,
        record_json: str | JsonObject | None = None,
        digest: str | None = None,
        *,
        record: JsonObject | None = None,
        sha256: str | None = None,
    ) -> None:
        """Create a record, canonicalising dictionaries exactly once.

        Raises:
            TypeError:
                If neither a JSON string nor an object is supplied.
        """
        if record_json is None:
            record_json = record
        if isinstance(record_json, dict):
            record_json = canonical_json(record_json)
        if not isinstance(record_json, str):
            raise TypeError("record_json must be a JSON string or object")
        _validate_json_text(record_json)
        actual_digest = digest if digest is not None else sha256
        if actual_digest is None:
            actual_digest = hashlib.sha256(record_json.encode("utf-8")).hexdigest()
        object.__setattr__(self, "record_json", record_json)
        object.__setattr__(self, "digest", actual_digest)

    @property
    def sha256(self) -> str:
        """Expose the digest under the legacy attribute name."""
        return self.digest

    @property
    def record(self) -> JsonObject:
        """Decode the record for compatibility with local evaluator callers.

        Raises:
            ValueError:
                If the stored JSON does not contain an object.
        """
        value = json.loads(self.record_json)
        if not isinstance(value, dict):
            raise ValueError("EEE record JSON must contain an object")
        return t.cast(JsonObject, value)


@dataclasses.dataclass(frozen=True)
class Claim:
    """Broker claim response."""

    lease: Lease | None


def canonical_json(value: JsonObject) -> str:
    """Serialise an EEE object deterministically and reject non-finite values.

    Returns:
        The canonical JSON text.

    """
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_json_text(value: str) -> None:
    try:
        parsed = json.loads(value, parse_constant=_reject_constant)
    except (ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "record_json must be valid JSON without non-finite values"
        ) from error
    if not isinstance(parsed, dict):
        raise ValueError("record_json must contain a JSON object")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _protocol(data: dict[str, object]) -> None:
    """Reject responses from a different broker protocol.

    Raises:
        ValueError:
            If the response protocol is unsupported.
    """
    if data.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("broker response has an unsupported protocol_version")


def _string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"broker response field {key!r} must be a non-empty string")
    return value


def auth_start_from_dict(data: dict[str, object]) -> AuthStart:
    """Decode an auth/start response.

    Returns:
        The typed device-flow response.
    """
    _protocol(data)
    return AuthStart(
        session_id=_string(data, "session_id"),
        user_code=_string(data, "user_code"),
        verification_uri=_string(data, "verification_uri"),
        expires_in=_integer(data, "expires_in", 0),
        interval=_integer(data, "interval", 5),
    )


def auth_poll_from_dict(data: dict[str, object]) -> AuthPoll:
    """Decode an auth/poll response.

    Returns:
        The typed poll response.
    """
    _protocol(data)
    pending = data.get("status") == "pending" or bool(data.get("pending", False))
    return AuthPoll(
        pending=pending,
        credential=_optional_string(data, "credential"),
        github_login=_optional_string(data, "github_login"),
        retry_after=_optional_positive_integer(data, "retry_after"),
    )


def lease_from_dict(data: dict[str, object]) -> Lease:
    """Decode a lease response.

    Returns:
        The typed lease.

    Raises:
        ValueError:
            If the broker response is malformed or has an invalid fit value.
    """
    _protocol(data)
    selected_gpu_uuid = data.get("selected_gpu_uuid")
    if selected_gpu_uuid is not None and (
        not isinstance(selected_gpu_uuid, str) or not selected_gpu_uuid
    ):
        raise ValueError("broker response field 'selected_gpu_uuid' must be a string")
    gpu_memory_utilisation = _number(data, "gpu_memory_utilisation", 0.8)
    if not 0 < gpu_memory_utilisation <= 1:
        raise ValueError(
            "broker response gpu_memory_utilisation must be between 0 and 1"
        )
    return Lease(
        lease_id=_string(data, "lease_id"),
        issue_number=_integer(data, "issue_number"),
        model_id=_string(data, "model_id"),
        model_revision=_string(data, "model_revision"),
        language=_string(data, "language"),
        euroeval_version=_string(data, "euroeval_version"),
        image_digest=_string(data, "image_digest"),
        worker_version=_string(data, "worker_version"),
        gpu_memory_utilisation=gpu_memory_utilisation,
        expires_at=_string(data, "expires_at"),
        model_profile=_optional_string(data, "model_profile"),
        selected_gpu_uuid=selected_gpu_uuid,
        selected_gpu_index=_optional_integer(data, "selected_gpu_index"),
    )


def _integer(data: dict[str, object], key: str, default: int | None = None) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"broker response field {key!r} must be an integer")
    return int(value)


def _number(data: dict[str, object], key: str, default: float | None = None) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"broker response field {key!r} must be a number")
    return float(value)


def _optional_string(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) else None


def _optional_integer(data: dict[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"broker response field {key!r} must be an integer")
    return int(value)


def _optional_positive_integer(data: dict[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    parsed = _integer(data, key)
    return max(1, parsed)
