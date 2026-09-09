"""Typed protocol objects shared by the worker and its broker."""

import dataclasses
import typing as t

PROTOCOL_VERSION = "volunteer-worker/v1"

JsonValue: t.TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


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


@dataclasses.dataclass(frozen=True)
class Gpu:
    """A single NVIDIA GPU discovered through nvidia-smi."""

    name: str
    uuid: str
    free_memory_bytes: int
    total_memory_bytes: int
    compute_capability: str | None


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


@dataclasses.dataclass(frozen=True)
class EEERecord:
    """An EEE JSON object and its deterministic digest."""

    record: dict[str, JsonValue]
    sha256: str


@dataclasses.dataclass(frozen=True)
class Claim:
    """Broker claim response."""

    lease: Lease | None


def _protocol(data: dict[str, object]) -> None:
    """Reject responses from a different broker protocol.

    Raises:
        ValueError: If the response protocol is unsupported.
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
    """
    _protocol(data)
    return Lease(
        lease_id=_string(data, "lease_id"),
        issue_number=_integer(data, "issue_number"),
        model_id=_string(data, "model_id"),
        model_revision=_string(data, "model_revision"),
        language=_string(data, "language"),
        euroeval_version=_string(data, "euroeval_version"),
        image_digest=_string(data, "image_digest"),
        expires_at=_string(data, "expires_at"),
    )


def _integer(data: dict[str, object], key: str, default: int | None = None) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"broker response field {key!r} must be an integer")
    return int(value)


def _optional_string(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) else None


def _optional_positive_integer(data: dict[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    parsed = _integer(data, key)
    return max(1, parsed)
