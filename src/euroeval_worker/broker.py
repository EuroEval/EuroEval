"""HTTP client for the v1 volunteer-worker broker protocol."""

import collections.abc as c
import email.utils
import json
import logging
import time
import typing as t
import urllib.error
import urllib.request

from . import __version__
from .types import (
    PROTOCOL_VERSION,
    AuthPoll,
    AuthStart,
    Claim,
    EEERecord,
    HardwareReport,
    Lease,
    auth_poll_from_dict,
    auth_start_from_dict,
    lease_from_dict,
)

logger = logging.getLogger(__name__)
JsonObject = dict[str, object]
Request = c.Callable[[str, str, dict[str, str], JsonObject | None], JsonObject]


class BrokerClient:
    """Small, dependency-free client for the volunteer-worker broker."""

    def __init__(
        self,
        server: str,
        request: Request | None = None,
        worker_version: str = __version__,
    ) -> None:
        """Initialise a broker client.

        Args:
            server:
                Base URL of the worker broker.
            request (optional):
                Injectable request function, primarily for tests.
            worker_version:
                Version advertised in every claim and result envelope.
        """
        self.server = server.rstrip("/")
        self._request = request or _http_request
        self.worker_version = worker_version

    def claim(self, credential: str, hardware: HardwareReport) -> Claim:
        """Claim one available lease, if any.

        Returns:
            A lease, or an empty claim when no work is available.
        """
        payload: JsonObject = {
            "protocol_version": PROTOCOL_VERSION,
            "worker_version": self.worker_version,
            "hardware": _hardware_dict(hardware),
        }
        result = self._post("claim", payload, credential=credential)
        if result.get("status") == "no_work":
            return Claim(lease=None)
        return Claim(lease=lease_from_dict(result))

    def _post(
        self, path: str, payload: JsonObject, credential: str | None = None
    ) -> JsonObject:
        headers = {"Content-Type": "application/json"}
        if credential is not None:
            headers["Authorization"] = f"Bearer {credential}"
        try:
            result = self._request("POST", f"{self.server}/{path}", headers, payload)
        except BrokerError:
            raise
        except Exception as error:
            raise BrokerError(f"broker request {path!r} failed: {error}") from error
        if result.get("protocol_version") != PROTOCOL_VERSION:
            raise BrokerError("broker response has an unsupported protocol_version")
        return result

    def finalise(self, credential: str, lease_id: str) -> str:
        """Mark all records for a lease as accepted and return its identifier.

        Returns:
            The broker submission identifier.

        Raises:
            BrokerError:
                If the response omits the submission identifier.
        """
        result = self._post(
            "finalise",
            {"protocol_version": PROTOCOL_VERSION, "lease_id": lease_id},
            credential=credential,
        )
        submission_id = result.get("submission_id")
        if not isinstance(submission_id, str) or not submission_id:
            raise BrokerError("broker finalise response omitted submission_id")
        return submission_id

    def heartbeat(self, credential: str, lease_id: str) -> str:
        """Renew a lease and return the broker's renewed expiry timestamp.

        Returns:
            The broker-issued ISO-8601 expiry timestamp.

        Raises:
            BrokerError:
                If the response is invalid or the lease cannot be renewed.
        """
        result = self._post(
            "heartbeat",
            {"protocol_version": PROTOCOL_VERSION, "lease_id": lease_id},
            credential=credential,
        )
        if result.get("lease_id") != lease_id:
            raise BrokerError("broker heartbeat response changed lease identity")
        expires_at = result.get("expires_at")
        if not isinstance(expires_at, str) or not expires_at:
            raise BrokerError("broker heartbeat response omitted expires_at")
        return expires_at

    def poll_auth(self, session_id: str) -> AuthPoll:
        """Poll a device authorisation flow.

        Returns:
            The current device-flow state.
        """
        return auth_poll_from_dict(
            self._post(
                "auth/poll",
                {"protocol_version": PROTOCOL_VERSION, "session_id": session_id},
            )
        )

    def release(self, credential: str, lease_id: str, reason: str) -> None:
        """Release a lease without losing the broker's retry state."""
        self._post(
            "release",
            {
                "protocol_version": PROTOCOL_VERSION,
                "lease_id": lease_id,
                "reason": reason,
            },
            credential=credential,
        )

    def start_auth(self) -> AuthStart:
        """Start a device authorisation flow.

        Returns:
            Device-flow details.
        """
        return auth_start_from_dict(
            self._post("auth/start", {"protocol_version": PROTOCOL_VERSION})
        )

    def submit_result(self, credential: str, lease: Lease, result: EEERecord) -> None:
        """Submit one exact result record; the broker makes this idempotent."""
        payload: JsonObject = {
            "protocol_version": PROTOCOL_VERSION,
            "lease_id": lease.lease_id,
            "issue_number": lease.issue_number,
            "model_id": lease.model_id,
            "model_revision": lease.model_revision,
            "language": lease.language,
            "euroeval_version": lease.euroeval_version,
            "image_digest": lease.image_digest,
            "worker_version": lease.worker_version,
            "record_json": result.record_json,
            "digest": result.digest,
        }
        self._post("result", payload, credential=credential)


class BrokerError(RuntimeError):
    """Raised when the broker rejects or cannot answer a request."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
        body: JsonObject | None = None,
        code: str | None = None,
    ) -> None:
        """Initialise an error with safe structured broker metadata."""
        safe_body = _redact(body) if body is not None else None
        broker_message = safe_body.get("error") if safe_body else None
        if isinstance(broker_message, str) and broker_message:
            message = broker_message
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.body = safe_body
        self.code = code or _string_value(safe_body, "code")
        self.message = message


def _hardware_dict(hardware: HardwareReport) -> JsonObject:
    return {
        "architecture": hardware.architecture,
        "ram_bytes": hardware.ram_bytes,
        "free_disk_bytes": hardware.free_disk_bytes,
        "driver_version": hardware.driver_version,
        "cuda_version": hardware.cuda_version,
        "pytorch_version": hardware.pytorch_version,
        "gpu_memory_utilisation": hardware.gpu_memory_utilisation,
        "selected_gpu_index": hardware.selected_gpu_index,
        "selected_gpu_uuid": hardware.selected_gpu_uuid,
        "gpus": [
            {
                "index": gpu.index,
                "name": gpu.name,
                "uuid": gpu.uuid,
                "free_memory_bytes": gpu.free_memory_bytes,
                "total_memory_bytes": gpu.total_memory_bytes,
                "compute_capability": gpu.compute_capability,
            }
            for gpu in hardware.gpus
        ],
    }


class BrokerProtocol(t.Protocol):
    """Operations required by the worker runtime."""

    def claim(self, credential: str, hardware: HardwareReport) -> Claim:
        """Claim available work."""
        ...

    def finalise(self, credential: str, lease_id: str) -> str:
        """Finalise a lease and return its submission identifier."""
        ...

    def heartbeat(self, credential: str, lease_id: str) -> str:
        """Renew a lease and return its new expiry timestamp."""
        ...

    def poll_auth(self, session_id: str) -> AuthPoll:
        """Poll a device flow."""
        ...

    def release(self, credential: str, lease_id: str, reason: str) -> None:
        """Release a lease."""
        ...

    def start_auth(self) -> AuthStart:
        """Start a device flow."""
        ...

    def submit_result(self, credential: str, lease: Lease, result: EEERecord) -> None:
        """Submit one exact result JSON string."""
        ...


def _http_request(
    method: str, url: str, headers: dict[str, str], payload: JsonObject | None
) -> JsonObject:
    body = (
        None
        if payload is None
        else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    )
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            decoded = json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = _error_body(error)
        raise BrokerError(
            f"broker returned HTTP {error.code}",
            status=error.code,
            retry_after=_retry_after(error.headers.get("Retry-After")),
            body=body,
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise BrokerError(f"invalid broker response: {error}") from error
    if not isinstance(decoded, dict):
        raise BrokerError("broker response was not a JSON object")
    return decoded


def _error_body(error: urllib.error.HTTPError) -> JsonObject | None:
    """Decode an error response without exposing request headers or credentials.

    Returns:
        The JSON object returned by the broker, if it was one.
    """
    try:
        value = json.loads(error.read())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return _redact(value) if isinstance(value, dict) else None


def _string_value(value: JsonObject | None, key: str) -> str | None:
    """Return a non-empty string field from a broker response."""
    item = value.get(key) if value is not None else None
    return item if isinstance(item, str) and item else None


def _redact(value: object) -> JsonObject:
    """Copy a JSON object while removing credential-bearing response fields.

    Returns:
        A response object without sensitive fields.
    """
    if not isinstance(value, dict):
        return {}
    sensitive = {
        "access_token",
        "authorization",
        "credential",
        "refresh_token",
        "secret",
        "token",
    }
    result: JsonObject = {}
    for key, item in value.items():
        if not isinstance(key, str) or key.lower() in sensitive:
            continue
        if isinstance(item, dict):
            result[key] = _redact(item)
        elif isinstance(item, list):
            result[key] = [
                _redact(entry) if isinstance(entry, dict) else entry for entry in item
            ]
        else:
            result[key] = item
    return result


def _retry_after(value: str | None) -> float | None:
    """Parse seconds or an HTTP date from a Retry-After header.

    Returns:
        The delay in seconds, or ``None`` for an invalid header.
    """
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return max(0.0, parsed.timestamp() - time.time()) if parsed else None
