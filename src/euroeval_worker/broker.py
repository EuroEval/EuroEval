"""HTTP client for the v1 volunteer-worker broker protocol."""

import collections.abc as c
import json
import logging
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


class BrokerProtocol(t.Protocol):
    """Operations required by the worker runtime."""

    def start_auth(self) -> AuthStart:
        """Start a device flow."""
        ...

    def poll_auth(self, session_id: str) -> AuthPoll:
        """Poll a device flow."""
        ...

    def claim(self, credential: str, hardware: HardwareReport) -> Claim:
        """Claim available work."""
        ...

    def heartbeat(self, credential: str, lease_id: str) -> None:
        """Renew a lease."""
        ...

    def submit_result(self, credential: str, lease: Lease, result: EEERecord) -> None:
        """Submit one EEE record."""
        ...

    def finalise(self, credential: str, lease_id: str) -> None:
        """Finalise a lease."""
        ...

    def release(self, credential: str, lease_id: str, reason: str) -> None:
        """Release a lease."""
        ...


class BrokerError(RuntimeError):
    """Raised when the broker rejects or cannot answer a request."""


class BrokerClient:
    """Small, dependency-free client for the worker broker."""

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
                Version advertised in every claim.
        """
        self.server = server.rstrip("/")
        self._request = request or _http_request
        self.worker_version = worker_version

    def start_auth(self) -> AuthStart:
        """Start a device authorisation flow.

        Returns:
            Device-flow details.
        """
        return auth_start_from_dict(
            self._post("auth/start", {"protocol_version": PROTOCOL_VERSION})
        )

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

    def claim(self, credential: str, hardware: HardwareReport) -> Claim:
        """Claim one available lease, if any.

        Returns:
            A lease, or an empty claim when no work is available.

        Raises:
            BrokerError: If the broker rejects the claim.
        """
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "worker_version": self.worker_version,
            "hardware": _hardware_dict(hardware),
        }
        result = self._post("claim", dict(payload), credential=credential)
        if result.get("protocol_version") != PROTOCOL_VERSION:
            raise BrokerError("broker response has an unsupported protocol_version")
        if result.get("status") == "no_work":
            return Claim(lease=None)
        return Claim(lease=lease_from_dict(result))

    def heartbeat(self, credential: str, lease_id: str) -> None:
        """Renew a lease."""
        self._post(
            "heartbeat",
            {"protocol_version": PROTOCOL_VERSION, "lease_id": lease_id},
            credential=credential,
        )

    def submit_result(self, credential: str, lease: Lease, result: EEERecord) -> None:
        """Submit one result record; the broker makes this idempotent."""
        payload: JsonObject = {
            "protocol_version": PROTOCOL_VERSION,
            "lease_id": lease.lease_id,
            "issue_number": lease.issue_number,
            "model_id": lease.model_id,
            "model_revision": lease.model_revision,
            "language": lease.language,
            "euroeval_version": lease.euroeval_version,
            "image_digest": lease.image_digest,
            "record": result.record,
            "digest": result.sha256,
        }
        self._post("result", payload, credential=credential)

    def finalise(self, credential: str, lease_id: str) -> None:
        """Mark all records for a lease as accepted."""
        self._post(
            "finalise",
            {"protocol_version": PROTOCOL_VERSION, "lease_id": lease_id},
            credential=credential,
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

    def _post(
        self, path: str, payload: JsonObject, credential: str | None = None
    ) -> JsonObject:
        headers = {"Content-Type": "application/json"}
        if credential is not None:
            headers["Authorization"] = f"Bearer {credential}"
        try:
            return self._request("POST", f"{self.server}/{path}", headers, payload)
        except BrokerError:
            raise
        except Exception as error:
            raise BrokerError(f"broker request {path!r} failed: {error}") from error


def _http_request(
    method: str, url: str, headers: dict[str, str], payload: JsonObject | None
) -> JsonObject:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            decoded = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise BrokerError(
            f"broker returned HTTP {error.code}: {detail[:200]}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise BrokerError(f"invalid broker response: {error}") from error
    if not isinstance(decoded, dict):
        raise BrokerError("broker response was not a JSON object")
    return decoded


def _hardware_dict(hardware: HardwareReport) -> JsonObject:
    return {
        "architecture": hardware.architecture,
        "ram_bytes": hardware.ram_bytes,
        "free_disk_bytes": hardware.free_disk_bytes,
        "driver_version": hardware.driver_version,
        "cuda_version": hardware.cuda_version,
        "pytorch_version": hardware.pytorch_version,
        "gpus": [
            {
                "name": gpu.name,
                "uuid": gpu.uuid,
                "free_memory_bytes": gpu.free_memory_bytes,
                "total_memory_bytes": gpu.total_memory_bytes,
                "compute_capability": gpu.compute_capability,
            }
            for gpu in hardware.gpus
        ],
    }
