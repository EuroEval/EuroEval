"""Device-flow authentication without GitHub or Hugging Face tokens."""

import collections.abc as c
import logging
import time
import typing as t

from .broker import BrokerError
from .state import StateStore
from .types import AuthPoll, AuthStart

logger = logging.getLogger(__name__)


class AuthProtocol(t.Protocol):
    """Operations required for device-flow authentication."""

    def start_auth(self) -> AuthStart:
        """Start a device flow."""
        raise NotImplementedError

    def poll_auth(self, session_id: str) -> AuthPoll:
        """Poll a device flow."""
        raise NotImplementedError


def authenticate(
    client: "AuthProtocol",
    state: StateStore,
    sleep: c.Callable[[float], None] = time.sleep,
) -> tuple[str, str]:
    """Load saved broker auth or complete and persist a device flow.

    Returns:
        The opaque credential and verified GitHub login.

    Raises:
        TimeoutError:
            If the user does not approve the device flow in time.
        BrokerError:
            If the broker returns an authentication error without retry metadata.
    """
    saved = state.load_auth()
    if saved is not None:
        return saved
    start = client.start_auth()
    logger.info("Open %s and enter code %s", start.verification_uri, start.user_code)
    deadline = time.monotonic() + start.expires_in
    while time.monotonic() < deadline:
        try:
            result = client.poll_auth(session_id=start.session_id)
        except BrokerError as error:
            if error.retry_after is None:
                raise
            sleep(min(error.retry_after, max(0.0, deadline - time.monotonic())))
            continue
        if not result.pending and result.credential and result.github_login:
            state.save_auth(
                credential=result.credential, github_login=result.github_login
            )
            logger.info("Authenticated broker account %s", result.github_login)
            return result.credential, result.github_login
        sleep(
            min(
                result.retry_after or start.interval,
                max(0.0, deadline - time.monotonic()),
            )
        )
    raise TimeoutError("device authorisation expired before it was approved")
