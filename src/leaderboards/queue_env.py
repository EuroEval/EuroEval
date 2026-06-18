"""Environment, credential, VM-id, and single-instance lock helpers.

The queue processor wakes up with no ambient credentials and needs to:

* read a ``.env`` file into ``os.environ`` for keys that aren't already set,
* interactively prompt for missing tokens when stdin is a TTY,
* resolve a stable per-VM identifier used to reclaim orphaned issues,
* resolve the GitHub login that owns the configured PAT, and
* take a process-wide lock so two queue processors don't race.

These responsibilities are shared by anyone running the queue and have no
business living inside the main orchestrator script.
"""

from __future__ import annotations

import fcntl
import getpass
import logging
import os
import socket
import sys
import urllib.error
import uuid
from pathlib import Path

from .github_api import gh_request

logger = logging.getLogger(__name__)


def load_dotenv_into_environ(env_path: Path) -> None:
    """Populate ``os.environ`` from ``env_path`` for keys not already set.

    Existing env vars win, so an explicit ``GITHUB_TOKEN=... python ...``
    override is honoured. Malformed lines are ignored.

    Args:
        env_path:
            The dotenv file to read.
    """
    if not env_path.is_file():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and value and key not in os.environ:
                os.environ[key] = value
    except OSError as e:
        logger.warning(f"Could not read {env_path}: {e}")


def prompt_and_persist_env_var(
    env_path: Path, name: str, prompt_text: str, secret: bool = False
) -> str:
    """Interactively prompt for an env var value and persist it to ``env_path``.

    Args:
        env_path:
            The dotenv file to update.
        name:
            The env var name to set and persist.
        prompt_text:
            Text shown to the user before the input prompt.
        secret (optional):
            When True, read the value via ``getpass`` so the input is not
            echoed. Defaults to False.

    Returns:
        The non-empty value entered by the user.
    """
    if not sys.stdin.isatty():
        logger.error(
            f"{name} env var is required but stdin is not a TTY; cannot prompt. "
            "Set it in the environment or in the .env file and re-run."
        )
        sys.exit(1)
    if secret:

        def reader() -> str:
            """Read the value without echoing it to the terminal.

            Returns:
                The value entered by the user.
            """
            return getpass.getpass(f"{prompt_text}: ")
    else:

        def reader() -> str:
            """Read the value from standard input.

            Returns:
                The value entered by the user.
            """
            return input(f"{prompt_text}: ").strip()

    value = ""
    while not value:
        try:
            value = reader().strip()
        except (EOFError, KeyboardInterrupt):
            logger.error(f"Aborted while reading {name}.")
            sys.exit(1)
        if not value:
            print("Value cannot be empty; please try again.", file=sys.stderr)
    persist_env_var(env_path=env_path, name=name, value=value)
    return value


def persist_env_var(env_path: Path, name: str, value: str) -> None:
    """Write ``NAME=VALUE`` to ``env_path``, replacing any prior entry.

    Args:
        env_path:
            The dotenv file to update.
        name:
            The env var name to persist.
        value:
            The value to associate with ``name``.
    """
    try:
        existing = ""
        if env_path.is_file():
            existing = env_path.read_text(encoding="utf-8")
        lines = existing.splitlines()
        replaced = False
        for i, raw_line in enumerate(lines):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, _ = stripped.partition("=")
            if key.strip() == name:
                lines[i] = f"{name}={value}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{name}={value}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"Saved {name} to {env_path}.")
    except OSError as e:
        logger.warning(f"Could not persist {name} to {env_path}: {e}")


def resolve_vm_id(env_path: Path) -> str:
    """Return a stable VM id, reading from or writing to ``env_path``.

    Resolution order:

    1. ``EUROEVAL_VM_ID=...`` line in ``env_path``.
    2. Newly generated ``<hostname>-<8-char-uuid>-<pid>``, appended to
       ``env_path`` so subsequent runs reuse the same id.

    The PID suffix ensures uniqueness even when multiple queue processors
    run on the same machine (e.g., during development or accidental dual-launch).

    Failures to persist a freshly generated id are logged and the in-memory
    id is still returned.

    Args:
        env_path:
            The dotenv file to read from and append to.

    Returns:
        The resolved VM id.
    """
    if env_path.is_file():
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip() != "EUROEVAL_VM_ID":
                    continue
                value = value.strip().strip("'\"")
                if value:
                    return value
        except OSError as e:
            logger.warning(f"Could not read {env_path}: {e}")

    generated = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}-{os.getpid()}"
    try:
        existing = ""
        if env_path.is_file():
            existing = env_path.read_text(encoding="utf-8")
        separator = "" if not existing or existing.endswith("\n") else "\n"
        with env_path.open("a", encoding="utf-8") as f:
            f.write(f"{separator}EUROEVAL_VM_ID={generated}\n")
        logger.info(f"Generated new vm-id {generated!r}, saved to {env_path}.")
    except OSError as e:
        logger.warning(
            f"Could not persist vm-id to {env_path}: {e}. "
            f"Using ephemeral vm-id {generated!r}."
        )
    return generated


def resolve_assignee_from_token() -> str:
    """Return the GitHub login of the ``GITHUB_TOKEN`` owner, or exit on failure."""
    try:
        user = gh_request(path="/user")
    except urllib.error.HTTPError as e:
        logger.error(f"Could not resolve GITHUB_TOKEN owner via /user: {e}")
        sys.exit(1)
    if not isinstance(user, dict) or not user.get("login"):
        logger.error("GITHUB_TOKEN /user response did not include a login.")
        sys.exit(1)
    return str(user["login"])


def acquire_single_instance_lock(lock_path: Path) -> int:
    """Acquire an exclusive flock on ``lock_path`` or exit with an error.

    The returned file descriptor must be kept alive for the lock to hold;
    the kernel releases the lock when the process exits, so no stale lock
    file is left behind.

    Args:
        lock_path:
            The on-disk lock file path.

    Returns:
        The open file descriptor holding the lock.
    """
    try:
        # Open (or create) the lock file as read/write so we can both lock it
        # and update its contents with the current process PID.
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as e:
        logger.error(f"Failed to open lock file {lock_path}: {e}")
        sys.exit(1)
    try:
        # Request a non-blocking exclusive lock: fail immediately instead of
        # waiting if another process already holds the queue lock.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        existing_pid = "unknown"
        try:
            existing_pid = lock_path.read_text(encoding="utf-8").strip() or "unknown"
        except OSError:
            pass
        os.close(fd)
        logger.error(
            f"Another process_evaluation_queue.py is already running "
            f"(pid {existing_pid}); aborting. Set EUROEVAL_QUEUE_LOCK to a "
            f"different path if you need to run a second instance."
        )
        sys.exit(1)

    # Truncate the lock file first so stale bytes from a longer old PID are
    # removed before writing the current PID.
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())

    # Flush file contents to disk so other processes can reliably read the PID
    # associated with the lock holder.
    os.fsync(fd)

    return fd
