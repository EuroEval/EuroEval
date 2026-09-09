"""Private worker state and retry-safe result storage."""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class StateStore:
    """Persist only the broker credential, login, and pending EEE records."""

    def __init__(self, directory: Path) -> None:
        """Initialise state beneath ``directory``."""
        self.directory = directory
        self.path = directory / "state.json"
        self.results_path = directory / "pending-results.jsonl"

    def load_auth(self) -> tuple[str, str] | None:
        """Return the saved opaque credential and verified login.

        Raises:
            RuntimeError:
                If the state file is malformed.
        """
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            credential = data["credential"]
            login = data["github_login"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid worker state: {self.path}") from error
        if not isinstance(credential, str) or not isinstance(login, str):
            raise RuntimeError(f"invalid worker state: {self.path}")
        return credential, login

    def clear_auth(self) -> None:
        """Forget an expired broker credential without touching results."""
        self.path.unlink(missing_ok=True)

    def save_auth(self, credential: str, github_login: str) -> None:
        """Save device-flow credentials with restrictive permissions."""
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"credential": credential, "github_login": github_login}) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)
        os.chmod(self.path, 0o600)

    def append_result(self, line: str) -> None:
        """Keep an isolated result available when submission is interrupted."""
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        os.chmod(self.results_path, 0o600)

    def clear_results(self) -> None:
        """Remove pending results after broker finalisation."""
        self.results_path.unlink(missing_ok=True)


def default_state_dir() -> Path:
    """Select a writable container cache, then the user's cache directory.

    Returns:
        A writable state directory.

    Raises:
        RuntimeError:
            If no candidate directory is writable.
    """
    configured = os.environ.get("EUROEVAL_WORKER_CACHE")
    candidates = [Path(configured)] if configured else []
    candidates.extend([Path("/cache"), Path.home() / ".cache" / "euroeval-worker"])
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write-test"
            probe.touch()
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise RuntimeError("no writable worker state directory; use --state-dir")
