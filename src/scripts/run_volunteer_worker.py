"""Host-only, standard-library launcher for immutable EuroEval worker images."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from types import FrameType
from typing import Any, TypeVar

T = TypeVar("T")

LOGGER = logging.getLogger(__name__)
REPOSITORY = "ghcr.io/euroeval/euroeval-worker"
BROKER = "https://euroeval.com/api/worker/release-info?architecture="
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def main(
    argv: Sequence[str] | None = None,
    *,
    command: Callable[..., Any] = subprocess.run,
    http_get: Callable[..., Any] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Run the host lifecycle, returning a worker exit status or 1 on refusal.

    Args:
        argv: CLI arguments (defaults to the process arguments).
        command: subprocess.run-compatible Docker command seam.
        http_get: urlopen-compatible public release discovery seam.
        sleep: Backoff seam.

    Returns:
        Zero for a completed --once handoff/idle, 130 for cancellation, or failure.

    Raises:
        ValueError: If the operation cannot proceed safely.
    """
    args = _parser().parse_args(argv)
    try:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", args.volume):
            raise ValueError("--volume must be a named Docker volume")
        if not 0 < args.gpu_memory_utilisation <= 1:
            raise ValueError("GPU memory utilisation must be in (0, 1]")
        state_path = PurePosixPath(args.state_dir)
        if (
            not state_path.is_absolute()
            or ".." in state_path.parts
            or (
                state_path != PurePosixPath("/cache")
                and PurePosixPath("/cache") not in state_path.parents
            )
        ):
            raise ValueError("--state-dir must be /cache or a directory below it")
        if args.image:
            _digest(args.image)
        docker = ["docker"] + (["--context", args.context] if args.context else [])
        context = command(
            docker + ["context", "show"], check=True, capture_output=True, text=True
        ).stdout.strip()
        host_arch = command(
            docker + ["info", "--format", "{{.Architecture}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        arch = {
            "amd64": "amd64",
            "x86_64": "amd64",
            "arm64": "arm64",
            "aarch64": "arm64",
        }.get(host_arch)
        if arch is None or (args.architecture and args.architecture != arch):
            raise ValueError("Unsupported or mismatched Docker host architecture")
        identity = hashlib.sha256(f"{context}\n{args.volume}".encode()).hexdigest()
        directory = Path(args.launcher_state_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (directory / f"{identity}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Another launcher owns this context/cache") from error
            return _Lifecycle(
                args,
                docker,
                arch,
                directory / f"{identity}.json",
                command,
                http_get,
                sleep,
            ).run()
    except KeyboardInterrupt:
        return 130
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        LOGGER.error("Launcher stopped: %s", error)
        return 1


class _Lifecycle:
    """Keep cache provenance and container ownership local to one locked lifecycle."""

    def __init__(
        self,
        args: argparse.Namespace,
        docker: list[str],
        arch: str,
        journal: Path,
        command: Callable[..., Any],
        http_get: Callable[..., Any],
        sleep: Callable[[float], None],
    ) -> None:
        """Store the host configuration and injected seams.

        Args:
            args: Parsed CLI configuration.
            docker: Docker command prefix including context selection.
            arch: Validated Docker host architecture.
            journal: Nonsecret host journal path.
            command: Docker execution seam.
            http_get: Public discovery seam.
            sleep: Backoff seam.
        """
        self.args = args
        self.docker = docker
        self.arch = arch
        self.journal = journal
        self.command = command
        self.http_get = http_get
        self.sleep = sleep
        self.cancelled = False
        self.owned: str | None = None

    def interrupt(self, signum: int, frame: FrameType | None) -> None:
        """Forward cancellation only to this launcher's container, then wait.

        Args:
            signum: Host signal number.
            frame: Python signal frame (unused).
        """
        self.cancelled = True
        if self.owned:
            self.command(
                self.docker
                + ["kill", "--signal", signal.Signals(signum).name, self.owned],
                check=False,
                capture_output=True,
                text=True,
            )

    def release(self) -> str:
        """Fetch and strictly validate the public release record.

        Returns:
            Official immutable image reference.

        Raises:
            ValueError: If the operation cannot proceed safely.
        """
        with self.http_get(BROKER + self.arch, timeout=20) as response:
            if response.geturl() != BROKER + self.arch:
                raise ValueError(
                    "Release discovery redirected away from the official broker"
                )
            data = json.loads(response.read())
        if not isinstance(data, dict):
            raise ValueError("Invalid release metadata")
        expected = {
            "protocol_version": "volunteer-worker/v1",
            "launcher_protocol_version": "volunteer-launcher/v1",
            "architecture": self.arch,
            "image_repository": REPOSITORY,
        }
        if any(data.get(key) != value for key, value in expected.items()):
            raise ValueError("Untrusted release protocol, architecture or repository")
        image = data.get("image_reference")
        digest = _digest(image)
        if data.get("image_digest") != digest or any(
            not isinstance(data.get(key), str) or not data[key].strip()
            for key in ("worker_version", "euroeval_version")
        ):
            raise ValueError("Incomplete release metadata")
        return image

    def run(self) -> int:
        """Orchestrate discovery, provenance inspection, resume and safe handoffs.

        Returns:
            Final CLI status.

        Raises:
            ValueError: If the operation cannot proceed safely.
        """
        self.conflicts()
        self.call("volume", "create", self.args.volume, capture=True)
        approved = self.retry(self.release)
        prior = self.load()
        inspector = prior or approved
        self.retry(lambda: self.call("pull", inspector))
        state = self.inspect(inspector)
        current = self.resume(state, prior) or approved
        updates = 0
        while True:
            if self.args.image and current == approved and self.args.image != approved:
                raise ValueError(
                    "Manual image pin is no longer approved; update --image "
                    "or remove it"
                )
            self.conflicts()
            self.retry(lambda: self.call("pull", current))
            self.save(current)
            status = self.worker(current)
            if status not in (0, 10, 75):
                return status
            if status in (0, 10) and (self.args.once or self.args.gpu_health_check):
                return 0
            state = self.inspect(current)
            if state["active"]:
                raise ValueError(
                    "Worker returned a handoff with uncleared active state; restart to "
                    "resume the journalled image"
                )
            if status == 75:
                updates += 1
                if updates >= 5:
                    raise ValueError(
                        "Repeated update-required responses; refresh the "
                        "launcher or contact maintainers"
                    )
                LOGGER.warning(
                    "Worker requires an update; rediscovering (%s/5)", updates
                )
                self.sleep(min(2**updates, 30))
            else:
                updates = 0
                if status == 10:
                    self.sleep(30)
            approved = self.retry(self.release)
            current = approved

    def call(
        self, *parts: str, capture: bool = False, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Execute a Docker command in the selected context.

        Args:
            parts: Docker arguments.
            capture: Capture text output instead of inheriting the terminal.
            check: Raise on nonzero exit.

        Returns:
            The subprocess-compatible result.

        Raises:
            KeyboardInterrupt: If the operation cannot proceed safely.
        """
        if self.cancelled:
            raise KeyboardInterrupt
        return self.command(
            self.docker + list(parts), check=check, capture_output=capture, text=True
        )

    def conflicts(self) -> None:
        """Refuse every existing container using the cache, including stopped ones.

        Raises:
            ValueError: If the operation cannot proceed safely.
        """
        if self.call(
            "ps",
            "-a",
            "--filter",
            f"volume={self.args.volume}",
            "--format",
            "{{.ID}}",
            capture=True,
        ).stdout.strip():
            raise ValueError(
                "Another container uses this cache; resolve it manually (no "
                "containers were stopped)"
            )

    def inspect(self, image: str) -> dict[str, Any]:
        """Inspect cache read-only without requesting a GPU.

        Args:
            image: Immutable inspector image.

        Returns:
            Validated worker state summary.

        Raises:
            ValueError: If the operation cannot proceed safely.
        """
        result = self.call(
            "run",
            "--rm",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=void",
            "-v",
            f"{self.args.volume}:/cache:ro",
            "--pull=never",
            image,
            "--state-dir",
            self.args.state_dir,
            "--inspect-state",
            capture=True,
        )
        state = json.loads(result.stdout)
        if (
            not isinstance(state, dict)
            or state.get("schema_version") != "volunteer-worker-state/v1"
            or any(type(state.get(key)) is not bool for key in ("active", "finalised"))
        ):
            raise ValueError("Unrecognised cache inspection schema")
        for key in ("managed_image_digest", "lease_image_digest"):
            if key not in state:
                raise ValueError("Incomplete cache inspection schema")
            if state.get(key) is not None and (
                not isinstance(state[key], str) or not DIGEST.fullmatch(state[key])
            ):
                raise ValueError("Invalid cache image provenance")
        return state

    def load(self) -> str | None:
        """Load and validate the nonsecret host journal.

        Returns:
            Original immutable image, if recorded.

        Raises:
            ValueError: If the operation cannot proceed safely.
        """
        if not self.journal.exists():
            return None
        data = json.loads(self.journal.read_text())
        if not isinstance(data, dict):
            raise ValueError("Invalid host journal")
        if (
            data.get("schema_version") != "volunteer-launcher-state/v1"
            or data.get("volume") != self.args.volume
            or data.get("state_dir") != self.args.state_dir
        ):
            raise ValueError(
                "Journal/cache configuration mismatch; restore the original "
                "launcher options"
            )
        _digest(data.get("image"))
        return data["image"]

    def resume(self, state: dict[str, Any], prior: str | None) -> str | None:
        """Require exact provenance until active state has been cleared.

        Args:
            state: Read-only cache summary.
            prior: Host journal image.

        Returns:
            Original image for any active state, otherwise None.

        Raises:
            ValueError: If the operation cannot proceed safely.
        """
        # Finalised leases still undergo runtime provenance checks before cleanup.
        if not state["active"]:
            return None
        managed = state.get("managed_image_digest")
        lease = state.get("lease_image_digest")
        if not prior or not managed:
            raise ValueError(
                "Legacy active work has unknown provenance. Resume with the "
                "original manual Docker image/cache until completion; do not "
                "delete state, then restart this launcher"
            )
        if managed != _digest(prior) or (lease is not None and lease != managed):
            raise ValueError(
                "Host journal and active cache image disagree; restore "
                "original journal/image; do not delete state"
            )
        return prior

    def retry(self, action: Callable[[], T]) -> T:
        """Retry discovery/pull with visible bounded backoff.

        Args:
            action: Operation which cannot claim work.

        Returns:
            Successful operation result.

        Raises:
            OSError: If the operation cannot proceed safely.
            RuntimeError: If the operation cannot proceed safely.
            subprocess.SubprocessError: If the operation cannot proceed safely.
        """
        for attempt in range(5):
            try:
                return action()
            except (OSError, subprocess.SubprocessError) as error:
                if attempt == 4:
                    raise
                delay = min(2**attempt, 30)
                LOGGER.warning("%s; retrying in %ss (%s/5)", error, delay, attempt + 1)
                self.sleep(delay)
        raise RuntimeError("Unreachable retry state")

    def save(self, image: str) -> None:
        """Atomically journal the exact image before starting a writable container.

        Args:
            image: Official immutable image reference.
        """
        temporary = self.journal.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(
                {
                    "schema_version": "volunteer-launcher-state/v1",
                    "image": image,
                    "volume": self.args.volume,
                    "state_dir": self.args.state_dir,
                },
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.journal)

    def worker(self, image: str) -> int:
        """Run one safe handoff with signal ownership and no automatic Docker pull.

        Args:
            image: Pulled, journalled immutable image reference.

        Returns:
            Worker status, or 130 after cancellation.
        """
        name = "euroeval-worker-" + uuid.uuid4().hex
        previous = {
            sig: signal.signal(sig, self.interrupt)
            for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            self.owned = name
            result = self.call(
                "run",
                "--rm",
                "-i",
                "--name",
                name,
                "--pull=never",
                "--gpus",
                self.args.gpus,
                "--read-only",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=2g",
                "--shm-size=2g",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit=512",
                "-e",
                "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                "-v",
                f"{self.args.volume}:/cache",
                image,
                *(["--gpu-health-check"] if self.args.gpu_health_check else []),
                "--server",
                self.args.server,
                "--state-dir",
                self.args.state_dir,
                "--launcher-managed",
                "--image-digest",
                _digest(image),
                "--gpu-memory-utilisation",
                str(self.args.gpu_memory_utilisation),
                check=False,
            )
            if self.cancelled:
                # The attached Docker client may exit before the container. Wait only
                # for our owned name; --rm may already have removed it.
                self.command(
                    self.docker + ["wait", name],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                return 130
            return result.returncode
        finally:
            self.owned = None
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def _digest(image: object) -> str:
    """Validate an official immutable image reference.

    Args:
        image: Untrusted image reference.

    Returns:
        Validated manifest-index digest.

    Raises:
        ValueError: If the operation cannot proceed safely.
    """
    if not isinstance(image, str) or not re.fullmatch(
        re.escape(REPOSITORY) + r"@sha256:[0-9a-f]{64}", image
    ):
        raise ValueError("Image must be an official immutable digest reference")
    return image.split("@", 1)[1]


def _parser() -> argparse.ArgumentParser:
    """Build the small host CLI.

    Returns:
        CLI parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--gpu-health-check",
        action="store_true",
        help="Run the worker GPU diagnostic once without claiming work",
    )
    parser.add_argument(
        "--server",
        choices=("https://euroeval.com/api/worker",),
        default="https://euroeval.com/api/worker",
    )
    parser.add_argument("--image", help="Manual official repository@sha256:index pin")
    parser.add_argument(
        "--context", help="Docker context (defaults to current context)"
    )
    parser.add_argument(
        "--architecture",
        choices=("amd64", "arm64"),
        help="Require this Docker host architecture",
    )
    parser.add_argument(
        "--gpus", default="all", help="Docker GPU selection, e.g. device=0"
    )
    parser.add_argument("--volume", default="euroeval-worker-cache")
    parser.add_argument(
        "--state-dir", default="/cache", help="Worker state location in the cache"
    )
    parser.add_argument(
        "--launcher-state-dir", default="~/.cache/euroeval-worker-launcher"
    )
    parser.add_argument(
        "--gpu-memory-utilisation", "--gpu-memory-utilization", type=float, default=0.8
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
