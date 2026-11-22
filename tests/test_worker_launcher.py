"""Exercise the host launcher through its public command/network seams."""

import fcntl
import json
import signal
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import run_volunteer_worker as launcher

from euroeval_worker import cli, runtime
from euroeval_worker.broker import BrokerClient
from euroeval_worker.state import StateStore

A = "sha256:" + "a" * 64
B = "sha256:" + "b" * 64
REPO = "ghcr.io/euroeval/euroeval-worker"


def test_architecture_mismatch_refused(tmp_path: Path) -> None:
    """An explicit architecture cannot override the actual Docker host."""
    host = FakeHost()
    assert host.run(tmp_path, "--architecture", "arm64") == 1
    assert not host.work


class FakeHost:
    """Model Docker, public discovery and cache state without running containers."""

    def __init__(self) -> None:
        """Initialise an empty cache and one approved release."""
        self.approved = A
        self.codes = [0]
        self.calls: list[list[str]] = []
        self.sleeps: list[float] = []
        self.state: dict[str, Any] = {
            "schema_version": "volunteer-worker-state/v1",
            "active": False,
            "finalised": False,
            "managed_image_digest": None,
            "lease_image_digest": None,
        }
        self.pull_failure = False
        self.network_failure = False
        self.conflict = False
        self.interrupt = False
        self.metadata: dict[str, Any] = {}
        self.work: list[str] = []
        self.change_after_work = False
        self.fail_after_work: str | None = None
        self.directory: Path | None = None

    def command(
        self, args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        """Implement the Docker seam, including writable worker handoffs.

        Args:
            args: Docker command.
            kwargs: subprocess options.

        Returns:
            Fake completed command.

        Raises:
            subprocess.CalledProcessError: If the operation cannot proceed safely.
        """
        self.calls.append(args)
        output = ""
        if "show" in args:
            output = "default"
        elif "info" in args:
            output = "x86_64"
        elif "ps" in args:
            output = "unrelated-cache-owner" if self.conflict else ""
        elif "pull" in args and self.pull_failure:
            raise subprocess.CalledProcessError(1, args)
        elif "--inspect-state" in args:
            assert "--gpus" not in args
            assert any(item.endswith(":/cache:ro") for item in args)
            output = json.dumps(self.state)
        elif "--launcher-managed" in args:
            image = next(item for item in args if item.startswith(REPO + "@"))
            assert self.directory is not None
            journal = json.loads(next(self.directory.glob("*.json")).read_text())
            assert journal["image"] == image
            self.work.append(image)
            if self.fail_after_work:
                setattr(self, self.fail_after_work, True)
            assert args[args.index("--image-digest") + 1] == image.split("@")[1]
            if self.change_after_work:
                self.approved = B
            if self.interrupt:
                handler = signal.getsignal(signal.SIGINT)
                assert callable(handler)
                cast(Callable[..., object], handler)(signal.SIGINT, None)
                return subprocess.CompletedProcess(args, 130, "")
            code = self.codes.pop(0)
            if code == 130:
                self.state.update(active=True, managed_image_digest=image.split("@")[1])
            else:
                self.state.update(active=False, finalised=True)
            return subprocess.CompletedProcess(args, code, "")
        return subprocess.CompletedProcess(args, 0, output)

    def http(self, url: str, **kwargs: object) -> "Response":
        """Return a context-managed official release response.

        Args:
            url: Requested release URL.
            kwargs: Network options.

        Returns:
            Public response fake.

        Raises:
            OSError: If the operation cannot proceed safely.
        """
        if self.network_failure:
            raise OSError("offline")
        arch = url.split("=")[-1]
        data = {
            "protocol_version": "volunteer-worker/v1",
            "launcher_protocol_version": "volunteer-launcher/v1",
            "architecture": arch,
            "image_repository": REPO,
            "image_digest": self.approved,
            "image_reference": REPO + "@" + self.approved,
            "worker_version": "1.0",
            "euroeval_version": "1.0",
        }
        data.update(self.metadata)
        return Response(url, data)

    def run(self, path: Path, *options: str) -> int:
        """Invoke the public launcher CLI.

        Args:
            path: Host journal directory.
            options: Additional CLI arguments.

        Returns:
            Launcher exit status.
        """
        self.directory = path
        return launcher.main(
            ["--launcher-state-dir", str(path), "--architecture", "amd64", *options],
            command=self.command,
            http_get=self.http,
            sleep=self.sleeps.append,
        )


class Response:
    """Minimal urlopen response fake."""

    def __init__(self, url: str, data: dict[str, Any]) -> None:
        """Store a public response.

        Args:
            url: Public discovery URL.
            data: Release metadata.
        """
        self.url = url
        self.data = data

    def __enter__(self) -> "Response":
        """Enter the response context.

        Returns:
            This response.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Close the fake response."""

    def geturl(self) -> str:
        """Return the final response URL.

        Returns:
            Official URL.
        """
        return self.url

    def read(self) -> bytes:
        """Read release metadata.

        Returns:
            JSON bytes.
        """
        return json.dumps(self.data).encode()


@pytest.mark.parametrize("status", [0, 10, 75])
def test_between_job_discovery(tmp_path: Path, status: int) -> None:
    """Every safe handoff rediscovers, pulls and runs the new immutable release."""
    host = FakeHost()
    host.codes = [status, 1]
    host.change_after_work = True
    assert host.run(tmp_path) == 1
    assert host.work == [REPO + "@" + A, REPO + "@" + B]
    assert any("pull" in args and REPO + "@" + B in args for args in host.calls)


def test_cache_container_conflict(tmp_path: Path) -> None:
    """Even a stopped cache owner is never stopped or reused automatically."""
    host = FakeHost()
    host.conflict = True
    assert host.run(tmp_path) == 1
    assert not host.work
    assert not any("kill" in args for args in host.calls)


def test_context_and_gpu_options(tmp_path: Path) -> None:
    """Host context/GPU selection reaches Docker but inspection never gets GPUs."""
    host = FakeHost()
    assert (
        host.run(
            tmp_path,
            "--once",
            "--context",
            "remote",
            "--gpus",
            "device=1",
            "--gpu-memory-utilization",
            "0.5",
        )
        == 0
    )
    assert all(args[:3] == ["docker", "--context", "remote"] for args in host.calls)
    run = next(args for args in host.calls if "--launcher-managed" in args)
    assert run[run.index("--gpus") + 1] == "device=1"
    assert run[run.index("--gpu-memory-utilisation") + 1] == "0.5"


def test_exclusive_lock(tmp_path: Path) -> None:
    """A simultaneous launcher is refused through the host lock."""
    host = FakeHost()
    assert host.run(tmp_path, "--once") == 0
    host.work.clear()
    with next(tmp_path.glob("*.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert host.run(tmp_path, "--once") == 1
    assert not host.work


@pytest.mark.parametrize("failure", ["network_failure", "pull_failure"])
def test_fail_closed(tmp_path: Path, failure: str) -> None:
    """Failed discovery or pull never launches a stale claimant."""
    host = FakeHost()
    setattr(host, failure, True)
    assert host.run(tmp_path) == 1
    assert not host.work
    assert len(host.sleeps) == 4


def test_finalised_handoff_requires_clear(tmp_path: Path) -> None:
    """A handoff with uncleared finalised state cannot overwrite provenance.

    Args:
        tmp_path: Temporary host journal directory.
    """
    host = FakeHost()
    host.change_after_work = True

    def command(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Simulate a handoff interrupted between finalisation and state clear.

        Args:
            args: Docker command.
            kwargs: Subprocess options.

        Returns:
            Fake Docker result with finalised active state retained.
        """
        result = host.command(args, **kwargs)
        if "--launcher-managed" in args:
            host.state.update(
                active=True,
                finalised=True,
                managed_image_digest=A,
                lease_image_digest=A,
            )
        return result

    host.directory = tmp_path
    assert (
        launcher.main(
            ["--launcher-state-dir", str(tmp_path)],
            command=command,
            http_get=host.http,
            sleep=host.sleeps.append,
        )
        == 1
    )
    assert host.work == [REPO + "@" + A]
    assert (
        json.loads(next(tmp_path.glob("*.json")).read_text())["image"] == host.work[0]
    )


def test_first_pull_once_and_journal(tmp_path: Path) -> None:
    """First startup pulls an exact pin and journals before a one-job handoff."""
    host = FakeHost()
    assert host.run(tmp_path, "--once") == 0
    assert host.work == [REPO + "@" + A]
    journal = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert journal["image"] == host.work[0]
    pull = next(i for i, args in enumerate(host.calls) if "pull" in args)
    inspect = next(i for i, args in enumerate(host.calls) if "--inspect-state" in args)
    assert pull < inspect


def test_gpu_diagnostic_once(tmp_path: Path) -> None:
    """The supported diagnostic is forwarded and exits instead of looping."""
    host = FakeHost()
    assert host.run(tmp_path, "--gpu-health-check") == 0
    assert len(host.work) == 1
    assert any("--gpu-health-check" in args for args in host.calls)


@pytest.mark.parametrize("failure", ["network_failure", "pull_failure"])
def test_handoff_failure_never_reuses_stale_claimant(
    tmp_path: Path, failure: str
) -> None:
    """A failed upgrade leaves the last journal intact without a second claim."""
    host = FakeHost()
    host.change_after_work = True
    host.fail_after_work = failure
    assert host.run(tmp_path) == 1
    assert host.work == [REPO + "@" + A]
    assert (
        json.loads(next(tmp_path.glob("*.json")).read_text())["image"] == host.work[0]
    )


def test_journal_configuration_disagreement(tmp_path: Path) -> None:
    """An inconsistent journal fails closed and is not rewritten."""
    host = FakeHost()
    assert host.run(tmp_path, "--once") == 0
    journal = next(tmp_path.glob("*.json"))
    data = json.loads(journal.read_text())
    data["volume"] = "another-cache"
    journal.write_text(json.dumps(data))
    host.work.clear()
    assert host.run(tmp_path) == 1
    assert not host.work
    assert json.loads(journal.read_text()) == data


@pytest.mark.parametrize("once", [False, True])
def test_launcher_native_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, once: bool
) -> None:
    """Real native CLI/run accepts launcher arguments before reaching safe auth.

    Args:
        tmp_path: Temporary host journal directory.
        monkeypatch: Authentication replacement seam.
        once: Whether the host should request a single handoff.
    """
    host = FakeHost()
    authenticated: list[Path] = []

    def stop_at_auth(*, client: BrokerClient, state: StateStore) -> None:
        """Record authentication entry and stop before any external activity.

        Args:
            client: Real native broker client, never called.
            state: Real native state store, never read or written.

        Raises:
            KeyboardInterrupt: Safe sentinel handled by the real managed runtime.
        """
        assert client.server == "https://euroeval.com/api/worker"
        authenticated.append(state.directory)
        raise KeyboardInterrupt

    def command(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Send the actual container argv through the real native CLI/runtime.

        Args:
            args: Docker invocation from the launcher.
            kwargs: Subprocess options.

        Returns:
            Native worker status or fake Docker metadata.
        """
        if "--launcher-managed" not in args:
            return host.command(args, **kwargs)
        image_index = next(
            index for index, item in enumerate(args) if item.startswith(REPO + "@")
        )
        worker_args = args[image_index + 1 :]
        assert "--once" not in worker_args
        assert worker_args[worker_args.index("--image-digest") + 1] == A
        return subprocess.CompletedProcess(args, cli.main(argv=worker_args), "")

    monkeypatch.setattr(runtime, "authenticate", stop_at_auth)
    assert (
        launcher.main(
            ["--launcher-state-dir", str(tmp_path), *(["--once"] if once else [])],
            command=command,
            http_get=host.http,
            sleep=host.sleeps.append,
        )
        == 130
    )
    assert authenticated == [Path("/cache")]


@pytest.mark.parametrize("finalised", [False, True])
@pytest.mark.parametrize("managed", [None, B])
def test_legacy_and_inconsistent_cache_refused(
    tmp_path: Path, managed: str | None, finalised: bool
) -> None:
    """Neither missing provenance nor disagreeing cache permits a new claim."""
    host = FakeHost()
    host.codes = [130]
    assert host.run(tmp_path) == 130
    host.state.update(active=True, finalised=finalised, managed_image_digest=managed)
    host.work.clear()
    journal = next(tmp_path.glob("*.json"))
    original = journal.read_bytes()
    host.approved = B
    assert host.run(tmp_path) == 1
    assert not host.work
    assert journal.read_bytes() == original


def test_manual_pin_cannot_advance(tmp_path: Path) -> None:
    """An explicit obsolete pin cannot silently move to the approved image."""
    host = FakeHost()
    assert host.run(tmp_path, "--image", REPO + "@" + B) == 1
    assert not host.work


def test_manual_pin_stops_on_upgrade(tmp_path: Path) -> None:
    """An initially approved explicit pin never advances silently between jobs."""
    host = FakeHost()
    host.change_after_work = True
    assert host.run(tmp_path, "--image", REPO + "@" + A) == 1
    assert host.work == [REPO + "@" + A]


@pytest.mark.parametrize("status", [0, 10])
def test_once_stops_after_handoff(tmp_path: Path, status: int) -> None:
    """Once mode completes one job or one idle response and exits."""
    host = FakeHost()
    host.codes = [status]
    assert host.run(tmp_path, "--once") == 0
    assert len(host.work) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("worker_version", ""),
        ("euroeval_version", None),
        ("architecture", "arm64"),
        ("image_repository", "evil"),
        ("launcher_protocol_version", "v2"),
        ("image_digest", B),
        ("image_reference", REPO + ":latest"),
    ],
)
def test_release_schema(tmp_path: Path, field: str, value: object) -> None:
    """Malformed or untrusted discovery records fail closed."""
    host = FakeHost()
    host.metadata[field] = value
    assert host.run(tmp_path) == 1
    assert not host.work


def test_repeated_update_is_bounded(tmp_path: Path) -> None:
    """An unchanged approval cannot cause an infinite update-required loop."""
    host = FakeHost()
    host.codes = [75] * 6
    assert host.run(tmp_path) == 1
    assert len(host.work) == 5
    assert len(host.sleeps) == 4


def test_resume_finalised_original_until_clear(tmp_path: Path) -> None:
    """Finalised active state retains its original image/journal until cleanup.

    Args:
        tmp_path: Temporary host journal directory.
    """
    host = FakeHost()
    host.codes = [130]
    assert host.run(tmp_path, "--once") == 130
    host.state.update(finalised=True, lease_image_digest=A)
    journal = next(tmp_path.glob("*.json"))
    original = journal.read_bytes()
    host.approved = B

    # A second interruption before clear must not advance image or journal.
    host.codes = [130]
    assert host.run(tmp_path, "--once") == 130
    assert host.state["active"] and host.state["finalised"]
    assert journal.read_bytes() == original
    assert host.work == [REPO + "@" + A] * 2

    # FakeHost checks the journal at every launch, then clears active on success.
    host.codes = [0, 1]
    assert host.run(tmp_path) == 1
    assert host.work == [REPO + "@" + A] * 3 + [REPO + "@" + B]
    assert json.loads(journal.read_text())["image"] == REPO + "@" + B


def test_resume_original_before_upgrade(tmp_path: Path) -> None:
    """Interrupted managed work resumes its original image despite newer approval."""
    host = FakeHost()
    host.codes = [130]
    assert host.run(tmp_path, "--once") == 130
    host.approved = B
    host.codes = [0]
    assert host.run(tmp_path, "--once") == 0
    assert host.work == [REPO + "@" + A] * 2


def test_signal_only_owned_container(tmp_path: Path) -> None:
    """Ctrl-C targets only the owned container and never restarts."""
    host = FakeHost()
    host.interrupt = True
    assert host.run(tmp_path) == 130
    assert len(host.work) == 1
    kill = next(args for args in host.calls if "kill" in args)
    assert kill[-1].startswith("euroeval-worker-")


@pytest.mark.parametrize("finalised", [False, True])
def test_unknown_legacy_without_journal(tmp_path: Path, finalised: bool) -> None:
    """Legacy active state needs manual migration, not guessed provenance."""
    host = FakeHost()
    host.state.update(active=True, finalised=finalised)
    assert host.run(tmp_path) == 1
    assert not host.work
