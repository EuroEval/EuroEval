"""Tests for the collect_evaluation_results script, focusing on upload_results_to_hf."""

import json
import signal
import typing as t
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from euroeval_worker.review import PublicStagingError, VolunteerReviewer
from euroeval_worker.review_models import ReviewReport
from leaderboards import constants
from src.scripts import collect_evaluation_results


class FakeHfApi:
    """In-memory adapter for canonical result-bucket operations."""

    objects: dict[str, bytes] = {}

    def batch_bucket_files(
        self,
        bucket_id: str,
        add: list[tuple[str | Path | bytes, str]] | None = None,
        delete: list[str] | None = None,
        **kwargs,
    ) -> None:
        """No-op batch upload for testing.

        Args:
            bucket_id:
                Bucket id supplied to the test.
            add (optional):
                Add supplied to the test. Defaults to None.
            delete (optional):
                Delete supplied to the test. Defaults to None.
            kwargs:
                Kwargs supplied to the test.
        """
        del bucket_id, kwargs
        for local, name in add or []:
            self.objects[name] = (
                local if isinstance(local, bytes) else Path(local).read_bytes()
            )
        for name in delete or []:
            self.objects.pop(name, None)

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[str, Path]],
        *,
        raise_on_missing_files: bool,
        token: str,
    ) -> None:
        """Copy requested objects into their local destinations."""
        del bucket_id, raise_on_missing_files, token
        for name, destination in files:
            if name in self.objects:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(self.objects[name])

    def list_bucket_tree(
        self, bucket_id: str, prefix: str, recursive: bool, token: str
    ) -> list[object]:
        """Return no entries for test setups that do not seed a bucket."""
        del bucket_id, prefix, recursive, token
        return []

    def sync_bucket(
        self,
        source: str,
        dest: str,
        token: str | None = None,
        ignore_times: bool = False,
        **kwargs,
    ) -> None:
        """No-op sync for testing.

        Args:
            source:
                Source supplied to the test.
            dest:
                Dest supplied to the test.
            token (optional):
                Token supplied to the test. Defaults to None.
            ignore_times (optional):
                Ignore times supplied to the test. Defaults to False.
            kwargs:
                Kwargs supplied to the test.
        """
        pass


@pytest.fixture(autouse=True)
def clear_bucket_adapter() -> None:
    """Keep each test's in-memory bucket isolated."""
    FakeHfApi.objects.clear()


def test_initial_yes_does_not_approve_without_item_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overview consent alone cannot approve the pending submission."""
    monkeypatch.setenv("HF_STAGING_BUCKET", "private/staging")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setenv("GITHUB_ACTOR", "maintainer")
    monkeypatch.setattr(
        collect_evaluation_results,
        "reviewer_from_environment",
        lambda: PreflightReviewer(["one"]),
    )
    monkeypatch.setattr(
        collect_evaluation_results, "_stdin_is_interactive", lambda: True
    )
    answers = iter(["y", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    command = Mock(return_value=0)
    monkeypatch.setattr(collect_evaluation_results, "run_review_command", command)

    assert collect_evaluation_results.preflight_volunteer_review() is True
    command.assert_not_called()


class PreflightReviewer:
    """Read-only reviewer double for collection preflight tests."""

    def __init__(self, submission_ids: list[str] | None = None) -> None:
        """Initialise pending IDs for the test.

        Args:
            submission_ids (optional):
                Pending IDs. Defaults to the original unsafe-looking ID.
        """
        self.submission_ids = submission_ids or ["submission id'$(touch /tmp/nope)"]

    def decide(self, *args: object, **kwargs: object) -> None:
        """Fail if preflight ever attempts a decision.

        Args:
            args:
                Args supplied to the test.
            kwargs:
                Kwargs supplied to the test.

        Raises:
            AssertionError:
                If preflight attempts to decide a submission.
        """
        raise AssertionError("preflight must not decide")

    def list_pending_submissions(self) -> list[tuple[str, int, str, str]]:
        """Return one deliberately oddly quoted submission ID.

        Returns:
            A single pending submission tuple with an unsafe-looking ID.
        """
        return [
            (submission_id, 12, "alice", "da") for submission_id in self.submission_ids
        ]

    def show(self, submission_id: str) -> object:
        """Return deterministic displayed evidence for the public show workflow.

        Args:
            submission_id: Submission to display.

        Returns:
            A report-shaped object with a stable approval fingerprint.
        """
        return ReviewReport(
            submission_id=submission_id,
            issue_number=12,
            contributor="alice",
            model_id="model",
            model_revision="revision",
            language="da",
            euroeval_version="1.0",
            provenance={},
            expected_identities=(),
            records=(),
            canonical_observations=(),
            plan_fingerprint=f"fingerprint-{submission_id}",
            warnings=(),
            checks=(),
        )


def test_main_does_not_harvest_after_review_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed promotion preflight stops main before any collection operation."""
    monkeypatch.setattr(
        collect_evaluation_results, "check_required_env_vars", lambda: None
    )
    monkeypatch.setattr(
        collect_evaluation_results, "preflight_volunteer_review", lambda: False
    )
    fetch = Mock(side_effect=AssertionError("harvest started"))
    monkeypatch.setattr(collect_evaluation_results, "_fetch_issues", fetch)
    callback = collect_evaluation_results.main.callback
    assert callback is not None
    callback(force=False)
    fetch.assert_not_called()


def test_main_pauses_before_harvesting(monkeypatch: pytest.MonkeyPatch) -> None:
    """A review pause occurs before the first GitHub operation.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    monkeypatch.setattr(
        collect_evaluation_results, "check_required_env_vars", lambda: None
    )
    monkeypatch.setattr(
        collect_evaluation_results, "preflight_volunteer_review", lambda: False
    )
    monkeypatch.setattr(
        collect_evaluation_results,
        "_fetch_issues",
        Mock(side_effect=AssertionError("harvest started")),
    )

    callback = collect_evaluation_results.main.callback
    assert callback is not None
    assert callback(force=False) is None


@pytest.mark.parametrize("failure", ["timeout", "oserror"])
def test_missing_reviewer_login_stops_opted_in_review(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A missing GitHub login cancels preflight without a decision command."""
    monkeypatch.setenv("HF_STAGING_BUCKET", "private/staging")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.delenv("GITHUB_ACTOR", raising=False)
    monkeypatch.setattr(
        collect_evaluation_results,
        "reviewer_from_environment",
        lambda: PreflightReviewer(["one"]),
    )
    monkeypatch.setattr(
        collect_evaluation_results, "_stdin_is_interactive", lambda: True
    )
    monkeypatch.setattr("builtins.input", lambda _: "y")
    error: Exception = (
        collect_evaluation_results.subprocess.TimeoutExpired("gh", 10)
        if failure == "timeout"
        else OSError("gh unavailable")
    )
    gh = Mock(side_effect=error)
    monkeypatch.setattr(collect_evaluation_results.subprocess, "run", gh)
    command = Mock(side_effect=AssertionError("approval must not run"))
    monkeypatch.setattr(collect_evaluation_results, "run_review_command", command)

    assert collect_evaluation_results.preflight_volunteer_review() is False
    assert gh.call_args.kwargs["timeout"] == 10
    command.assert_not_called()


@pytest.mark.parametrize("answer", ["", "n", "no"])
def test_preflight_default_or_no_continues_without_approval(
    monkeypatch: pytest.MonkeyPatch, answer: str, caplog: pytest.LogCaptureFixture
) -> None:
    """No and the default answer continue without approving anything.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        answer:
            Answer supplied to the test.
        caplog:
            Fixture for capturing log records during the test.
    """
    monkeypatch.setenv("HF_STAGING_BUCKET", "private/staging")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    caplog.set_level("INFO", logger="collect_evaluation_results")
    monkeypatch.setattr(
        collect_evaluation_results,
        "reviewer_from_environment",
        lambda: PreflightReviewer(),
    )
    monkeypatch.setattr(
        collect_evaluation_results, "_stdin_is_interactive", lambda: True
    )
    monkeypatch.setattr("builtins.input", lambda prompt: answer)

    assert collect_evaluation_results.preflight_volunteer_review() is True
    assert "no volunteer submission was approved automatically" in caplog.text


def test_preflight_eof_shows_commands_and_continues(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """EOF cannot leave the collection command waiting for a response.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        caplog:
            Fixture for capturing log records during the test.
    """
    monkeypatch.setenv("HF_STAGING_BUCKET", "private/staging")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setattr(
        collect_evaluation_results,
        "reviewer_from_environment",
        lambda: PreflightReviewer(),
    )
    monkeypatch.setattr(
        collect_evaluation_results, "_stdin_is_interactive", lambda: True
    )
    monkeypatch.setattr("builtins.input", Mock(side_effect=EOFError))

    assert collect_evaluation_results.preflight_volunteer_review() is False
    assert "cancelled" in caplog.text.lower()


@pytest.mark.parametrize("interrupt", [EOFError, KeyboardInterrupt])
@pytest.mark.parametrize("prompt_number", [1, 2, 3])
def test_preflight_interrupt_stops_without_harvesting(
    monkeypatch: pytest.MonkeyPatch, interrupt: type[BaseException], prompt_number: int
) -> None:
    """Interrupts at the overview or any item prompt stop collection."""
    monkeypatch.setenv("HF_STAGING_BUCKET", "private/staging")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setenv("GITHUB_ACTOR", "maintainer")
    monkeypatch.setattr(
        collect_evaluation_results,
        "reviewer_from_environment",
        lambda: PreflightReviewer(["first", "second"]),
    )
    monkeypatch.setattr(
        collect_evaluation_results, "_stdin_is_interactive", lambda: True
    )
    inputs = iter(range(prompt_number - 1))
    count = 0

    def answer(_: str) -> str:
        nonlocal count
        count += 1
        if count == prompt_number:
            raise interrupt
        next(inputs)
        return "y"

    monkeypatch.setattr("builtins.input", answer)
    command = Mock(return_value=0)
    monkeypatch.setattr(collect_evaluation_results, "run_review_command", command)
    assert collect_evaluation_results.preflight_volunteer_review() is False
    assert command.call_count == (1 if prompt_number == 3 else 0)


def test_preflight_missing_staging_configuration_is_non_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing optional staging configuration does not block canonical work.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    monkeypatch.delenv("HF_STAGING_BUCKET", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert collect_evaluation_results.preflight_volunteer_review() is True


def test_preflight_mixed_decisions_review_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review commands retain each ID and only explicit yes approves."""
    monkeypatch.setenv("HF_STAGING_BUCKET", "private/staging")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setenv("GITHUB_ACTOR", "maintainer")
    ids = ["submission-one", "submission-two", "submission-three"]
    monkeypatch.setattr(
        collect_evaluation_results,
        "reviewer_from_environment",
        lambda: PreflightReviewer(ids),
    )
    monkeypatch.setattr(
        collect_evaluation_results, "_stdin_is_interactive", lambda: True
    )
    answers = iter(["y", "y", "n", ""])
    prompts: list[str] = []
    monkeypatch.setattr(
        "builtins.input", lambda prompt: prompts.append(prompt) or next(answers)
    )
    command = Mock(return_value=0)
    monkeypatch.setattr(collect_evaluation_results, "run_review_command", command)

    assert collect_evaluation_results.preflight_volunteer_review() is True
    assert command.call_args_list == [
        call(
            [
                "--reviewer",
                "maintainer",
                "approve",
                "submission-one",
                "--plan-fingerprint",
                "fingerprint-submission-one",
                "--reason",
                "Reviewed during collector preflight",
            ]
        )
    ]
    assert len(prompts) == 4


def test_preflight_non_tty_and_api_failure_do_not_hang(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Non-TTY input and staging failures continue without approval.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        caplog:
            Fixture for capturing log records during the test.
    """
    monkeypatch.setenv("HF_STAGING_BUCKET", "private/staging")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setattr(
        collect_evaluation_results,
        "reviewer_from_environment",
        lambda: PreflightReviewer(),
    )
    monkeypatch.setattr(
        collect_evaluation_results, "_stdin_is_interactive", lambda: False
    )

    assert collect_evaluation_results.preflight_volunteer_review() is True
    assert "not interactive" in caplog.text
    assert "submission id'\"'\"'$(touch /tmp/nope)" in caplog.text

    monkeypatch.setattr(
        collect_evaluation_results,
        "reviewer_from_environment",
        Mock(side_effect=RuntimeError("token=secret https://private.example")),
    )
    assert collect_evaluation_results.preflight_volunteer_review() is True
    assert "preflight unavailable" in caplog.text
    assert "secret" not in caplog.text
    assert "private.example" not in caplog.text


def test_preflight_public_staging_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A confirmed public staging bucket is never treated as unavailable.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    monkeypatch.setenv("HF_STAGING_BUCKET", "public/staging")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setattr(
        collect_evaluation_results,
        "reviewer_from_environment",
        Mock(side_effect=PublicStagingError("HF_STAGING_BUCKET must be private")),
    )

    with pytest.raises(PublicStagingError):
        collect_evaluation_results.preflight_volunteer_review()


@pytest.mark.parametrize("failure", ["status", "exception"])
@pytest.mark.parametrize("command_kind", ["show", "approve"])
def test_preflight_review_error_halts_after_inspecting_remaining_ids(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    command_kind: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed command does not stop later review but blocks collection."""
    monkeypatch.setenv("HF_STAGING_BUCKET", "private/staging")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setenv("GITHUB_ACTOR", "maintainer")
    ids = ["first", "second", "third"]
    monkeypatch.setattr(
        collect_evaluation_results,
        "reviewer_from_environment",
        lambda: PreflightReviewer(ids),
    )
    monkeypatch.setattr(
        collect_evaluation_results, "_stdin_is_interactive", lambda: True
    )
    prompt = Mock(return_value="y")
    monkeypatch.setattr("builtins.input", prompt)
    calls: list[list[str]] = []
    shown: list[str] = []
    real_show_and_render = collect_evaluation_results.show_and_render

    def display(*, submission_id: str, reviewer: PreflightReviewer) -> object:
        shown.append(submission_id)
        if command_kind == "show" and submission_id == "first":
            if failure == "exception":
                raise RuntimeError("injected display failure")
            raise RuntimeError("injected display failure")
        return real_show_and_render(
            submission_id=submission_id, reviewer=t.cast(VolunteerReviewer, reviewer)
        )

    def run(command: list[str]) -> int:
        calls.append(command)
        failed = command[0] == "--reviewer" and command[3] == "first"
        if failed and failure == "exception":
            raise RuntimeError("injected approval failure")
        return 1 if failed else 0

    monkeypatch.setattr(collect_evaluation_results, "show_and_render", display)
    monkeypatch.setattr(collect_evaluation_results, "run_review_command", run)
    assert collect_evaluation_results.preflight_volunteer_review() is False
    assert shown == ids
    assert prompt.call_count == (3 if command_kind == "show" else 4)
    assert "halting collection" in caplog.text


def test_preflight_with_no_pending_reviews_continues_without_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty pending inventory proceeds without prompting or CLI calls."""
    monkeypatch.setenv("HF_STAGING_BUCKET", "private/staging")
    monkeypatch.setenv("HF_TOKEN", "secret-token")

    class EmptyReviewer(PreflightReviewer):
        """Reviewer with no pending submissions."""

        def list_pending_submissions(self) -> list[tuple[str, int, str, str]]:
            """Return an empty inventory.

            Returns:
                An empty pending-submission inventory.
            """
            return []

    monkeypatch.setattr(
        collect_evaluation_results, "reviewer_from_environment", EmptyReviewer
    )
    prompt = Mock(side_effect=AssertionError("should not prompt"))
    command = Mock(side_effect=AssertionError("should not run a command"))
    monkeypatch.setattr("builtins.input", prompt)
    monkeypatch.setattr(collect_evaluation_results, "run_review_command", command)

    assert collect_evaluation_results.preflight_volunteer_review() is True
    prompt.assert_not_called()
    command.assert_not_called()


def test_preflight_yes_reviews_submission(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Opt-in reviews and decides only after a separate approval response.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        caplog:
            Fixture for capturing log records during the test.
    """
    monkeypatch.setenv("HF_STAGING_BUCKET", "private/staging")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setenv("GITHUB_ACTOR", "maintainer; echo nope")
    caplog.set_level("INFO", logger="collect_evaluation_results")
    monkeypatch.setattr(
        collect_evaluation_results,
        "reviewer_from_environment",
        lambda: PreflightReviewer(),
    )
    monkeypatch.setattr(
        collect_evaluation_results, "_stdin_is_interactive", lambda: True
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    command = Mock(return_value=0)
    monkeypatch.setattr(collect_evaluation_results, "run_review_command", command)

    assert collect_evaluation_results.preflight_volunteer_review() is True
    assert command.call_count == 1
    assert command.call_args_list[0].args[0] == [
        "--reviewer",
        "maintainer; echo nope",
        "approve",
        "submission id'$(touch /tmp/nope)",
        "--plan-fingerprint",
        "fingerprint-submission id'$(touch /tmp/nope)",
        "--reason",
        "Reviewed during collector preflight",
    ]


def test_preview_in_dev_server_handles_early_exit(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An early server exit is reported without failing on missing output.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        caplog:
            Fixture for capturing log records during the test.
    """
    process = FakeDevProcess(return_code=1)

    monkeypatch.setattr(
        collect_evaluation_results.subprocess, "Popen", lambda *args, **kwargs: process
    )
    caplog.set_level("INFO", logger="collect_evaluation_results")

    assert collect_evaluation_results.preview_in_dev_server() is False

    assert "Dev server exited unexpectedly:" in caplog.text


class FakeDevProcess:
    """Fake Vercel process used to test preview lifecycle handling."""

    stdout: None = None

    def __init__(self, return_code: int | None = None) -> None:
        """Initialise the fake process with an optional exit code.

        Args:
            return_code (optional):
                Return code supplied to the test. Defaults to None.
        """
        self.return_code = return_code
        self.signals: list[signal.Signals] = []
        self.wait_timeouts: list[float | None] = []

    def kill(self) -> None:
        """Fail if the test unexpectedly needs to kill the process.

        Raises:
            AssertionError: Always, because killing is unexpected in these tests.
        """
        raise AssertionError("The fake process should not need to be killed.")

    def poll(self) -> int | None:
        """Return the configured process exit code.

        Returns:
            The fake process exit code, or ``None`` while it is running.
        """
        return self.return_code

    def send_signal(self, sig: signal.Signals) -> None:
        """Record a signal sent to the process.

        Args:
            sig:
                Sig supplied to the test.
        """
        self.signals.append(sig)

    def wait(self, timeout: float | None = None) -> None:
        """Record the timeout used while waiting for the process.

        Args:
            timeout (optional):
                Timeout supplied to the test. Defaults to None.
        """
        self.wait_timeouts.append(timeout)


def test_preview_in_dev_server_starts_and_stops_fixed_port(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preview uses the fixed Vite port and terminates the server afterwards.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        caplog:
            Fixture for capturing log records during the test.
        capsys:
            Fixture for capturing output during the test.
    """
    process = FakeDevProcess()
    popen_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_popen(*args: object, **kwargs: object) -> FakeDevProcess:
        """Fake popen for this test.

        Args:
            args:
                Args supplied to the test.
            kwargs:
                Kwargs supplied to the test.

        Returns:
            The value returned by the test helper.
        """
        popen_calls.append((args, kwargs))
        return process

    socket_instance = Mock()
    socket_factory = Mock(return_value=socket_instance)
    monkeypatch.setattr(collect_evaluation_results.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        collect_evaluation_results.socket, "create_connection", socket_factory
    )
    monkeypatch.setattr(collect_evaluation_results.time, "time", lambda: 0.0)
    monkeypatch.setattr(collect_evaluation_results.time, "sleep", lambda _: None)
    monkeypatch.setattr("builtins.input", lambda _: "y")
    caplog.set_level("INFO", logger="collect_evaluation_results")

    assert collect_evaluation_results.preview_in_dev_server() is True

    assert popen_calls == [
        (
            (["vercel", "dev", "--yes", "--non-interactive"],),
            {
                "cwd": collect_evaluation_results.REPO_ROOT,
                "stdin": collect_evaluation_results.subprocess.DEVNULL,
                "stdout": collect_evaluation_results.subprocess.PIPE,
                "stderr": collect_evaluation_results.subprocess.STDOUT,
                "text": True,
            },
        )
    ]
    socket_factory.assert_called_once_with(("localhost", 5173), timeout=1)
    socket_instance.close.assert_called_once_with()
    assert process.signals == [signal.SIGTERM]
    assert process.wait_timeouts == [10]
    assert "http://localhost:5173" in capsys.readouterr().out
    assert "http://localhost:5173" in caplog.text


def test_reviewer_login_actor_and_safe_gh_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight uses its actor or bounded authenticated gh fallback."""
    monkeypatch.setenv("HF_STAGING_BUCKET", "private/staging")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setenv("GITHUB_ACTOR", "actor")
    monkeypatch.setattr(
        collect_evaluation_results,
        "reviewer_from_environment",
        lambda: PreflightReviewer(["one"]),
    )
    monkeypatch.setattr(
        collect_evaluation_results, "_stdin_is_interactive", lambda: True
    )
    answers = iter(["y", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    command = Mock(return_value=0)
    monkeypatch.setattr(collect_evaluation_results, "run_review_command", command)
    subprocess_run = Mock(side_effect=AssertionError("gh must not run"))
    monkeypatch.setattr(collect_evaluation_results.subprocess, "run", subprocess_run)

    assert collect_evaluation_results.preflight_volunteer_review() is True
    assert command.call_args_list[-1] == call(
        [
            "--reviewer",
            "actor",
            "approve",
            "one",
            "--plan-fingerprint",
            "fingerprint-one",
            "--reason",
            "Reviewed during collector preflight",
        ]
    )
    subprocess_run.assert_not_called()

    monkeypatch.delenv("GITHUB_ACTOR")
    answers = iter(["y", "y"])
    subprocess_run.side_effect = None
    subprocess_run.return_value = Mock(stdout="fallback-login\n")
    command.reset_mock()
    assert collect_evaluation_results.preflight_volunteer_review() is True
    assert subprocess_run.call_args.args[0] == ["gh", "api", "user", "--jq", ".login"]
    assert subprocess_run.call_args.kwargs["timeout"] == 10
    assert command.call_args_list[-1] == call(
        [
            "--reviewer",
            "fallback-login",
            "approve",
            "one",
            "--plan-fingerprint",
            "fingerprint-one",
            "--reason",
            "Reviewed during collector preflight",
        ]
    )


def test_upload_results_to_hf_collision_leaves_results_dir_unmutated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Distinct-identity path collision raises AND leaves RESULTS_DIR unmutated.

    Two records with different identities that sanitise to the same path should
    raise ValueError BEFORE any files are deleted from RESULTS_DIR.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create existing record in RESULTS_DIR
    # Dataset "dataset/one" sanitises to "dataset_one"
    model_dir = results_dir / "foo_bar"
    model_dir.mkdir()
    existing_record_file = model_dir / "dataset_one__test__zeroshot.json"
    existing_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset/one",  # Sanitises to "dataset_one"
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704067200,  # 2024-01-01T00:00:00Z as Unix timestamp
    }
    existing_record_file.write_text(json.dumps(existing_record), encoding="utf-8")

    # Create new results file with a colliding identity
    # Dataset "dataset_one" also sanitises to "dataset_one" - same path!
    new_results_file = tmp_path / "new_results.jsonl"
    colliding_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset_one",  # Different identity, same sanitised path
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704153600,  # 2024-01-02T00:00:00Z as Unix timestamp
    }
    new_results_file.write_text(json.dumps(colliding_record), encoding="utf-8")

    # Monkeypatch RESULTS_DIR and HfApi
    monkeypatch.setattr(
        target=collect_evaluation_results, name="RESULTS_DIR", value=results_dir
    )
    monkeypatch.setattr(
        target=collect_evaluation_results, name="HfApi", value=FakeHfApi
    )

    # Should raise ValueError due to collision
    with pytest.raises(ValueError, match="Identity collision detected"):
        collect_evaluation_results.upload_results_to_hf(
            new_results_path=new_results_file
        )

    # Existing record should STILL exist (no data loss)
    assert existing_record_file.exists(), "Existing record was incorrectly deleted"
    content = json.loads(existing_record_file.read_text(encoding="utf-8"))
    assert content == existing_record


def test_upload_results_to_hf_deletes_stale_jsonl_in_results_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stale root RESULTS_DIR/*.jsonl is deleted on upload.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create stale jsonl file in RESULTS_DIR
    stale_jsonl = results_dir / "stale.jsonl"
    stale_jsonl.write_text('{"old": "data"}', encoding="utf-8")

    model_dir = results_dir / "foo_bar"
    model_dir.mkdir()
    existing_record_file = model_dir / "dataset__test__zeroshot.json"
    existing_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704067200,  # 2024-01-01T00:00:00Z
    }
    existing_record_file.write_text(json.dumps(existing_record), encoding="utf-8")

    # New results file
    new_results_file = tmp_path / "new_results.jsonl"
    new_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704067200,  # 2024-01-01T00:00:00Z
    }
    new_results_file.write_text(json.dumps(new_record), encoding="utf-8")

    # Monkeypatch RESULTS_DIR and HfApi
    monkeypatch.setattr(
        target=collect_evaluation_results, name="RESULTS_DIR", value=results_dir
    )
    monkeypatch.setattr(
        target=collect_evaluation_results, name="HfApi", value=FakeHfApi
    )

    # Should succeed
    result = collect_evaluation_results.upload_results_to_hf(
        new_results_path=new_results_file
    )
    assert result is True

    # Stale jsonl should be deleted
    assert not stale_jsonl.exists(), "Stale jsonl in RESULTS_DIR was not deleted"


def test_upload_results_to_hf_does_not_touch_repo_root_jsonl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Repo-root JSONL files are NOT touched by upload.

    The upload function only touches RESULTS_DIR/*.jsonl files, not repo-root
    JSONL files like new_results.jsonl or euroeval_benchmark_results.jsonl.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    repo_root = tmp_path / "repo_root"
    repo_root.mkdir()

    model_dir = results_dir / "foo_bar"
    model_dir.mkdir()
    existing_record_file = model_dir / "dataset__test__zeroshot.json"
    existing_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704067200,  # 2024-01-01T00:00:00Z
    }
    existing_record_file.write_text(json.dumps(existing_record), encoding="utf-8")

    # Create repo-root JSONL files that should NOT be touched
    new_results_file = repo_root / "new_results.jsonl"
    new_results_file.write_text('{"data": "new"}', encoding="utf-8")
    benchmark_results_file = repo_root / "euroeval_benchmark_results.jsonl"
    benchmark_results_file.write_text('{"data": "benchmark"}', encoding="utf-8")

    # Use new_results_file as input (will be read, not deleted by upload)
    input_file = tmp_path / "input.jsonl"
    input_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704067200,  # 2024-01-01T00:00:00Z
    }
    input_file.write_text(json.dumps(input_record), encoding="utf-8")

    # Monkeypatch RESULTS_DIR, REPO_ROOT, and HfApi
    monkeypatch.setattr(
        target=collect_evaluation_results, name="RESULTS_DIR", value=results_dir
    )
    monkeypatch.setattr(target=constants, name="REPO_ROOT", value=repo_root)
    monkeypatch.setattr(
        target=collect_evaluation_results, name="HfApi", value=FakeHfApi
    )

    # Should succeed
    result = collect_evaluation_results.upload_results_to_hf(
        new_results_path=input_file
    )
    assert result is True

    # Repo-root jsonl files should still exist untouched
    assert new_results_file.exists(), "new_results.jsonl was incorrectly deleted"
    assert benchmark_results_file.exists(), (
        "euroeval_benchmark_results.jsonl was incorrectly deleted"
    )
    assert new_results_file.read_text(encoding="utf-8") == '{"data": "new"}'
    assert benchmark_results_file.read_text(encoding="utf-8") == '{"data": "benchmark"}'


def test_upload_results_to_hf_same_identity_keeps_newer_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same-identity duplicate keeps the NEWER record.

    When two records share the same identity, the newer one wins (by euroeval_version,
    then retrieved_timestamp).

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create existing record with older version
    model_dir = results_dir / "foo_bar"
    model_dir.mkdir()
    existing_record_file = model_dir / "dataset__test__zeroshot.json"
    older_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704067200,  # 2024-01-01T00:00:00Z
    }
    existing_record_file.write_text(json.dumps(older_record), encoding="utf-8")

    # Create new results file with newer version (same identity)
    new_results_file = tmp_path / "new_results.jsonl"
    newer_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "2.0.0",  # Newer version
        },
        "retrieved_timestamp": 1704067200,  # 2024-01-01T00:00:00Z
    }
    new_results_file.write_text(json.dumps(newer_record), encoding="utf-8")

    # Monkeypatch RESULTS_DIR and HfApi
    monkeypatch.setattr(
        target=collect_evaluation_results, name="RESULTS_DIR", value=results_dir
    )
    monkeypatch.setattr(
        target=collect_evaluation_results, name="HfApi", value=FakeHfApi
    )

    # Should succeed
    result = collect_evaluation_results.upload_results_to_hf(
        new_results_path=new_results_file
    )
    assert result is True

    # The newer record should be written
    content = json.loads(existing_record_file.read_text(encoding="utf-8"))
    assert content["eval_library"]["version"] == "2.0.0"


def test_upload_results_to_hf_same_identity_uses_timestamp_tiebreaker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same-identity with same version uses retrieved_timestamp as tiebreaker.

    When versions are equal, the record with newer retrieved_timestamp wins.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create existing record with older timestamp
    model_dir = results_dir / "foo_bar"
    model_dir.mkdir()
    existing_record_file = model_dir / "dataset__test__zeroshot.json"
    older_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704067200,  # 2024-01-01T00:00:00Z
    }
    existing_record_file.write_text(json.dumps(older_record), encoding="utf-8")

    # Create new results file with newer timestamp (same version)
    new_results_file = tmp_path / "new_results.jsonl"
    newer_record = {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",  # Same version
        },
        "retrieved_timestamp": 1704153600,  # 2024-01-02T00:00:00Z (newer)
    }
    new_results_file.write_text(json.dumps(newer_record), encoding="utf-8")

    # Monkeypatch RESULTS_DIR and HfApi
    monkeypatch.setattr(
        target=collect_evaluation_results, name="RESULTS_DIR", value=results_dir
    )
    monkeypatch.setattr(
        target=collect_evaluation_results, name="HfApi", value=FakeHfApi
    )

    # Should succeed
    result = collect_evaluation_results.upload_results_to_hf(
        new_results_path=new_results_file
    )
    assert result is True

    # The newer record (by timestamp) should be written
    content = json.loads(existing_record_file.read_text(encoding="utf-8"))
    assert content["retrieved_timestamp"] == 1704153600
