"""Tests for the collect_evaluation_results script, focusing on upload_results_to_hf."""

import hashlib
import json
import signal
import typing as t
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from euroeval_worker.review import PublicStagingError
from euroeval_worker.review_models import ReviewReport
from leaderboards import canonical_publication, constants, result_loading
from leaderboards.result_conflicts import ConflictDecision
from leaderboards.result_identity import UnresolvedResultConflictError
from src.scripts import collect_evaluation_results
from tests.leaderboards.test_canonical_publication import MemoryLock


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


def test_collector_abort_preserves_staging_and_local_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An aborted conflict batch has no staging or result-file side effects."""
    results_dir = tmp_path / "results"
    record_path = results_dir / "foo_bar" / "task__test__zeroshot.json"
    record_path.parent.mkdir(parents=True)
    stored = _ranked_collector_record()
    original_record = (json.dumps(stored, indent=2) + "\n").encode()
    record_path.write_bytes(original_record)
    staging = tmp_path / "new_results.jsonl"
    staging.write_text("manual entry\n", encoding="utf-8")
    incoming = json.loads(json.dumps(stored))
    incoming["evaluation_results"][0]["score_details"]["score"] = 0.8
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(collect_evaluation_results, "NEW_RESULTS_PATH", staging)
    monkeypatch.setattr(
        collect_evaluation_results,
        "resolve_result_interactively",
        lambda **_kwargs: (_ for _ in ()).throw(
            UnresolvedResultConflictError("aborted")
        ),
    )
    assert not collect_evaluation_results.upload_results_to_hf(
        new_results_path=staging, pending_lines=[json.dumps(incoming)]
    )
    assert staging.read_text(encoding="utf-8") == "manual entry\n"
    assert record_path.read_bytes() == original_record


def _ranked_collector_record(
    *, version: str = "1.0.0", timestamp: int = 1704067200, score: float = 0.5
) -> dict[str, object]:
    """Build an EEE record with an evaluation payload.

    Args:
        version: EEE library version.
        timestamp: Result retrieval timestamp.
        score: Evaluation score.

    Returns:
        An EEE result record.
    """
    return {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": "task",
                "few_shot": False,
                "validation_split": False,
            },
            "version": version,
        },
        "retrieved_timestamp": timestamp,
        "evaluation_results": [
            {"evaluation_name": "accuracy", "score_details": {"score": score}}
        ],
    }


def test_collector_captures_predecessors_without_second_tree_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Public collection retains exact source bytes using its original tree load."""
    results_dir = tmp_path / "results"
    path = results_dir / "foo_bar" / "task__test__zeroshot.json"
    path.parent.mkdir(parents=True)
    predecessor = (json.dumps(_ranked_collector_record(), indent=2) + "\n").encode()
    path.write_bytes(predecessor)
    incoming = json.loads(json.dumps(_ranked_collector_record()))
    incoming["model_info"]["additional_details"] = {
        "open": True,
        "commercially_licensed": True,
        "trained_from_scratch": False,
    }
    new_results = tmp_path / "incoming.jsonl"
    new_results.write_text(json.dumps(incoming), encoding="utf-8")
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)
    FakeHfApi.objects["foo_bar/task__test__zeroshot.json"] = predecessor
    reads = 0
    original_read = Path.read_bytes

    def count_read(self: Path) -> bytes:
        """Count exact source captures without counting local semantic checks.

        Args:
            self: File being read.

        Returns:
            Original file bytes.
        """
        nonlocal reads
        if self == path:
            reads += 1
        return original_read(self)

    monkeypatch.setattr(Path, "read_bytes", count_read)
    assert collect_evaluation_results.upload_results_to_hf(new_results_path=new_results)
    assert reads == 1
    history = f"_history/{hashlib.sha256(predecessor).hexdigest()}.bin"
    assert FakeHfApi.objects[history] == predecessor


def _enable_real_collector_publisher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Use the real publisher with isolated in-memory adapters."""
    monkeypatch.setattr(collect_evaluation_results, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        canonical_publication.PublicationLock,
        "from_environment",
        classmethod(lambda cls, bucket="EuroEval/results": MemoryLock()),
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def test_collector_falls_back_to_exact_bytes_when_metadata_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exact predecessor bodies preserve safe metadata transforms on fallback."""
    results_dir = tmp_path / "results"
    record_path = results_dir / "foo_bar" / "task__test__zeroshot.json"
    record_path.parent.mkdir(parents=True)
    original = _ranked_collector_record()
    predecessor = (json.dumps(original, indent=2) + "\n").encode()
    record_path.write_bytes(predecessor)
    incoming_record = json.loads(json.dumps(original))
    incoming_record["model_info"]["additional_details"] = {
        "open": True,
        "commercially_licensed": True,
        "trained_from_scratch": False,
    }
    incoming = tmp_path / "incoming.jsonl"
    incoming.write_text(json.dumps(incoming_record), encoding="utf-8")
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)
    FakeHfApi.objects["foo_bar/task__test__zeroshot.json"] = predecessor
    attempts = 0

    def metadata_failure(
        self: FakeHfApi, bucket_id: str, paths: list[str], *, token: str
    ) -> list[object]:
        nonlocal attempts
        del self, bucket_id, paths, token
        attempts += 1
        if attempts == 1:
            raise RuntimeError("metadata unavailable")
        return []

    monkeypatch.setattr(
        FakeHfApi, "get_bucket_paths_info", metadata_failure, raising=False
    )

    assert collect_evaluation_results.upload_results_to_hf(new_results_path=incoming)
    transformed_local = record_path.read_bytes()
    expected_details = {
        "open": True,
        "commercially_licensed": True,
        "trained_from_scratch": False,
    }
    assert (
        json.loads(transformed_local)["model_info"]["additional_details"]
        == expected_details
    )
    canonical = FakeHfApi.objects["foo_bar/task__test__zeroshot.json"]
    assert json.loads(canonical)["model_info"]["additional_details"] == expected_details
    history = f"_history/{hashlib.sha256(predecessor).hexdigest()}.bin"
    assert FakeHfApi.objects[history] == predecessor


def test_collector_forwards_explicit_choice_proof_through_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit intent reaches both durable prepare and publish with exact bytes."""
    results_dir = tmp_path / "results"
    record_path = results_dir / "foo_bar" / "task__test__zeroshot.json"
    record_path.parent.mkdir(parents=True)
    stored = _ranked_collector_record()
    predecessor = (json.dumps(stored, indent=2) + "\n").encode()
    record_path.write_bytes(predecessor)
    incoming = json.loads(json.dumps(stored))
    incoming["evaluation_results"][0]["score_details"]["score"] = 0.8
    staging = tmp_path / "new_results.jsonl"
    staging.write_text("existing staging\n", encoding="utf-8")
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(collect_evaluation_results, "NEW_RESULTS_PATH", staging)
    monkeypatch.setattr(collect_evaluation_results, "HfApi", FakeHfApi)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    checkpoints: list[dict[str, object]] = []

    class Publisher:
        """Capture the public publisher calls without performing network I/O."""

        def __init__(self, **_kwargs: object) -> None:
            pass

        def prepare(self, **kwargs: object) -> None:
            checkpoints.append(kwargs)

        def publish(self, **kwargs: object) -> None:
            checkpoints.append(kwargs)

        def session(self) -> object:
            return nullcontext(self)

    monkeypatch.setattr(collect_evaluation_results, "CanonicalPublisher", Publisher)
    actual_resolver = collect_evaluation_results.resolve_result_interactively
    monkeypatch.setattr(
        collect_evaluation_results,
        "resolve_result_interactively",
        lambda **kwargs: actual_resolver(
            **kwargs, input_fn=lambda _prompt: "incoming", interactive=True
        ),
    )
    assert collect_evaluation_results.upload_results_to_hf(
        new_results_path=staging, pending_lines=[json.dumps(incoming)]
    )
    prepared = checkpoints[0]
    published = checkpoints[2]
    decision = t.cast(list[ConflictDecision], prepared["decisions"])[0]
    final_bytes = json.dumps(incoming, separators=(",", ":")).encode("utf-8")
    assert decision.evidence.current_sha256 == hashlib.sha256(predecessor).hexdigest()
    assert decision.evidence.candidate_sha256 == hashlib.sha256(final_bytes).hexdigest()
    assert published["decisions"] == prepared["decisions"]


def test_collector_prepare_failure_leaves_local_files_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed durable source checkpoint prevents all collector local writes."""
    results_dir = tmp_path / "results"
    record_path = results_dir / "foo_bar" / "task__test__zeroshot.json"
    record_path.parent.mkdir(parents=True)
    original = _ranked_collector_record()
    original_bytes = (json.dumps(original, indent=2) + "\n").encode()
    record_path.write_bytes(original_bytes)
    stale = results_dir / "stale.jsonl"
    stale.write_bytes(b"keep\n")
    incoming_record = json.loads(json.dumps(original))
    incoming_record["model_info"]["additional_details"] = {
        "open": True,
        "commercially_licensed": True,
        "trained_from_scratch": False,
    }
    incoming = tmp_path / "incoming.jsonl"
    incoming.write_text(json.dumps(incoming_record), encoding="utf-8")
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(
        canonical_publication.CanonicalPublisher,
        "prepare",
        lambda self, records, sources: (_ for _ in ()).throw(RuntimeError("prepare")),
    )

    assert not collect_evaluation_results.upload_results_to_hf(
        new_results_path=incoming
    )
    assert record_path.read_bytes() == original_bytes
    assert stale.read_bytes() == b"keep\n"
    assert FakeHfApi.objects == {}


def test_eof_after_earlier_choice_preserves_original_staging_and_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A later EOF cannot commit earlier choices or rewrite staged input."""
    results_dir = tmp_path / "results"
    first_path = results_dir / "foo_bar" / "task__test__zeroshot.json"
    second_path = results_dir / "foo_bar" / "another__test__zeroshot.json"
    first_path.parent.mkdir(parents=True)
    first = _ranked_collector_record(score=0.2)
    second = _ranked_collector_record(score=0.3)
    second_library = t.cast(dict[str, object], second["eval_library"])
    second_details = t.cast(dict[str, object], second_library["additional_details"])
    second_details["dataset"] = "another"
    first_bytes = (json.dumps(first, indent=2) + "\n").encode()
    second_bytes = (json.dumps(second, indent=2) + "\n").encode()
    first_path.write_bytes(first_bytes)
    second_path.write_bytes(second_bytes)
    staging = tmp_path / "new_results.jsonl"
    staging_bytes = b"original staged data\n"
    staging.write_bytes(staging_bytes)
    incoming_first = _ranked_collector_record(score=0.8)
    incoming_second = json.loads(json.dumps(second))
    incoming_second["evaluation_results"][0]["score_details"]["score"] = 0.9
    answers = iter(["incoming"])
    resolver = collect_evaluation_results.resolve_result_interactively

    def read_answer(_prompt: str) -> str:
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    def choose(
        *,
        current: dict[str, object],
        candidate: dict[str, object],
        current_bytes: bytes | None = None,
        candidate_bytes: bytes | None = None,
    ) -> ConflictDecision:
        return resolver(
            current=current,
            candidate=candidate,
            current_bytes=current_bytes,
            candidate_bytes=candidate_bytes,
            input_fn=read_answer,
            interactive=True,
        )

    monkeypatch.setattr(
        collect_evaluation_results, "resolve_result_interactively", choose
    )
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(collect_evaluation_results, "NEW_RESULTS_PATH", staging)
    lines = [json.dumps(incoming_first), json.dumps(incoming_second)]

    assert not collect_evaluation_results.upload_results_to_hf(
        new_results_path=staging, pending_lines=lines
    )
    assert staging.read_bytes() == staging_bytes
    assert first_path.read_bytes() == first_bytes
    assert second_path.read_bytes() == second_bytes
    assert FakeHfApi.objects == {}


def test_harvested_conflict_abort_preserves_canonical_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An aborted harvested issue batch leaves canonical bytes untouched."""
    results_dir = tmp_path / "results"
    canonical = results_dir / "foo_bar" / "task__test__zeroshot.json"
    canonical.parent.mkdir(parents=True)
    stored = _ranked_collector_record(score=0.2)
    original_bytes = (json.dumps(stored, indent=2) + "\n").encode()
    canonical.write_bytes(original_bytes)
    harvested = _ranked_collector_record(score=0.8)
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(
        collect_evaluation_results,
        "resolve_result_interactively",
        lambda **_kwargs: (_ for _ in ()).throw(
            UnresolvedResultConflictError("aborted")
        ),
    )
    harvest_dir = tmp_path / "harvest"
    harvest_dir.mkdir()
    issue_result = harvest_dir / "issue-result.jsonl"
    issue_result.write_text(json.dumps(harvested), encoding="utf-8")

    assert not collect_evaluation_results.upload_results_to_hf(
        new_results_path=issue_result, pending_lines=[issue_result.read_text()]
    )
    assert canonical.read_bytes() == original_bytes


def test_initial_yes_approves_without_cli_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overview consent leads to a per-model approval prompt."""
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
    staging = Path("new_results.jsonl")
    monkeypatch.setattr(collect_evaluation_results, "NEW_RESULTS_PATH", staging)
    answers = iter(["y", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    command = Mock(side_effect=AssertionError("approval must not invoke CLI"))
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

    def group_submissions_by_model(self, submission_ids: list[str]) -> list[list[str]]:
        """Group this fixture's distinct models without inspecting their records.

        Args:
            submission_ids: Pending submissions.

        Returns:
            One group for each distinct model in this fixture.
        """
        return [[sid] for sid in submission_ids]

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
            model_id=f"model-{submission_id}",
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
    assert command.call_args_list == []
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
    real_show = PreflightReviewer.show

    def display(self: PreflightReviewer, submission_id: str) -> object:
        shown.append(submission_id)
        if command_kind == "show" and submission_id == "first":
            if failure == "exception":
                raise RuntimeError("injected display failure")
            raise RuntimeError("injected display failure")
        return real_show(self, submission_id=submission_id)

    def run(command: list[str]) -> int:
        calls.append(command)
        failed = command[0] == "--reviewer" and command[3] == "first"
        if failed and failure == "exception":
            raise RuntimeError("injected approval failure")
        return 1 if failed else 0

    monkeypatch.setattr(PreflightReviewer, "show", display)
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
        "--quiet",
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


def test_public_upload_accepts_multiple_new_identity_candidates_without_proof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Multiple new identities need no predecessor choice evidence."""
    results_dir = tmp_path / "results"
    first = _ranked_collector_record(score=0.2)
    second = _ranked_collector_record(score=0.8)
    second_library = t.cast(dict[str, object], second["eval_library"])
    second_details = t.cast(dict[str, object], second_library["additional_details"])
    second_details["dataset"] = "another-task"
    incoming = tmp_path / "incoming.jsonl"
    lines = [json.dumps(first), json.dumps(second)]
    incoming.write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert collect_evaluation_results.upload_results_to_hf(new_results_path=incoming)
    assert len([name for name in FakeHfApi.objects if name.endswith(".json")]) == 2


def test_public_upload_reapproves_final_winner_after_multiple_contenders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A later contender cannot displace the approved middle contender."""
    results_dir = tmp_path / "results"
    canonical = results_dir / "foo_bar" / "task__test__zeroshot.json"
    canonical.parent.mkdir(parents=True)
    stored = _ranked_collector_record(score=0.2)
    original_bytes = (json.dumps(stored, indent=2) + "\n").encode()
    canonical.write_bytes(original_bytes)
    contender_b = _ranked_collector_record(score=0.5)
    contender_c = _ranked_collector_record(score=0.8)
    inputs = iter(["incoming", "stored", "incoming"])
    prompted_pairs: list[tuple[object, object]] = []
    approved_decisions: list[ConflictDecision] = []
    resolver = collect_evaluation_results.resolve_result_interactively

    def choose(
        *,
        current: dict[str, object],
        candidate: dict[str, object],
        current_bytes: bytes | None = None,
        candidate_bytes: bytes | None = None,
    ) -> ConflictDecision:
        prompted_pairs.append(
            (current["evaluation_results"], candidate["evaluation_results"])
        )
        decision = resolver(
            current=current,
            candidate=candidate,
            current_bytes=current_bytes,
            candidate_bytes=candidate_bytes,
            input_fn=lambda _prompt: next(inputs),
            interactive=True,
        )
        approved_decisions.append(decision)
        return decision

    monkeypatch.setattr(
        collect_evaluation_results, "resolve_result_interactively", choose
    )
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)
    FakeHfApi.objects["foo_bar/task__test__zeroshot.json"] = original_bytes
    incoming = tmp_path / "incoming.jsonl"
    lines = [json.dumps(contender_b), json.dumps(contender_c)]
    incoming.write_text("\n".join(lines), encoding="utf-8")

    assert collect_evaluation_results.upload_results_to_hf(
        new_results_path=incoming, pending_lines=lines
    )

    assert json.loads(canonical.read_bytes()) == contender_b
    assert prompted_pairs[-1] == (
        stored["evaluation_results"],
        contender_b["evaluation_results"],
    )
    final_bytes = json.dumps(contender_b, separators=(",", ":")).encode()
    assert (
        approved_decisions[-1].evidence.current_sha256
        == hashlib.sha256(original_bytes).hexdigest()
    )
    assert (
        approved_decisions[-1].evidence.candidate_sha256
        == hashlib.sha256(final_bytes).hexdigest()
    )


def test_public_upload_restores_original_when_final_choice_is_stored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Staging, local files, and publication use the same stored final winner."""
    results_dir = tmp_path / "results"
    canonical = results_dir / "foo_bar" / "task__test__zeroshot.json"
    canonical.parent.mkdir(parents=True)
    stored = _ranked_collector_record(score=0.2)
    original_bytes = (json.dumps(stored, indent=2) + "\n").encode()
    canonical.write_bytes(original_bytes)
    contender_b = _ranked_collector_record(score=0.5)
    contender_c = _ranked_collector_record(score=0.8)
    answers = iter(["incoming", "stored", "stored"])
    resolver = collect_evaluation_results.resolve_result_interactively
    monkeypatch.setattr(
        collect_evaluation_results,
        "resolve_result_interactively",
        lambda **kwargs: resolver(
            **kwargs, input_fn=lambda _prompt: next(answers), interactive=True
        ),
    )
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)
    FakeHfApi.objects["foo_bar/task__test__zeroshot.json"] = original_bytes
    staging = tmp_path / "custom-staging.jsonl"
    lines = [json.dumps(contender_b), json.dumps(contender_c)]
    staging.write_text("unmodified staging\n", encoding="utf-8")

    assert collect_evaluation_results.upload_results_to_hf(
        new_results_path=staging, pending_lines=lines
    )

    assert json.loads(canonical.read_bytes()) == stored
    assert json.loads(staging.read_text(encoding="utf-8")) == stored
    assert json.loads(FakeHfApi.objects["foo_bar/task__test__zeroshot.json"]) == stored


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
            "--quiet",
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
            "--quiet",
        ]
    )


def test_staged_resolved_winner_is_not_reloaded_with_rejected_contender(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The public loader sees the winner, not a rejected staged contender."""
    results_dir = tmp_path / "results"
    canonical = results_dir / "foo_bar" / "task__test__zeroshot.json"
    canonical.parent.mkdir(parents=True)
    stored = _ranked_collector_record(score=0.2)
    stored["schema_version"] = 1
    canonical.write_bytes(json.dumps(stored).encode())
    incoming_record = _ranked_collector_record(score=0.8)
    incoming_record["schema_version"] = 1
    staging = tmp_path / "new_results.jsonl"
    staged_line = json.dumps(incoming_record)
    staging.write_text("unmodified staging\n", encoding="utf-8")
    default_staging = tmp_path / "default_new_results.jsonl"
    default_staging.write_text("unrelated manual input\n", encoding="utf-8")
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(collect_evaluation_results, "NEW_RESULTS_PATH", default_staging)
    monkeypatch.setattr(result_loading, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(result_loading, "NEW_RESULTS_PATH", staging)
    monkeypatch.setattr(result_loading, "_sync_results_from_bucket", lambda: None)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)
    resolver = collect_evaluation_results.resolve_result_interactively
    monkeypatch.setattr(
        collect_evaluation_results,
        "resolve_result_interactively",
        lambda **kwargs: resolver(
            **kwargs, input_fn=lambda _prompt: "stored", interactive=True
        ),
    )
    predecessor = canonical.read_bytes()
    FakeHfApi.objects["foo_bar/task__test__zeroshot.json"] = predecessor

    assert collect_evaluation_results.upload_results_to_hf(
        new_results_path=staging, pending_lines=[staged_line]
    )
    assert default_staging.read_text(encoding="utf-8") == "unrelated manual input\n"
    assert (
        json.loads(staging.read_text())["evaluation_results"]
        == stored["evaluation_results"]
    )
    result_loading.load_raw_results.cache_clear()
    records = result_loading.load_raw_results()
    result_loading.load_raw_results.cache_clear()

    assert len(records) == 1
    assert records[0]["evaluation_results"] == stored["evaluation_results"]
    assert not staging.exists()


@pytest.mark.parametrize(
    ("first_version", "second_version", "first_timestamp", "second_timestamp"),
    [("1.0.0", "2.0.0", 2, 1), ("1.0.0", "1.0.0", 1, 2)],
)
def test_upload_results_to_hf_alias_reconciliation_is_order_independent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    first_version: str,
    second_version: str,
    first_timestamp: int,
    second_timestamp: int,
) -> None:
    """The later evaluation timestamp wins despite version/retrieval aliases."""
    results_dir = tmp_path / "results"
    directory = results_dir / "foo_bar"
    directory.mkdir(parents=True)
    first = _ranked_collector_record(
        version=first_version, timestamp=first_timestamp, score=0.1
    )
    first["eval_timestamp"] = "2024-01-01T00:00:00Z"
    second = _ranked_collector_record(
        version=second_version, timestamp=second_timestamp, score=0.9
    )
    second["eval_timestamp"] = "2024-01-02T00:00:00Z"
    (directory / "a.json").write_text(json.dumps(first), encoding="utf-8")
    (directory / "z.json").write_text(json.dumps(second), encoding="utf-8")
    incoming = tmp_path / "incoming.jsonl"
    incoming.write_text("", encoding="utf-8")
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert collect_evaluation_results.upload_results_to_hf(new_results_path=incoming)

    canonical = directory / "task__test__zeroshot.json"
    published = json.loads(canonical.read_text(encoding="utf-8"))
    assert published["evaluation_results"] == second["evaluation_results"]


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


def test_upload_results_to_hf_engine_preference_beats_canonical_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Canonical filenames do not override an engine-policy winner."""
    results_dir = tmp_path / "results"
    directory = results_dir / "foo_bar"
    directory.mkdir(parents=True)
    alias = _ranked_collector_record(score=0.9)
    alias_model_info = t.cast(dict[str, object], alias["model_info"])
    alias_model_info["inference_engine"] = {"name": "vLLM"}
    canonical = _ranked_collector_record(score=0.1)
    canonical_model_info = t.cast(dict[str, object], canonical["model_info"])
    canonical_model_info["inference_engine"] = {"name": "LiteLLM"}
    (directory / "a.json").write_text(json.dumps(alias), encoding="utf-8")
    canonical_path = directory / "task__test__zeroshot.json"
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    incoming = tmp_path / "incoming.jsonl"
    incoming.write_text("", encoding="utf-8")
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert collect_evaluation_results.upload_results_to_hf(new_results_path=incoming)

    assert json.loads(canonical_path.read_text(encoding="utf-8")) == alias
    assert json.loads(FakeHfApi.objects["foo_bar/task__test__zeroshot.json"]) == alias


def test_upload_results_to_hf_equal_rank_envelope_aliases_are_allowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Known HTML anchor envelope differences do not change evaluation data."""
    results_dir = tmp_path / "results"
    path = results_dir / "foo_bar" / "task__test__zeroshot.json"
    path.parent.mkdir(parents=True)
    original = _ranked_collector_record()
    predecessor = (json.dumps(original, indent=2) + "\n").encode()
    path.write_bytes(predecessor)
    enriched = json.loads(json.dumps(original))
    enriched.update(
        {"id": "anchor", "name": "result", "eval_id": "old", "eval_timestamp": 1}
    )
    incoming = tmp_path / "incoming.jsonl"
    incoming.write_text(json.dumps(enriched), encoding="utf-8")
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)
    FakeHfApi.objects["foo_bar/task__test__zeroshot.json"] = predecessor

    assert collect_evaluation_results.upload_results_to_hf(new_results_path=incoming)

    assert path.read_bytes() == predecessor


@pytest.mark.parametrize(
    ("first_raw", "second_raw"),
    [
        ([1, 2], [1, 3]),
        ([1, 2], "[1, 3]"),
        ({"iteration": 1, "score": 0.2}, '{"score": 0.8, "iteration": 1}'),
    ],
)
def test_upload_results_to_hf_equal_rank_raw_results_conflict_is_preflighted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    first_raw: object,
    second_raw: object,
) -> None:
    """Different iteration results conflict even when aggregates match."""
    results_dir = tmp_path / "results"
    directory = results_dir / "foo_bar"
    directory.mkdir(parents=True)
    first = _ranked_collector_record(score=0.5)
    second = _ranked_collector_record(score=0.5)
    first_library = t.cast(dict[str, object], first["eval_library"])
    second_library = t.cast(dict[str, object], second["eval_library"])
    first_details = t.cast(dict[str, object], first_library["additional_details"])
    second_details = t.cast(dict[str, object], second_library["additional_details"])
    first_details["raw_results"] = first_raw
    second_details["raw_results"] = second_raw
    paths = [directory / "a.json", directory / "z.json"]
    paths[0].write_text(json.dumps(first, indent=2), encoding="utf-8")
    paths[1].write_text(json.dumps(second, indent=2), encoding="utf-8")
    before = {path: path.read_bytes() for path in paths}
    incoming = tmp_path / "incoming.jsonl"
    incoming.write_text("", encoding="utf-8")
    incoming_bytes = incoming.read_bytes()
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert not collect_evaluation_results.upload_results_to_hf(
        new_results_path=incoming
    )

    assert incoming.read_bytes() == incoming_bytes
    assert {path: path.read_bytes() for path in paths} == before
    assert FakeHfApi.objects == {}


@pytest.mark.parametrize("reverse", [False, True])
def test_upload_results_to_hf_equal_rank_score_conflict_is_preflighted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reverse: bool
) -> None:
    """Conflicting evaluation payloads fail before local or remote mutation."""
    results_dir = tmp_path / "results"
    directory = results_dir / "foo_bar"
    directory.mkdir(parents=True)
    first = _ranked_collector_record(score=0.2)
    second = _ranked_collector_record(score=0.8)
    records = [first, second]
    if reverse:
        records.reverse()
    paths = [directory / "a.json", directory / "z.json"]
    for path, record in zip(paths, records, strict=True):
        path.write_bytes((json.dumps(record, indent=2) + "\n").encode())
    before = {path: path.read_bytes() for path in paths}
    incoming = tmp_path / "incoming.jsonl"
    incoming.write_text("", encoding="utf-8")
    incoming_bytes = incoming.read_bytes()
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert not collect_evaluation_results.upload_results_to_hf(
        new_results_path=incoming
    )

    assert incoming.read_bytes() == incoming_bytes
    assert {path: path.read_bytes() for path in paths} == before
    assert FakeHfApi.objects == {}


@pytest.mark.parametrize(
    "canary_dataset", ["contamination-canary", "contamination-canary-private-variant"]
)
def test_upload_results_to_hf_excludes_local_canaries_from_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, canary_dataset: str
) -> None:
    """Local canary files are neither published nor modified."""
    results_dir = tmp_path / "results"
    model_dir = results_dir / "foo_bar"
    model_dir.mkdir(parents=True)
    ordinary = _collector_test_record(dataset="ordinary")
    canary = _collector_test_record(dataset=canary_dataset)
    ordinary_path = model_dir / "ordinary__test__zeroshot.json"
    canary_path = model_dir / "canary__test__zeroshot.json"
    ordinary_path.write_text(json.dumps(ordinary), encoding="utf-8")
    private_bytes = (json.dumps(canary, indent=2) + "\n").encode()
    canary_path.write_bytes(private_bytes)
    incoming_path = tmp_path / "incoming.jsonl"
    incoming_path.write_text(json.dumps(ordinary), encoding="utf-8")
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(collect_evaluation_results, "HfApi", FakeHfApi)

    assert (
        collect_evaluation_results.upload_results_to_hf(new_results_path=incoming_path)
        is True
    )

    published_datasets = {
        json.loads(content)["eval_library"]["additional_details"]["dataset"]
        for content in FakeHfApi.objects.values()
    }
    assert "ordinary" in published_datasets
    assert canary_dataset not in published_datasets
    assert canary_path.read_bytes() == private_bytes


def _collector_test_record(dataset: str) -> dict[str, object]:
    """Build a minimal result record for collection upload tests.

    Args:
        dataset:
            Dataset name for the record.

    Returns:
        A minimally valid result record.
    """
    return {
        "model_info": {"id": "foo/bar"},
        "eval_library": {
            "additional_details": {
                "dataset": dataset,
                "few_shot": False,
                "validation_split": False,
            },
            "version": "1.0.0",
        },
        "retrieved_timestamp": 1704067200,
    }


def test_upload_results_to_hf_manual_alias_choice_beats_canonical_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit alias winner survives canonical-path reconciliation."""
    results_dir = tmp_path / "results"
    directory = results_dir / "foo_bar"
    directory.mkdir(parents=True)
    alias = _ranked_collector_record(score=0.9)
    canonical = _ranked_collector_record(score=0.1)
    (directory / "a.json").write_text(json.dumps(alias), encoding="utf-8")
    canonical_path = directory / "task__test__zeroshot.json"
    canonical_bytes = json.dumps(canonical).encode("utf-8")
    canonical_path.write_bytes(canonical_bytes)
    answers = iter(["stored", "incoming"])
    resolver = collect_evaluation_results.resolve_result_interactively
    monkeypatch.setattr(
        collect_evaluation_results,
        "resolve_result_interactively",
        lambda **kwargs: resolver(
            **kwargs, input_fn=lambda _prompt: next(answers), interactive=True
        ),
    )
    incoming = tmp_path / "incoming.jsonl"
    incoming.write_text("", encoding="utf-8")
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)
    FakeHfApi.objects["foo_bar/task__test__zeroshot.json"] = canonical_bytes

    assert collect_evaluation_results.upload_results_to_hf(new_results_path=incoming)

    assert json.loads(canonical_path.read_text(encoding="utf-8")) == alias
    assert json.loads(FakeHfApi.objects["foo_bar/task__test__zeroshot.json"]) == alias
    history_path = f"_history/{hashlib.sha256(canonical_bytes).hexdigest()}.bin"
    assert FakeHfApi.objects[history_path] == canonical_bytes


def test_upload_results_to_hf_metadata_enrichment_uses_original_source_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Equal-ranked metadata enrichment publishes against exact predecessor bytes."""
    results_dir = tmp_path / "results"
    path = results_dir / "foo_bar" / "task__test__zeroshot.json"
    path.parent.mkdir(parents=True)
    original = _ranked_collector_record()
    predecessor = (json.dumps(original, indent=2) + "\n").encode()
    path.write_bytes(predecessor)
    enriched = json.loads(json.dumps(original))
    enriched["model_info"]["additional_details"] = {
        "open": True,
        "commercially_licensed": True,
        "trained_from_scratch": False,
    }
    incoming = tmp_path / "incoming.jsonl"
    incoming.write_text(json.dumps(enriched), encoding="utf-8")
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)
    FakeHfApi.objects["foo_bar/task__test__zeroshot.json"] = predecessor

    assert collect_evaluation_results.upload_results_to_hf(new_results_path=incoming)

    assert path.read_bytes() == json.dumps(enriched, separators=(",", ":")).encode()
    history_suffix = f"_history/{hashlib.sha256(predecessor).hexdigest()}.bin"
    assert any(
        name.endswith(history_suffix) and content == predecessor
        for name, content in FakeHfApi.objects.items()
    )


def test_upload_results_to_hf_raw_results_json_encodings_compare_semantically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dict, JSON string, and object-key order variations remain equivalent."""
    results_dir = tmp_path / "results"
    directory = results_dir / "foo_bar"
    directory.mkdir(parents=True)
    canonical = _ranked_collector_record(score=0.5)
    alias = _ranked_collector_record(score=0.5)
    canonical_details = t.cast(
        dict[str, object],
        t.cast(dict[str, object], canonical["eval_library"])["additional_details"],
    )
    alias_details = t.cast(
        dict[str, object],
        t.cast(dict[str, object], alias["eval_library"])["additional_details"],
    )
    canonical_details["raw_results"] = {"iteration": 1, "score": 0.2}
    alias_details["raw_results"] = '{"score": 0.2, "iteration": 1}'
    canonical_path = directory / "task__test__zeroshot.json"
    canonical_bytes = json.dumps(canonical, indent=2).encode()
    canonical_path.write_bytes(canonical_bytes)
    (directory / "alias.json").write_text(json.dumps(alias), encoding="utf-8")
    incoming = tmp_path / "incoming.jsonl"
    incoming.write_text("", encoding="utf-8")
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)
    FakeHfApi.objects["foo_bar/task__test__zeroshot.json"] = canonical_bytes

    assert collect_evaluation_results.upload_results_to_hf(new_results_path=incoming)

    assert json.loads(canonical_path.read_bytes()) == canonical
    assert canonical_path.read_bytes() == canonical_bytes
    assert FakeHfApi.objects["foo_bar/task__test__zeroshot.json"] == canonical_bytes


def test_upload_results_to_hf_reconciles_legacy_alias_to_canonical_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Equal-ranked aliases choose the canonical record regardless of path order."""
    results_dir = tmp_path / "results"
    record = _ranked_collector_record()
    canonical = results_dir / "foo_bar" / "task__test__zeroshot.json"
    legacy = results_dir / "aaa_legacy" / "anchored.json"
    canonical.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    canonical_record = {**record, "model_info": {"id": "foo/bar"}}
    canonical_bytes = json.dumps(canonical_record).encode()
    canonical.write_bytes(canonical_bytes)
    legacy.write_text(json.dumps(canonical_record), encoding="utf-8")
    incoming = tmp_path / "incoming.jsonl"
    incoming.write_text("", encoding="utf-8")
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert collect_evaluation_results.upload_results_to_hf(new_results_path=incoming)

    published = [
        content for name, content in FakeHfApi.objects.items() if name.endswith(".json")
    ]
    assert len(published) == 1
    assert json.loads(published[0]) == canonical_record
    assert canonical.read_bytes() == canonical_bytes


def test_upload_results_to_hf_retry_without_changes_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A no-change retry accepts the already-published canonical record."""
    results_dir = tmp_path / "results"
    directory = results_dir / "foo_bar"
    directory.mkdir(parents=True)
    record = _ranked_collector_record()
    path = directory / "task__test__zeroshot.json"
    path.write_bytes(json.dumps(record).encode())
    incoming = tmp_path / "incoming.jsonl"
    incoming.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert collect_evaluation_results.upload_results_to_hf(new_results_path=incoming)
    first_remote = dict(FakeHfApi.objects)
    assert collect_evaluation_results.upload_results_to_hf(new_results_path=incoming)
    assert FakeHfApi.objects == first_remote


def test_upload_results_to_hf_same_identity_keeps_newer_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A version-only difference is an identical no-op under conflict policy.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Existing and incoming records differ only in an ignored version alias.
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

    # The changed version does not prove a newer evaluation.
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

    # The stored record remains canonical.
    content = json.loads(existing_record_file.read_text(encoding="utf-8"))
    assert content["eval_library"]["version"] == "1.0.0"


def test_upload_results_to_hf_same_identity_uses_timestamp_tiebreaker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A retrieval timestamp difference alone is an identical no-op.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
        tmp_path:
            Temporary directory for files created by the test.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Stored result has an earlier retrieval timestamp.
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

    # Retrieval time is not evidence of a newer evaluation.
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

    # The stored record remains canonical.
    content = json.loads(existing_record_file.read_text(encoding="utf-8"))
    assert content["retrieved_timestamp"] == 1704067200


def test_upload_results_to_hf_skips_incoming_canary_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Incoming canary JSONL lines do not replace or publish local results."""
    results_dir = tmp_path / "results"
    model_dir = results_dir / "foo_bar"
    model_dir.mkdir(parents=True)
    ordinary = _collector_test_record(dataset="ordinary")
    canary = _collector_test_record(dataset="contamination-canary-worker")
    ordinary_path = model_dir / "ordinary__test__zeroshot.json"
    ordinary_bytes = json.dumps(ordinary).encode()
    ordinary_path.write_bytes(ordinary_bytes)
    incoming_path = tmp_path / "incoming.jsonl"
    incoming_path.write_text(json.dumps(canary), encoding="utf-8")
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(collect_evaluation_results, "HfApi", FakeHfApi)

    assert (
        collect_evaluation_results.upload_results_to_hf(new_results_path=incoming_path)
        is True
    )

    assert ordinary_path.read_bytes() == ordinary_bytes
    assert {
        json.loads(content)["eval_library"]["additional_details"]["dataset"]
        for content in FakeHfApi.objects.values()
    } == {"ordinary"}


def test_upload_results_to_hf_unsupported_equal_rank_change_is_preflighted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unsupported equal-rank settings preserve every local artifact."""
    results_dir = tmp_path / "results"
    path = results_dir / "foo_bar" / "task__test__zeroshot.json"
    path.parent.mkdir(parents=True)
    original = _ranked_collector_record()
    predecessor = (json.dumps(original, indent=2) + "\n").encode()
    path.write_bytes(predecessor)
    stale = results_dir / "stale.jsonl"
    stale_bytes = b"stale\n"
    stale.write_bytes(stale_bytes)
    incoming_record = json.loads(json.dumps(original))
    incoming_record["use_bits_per_character"] = True
    incoming = tmp_path / "incoming.jsonl"
    incoming_bytes = (json.dumps(incoming_record) + "\n").encode()
    incoming.write_bytes(incoming_bytes)
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    _enable_real_collector_publisher(monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert not collect_evaluation_results.upload_results_to_hf(
        new_results_path=incoming
    )

    assert path.read_bytes() == predecessor
    assert stale.read_bytes() == stale_bytes
    assert incoming.read_bytes() == incoming_bytes
    assert FakeHfApi.objects == {}


def test_upload_results_to_hf_with_only_local_canaries_is_noop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An all-private local inventory remains untouched and unpublished."""
    results_dir = tmp_path / "results"
    model_dir = results_dir / "foo_bar"
    model_dir.mkdir(parents=True)
    private_path = model_dir / "private.json"
    private_bytes = (
        json.dumps(_collector_test_record(dataset="contamination-canary"), indent=2)
        + "\n"
    ).encode()
    private_path.write_bytes(private_bytes)
    incoming_path = tmp_path / "incoming.jsonl"
    incoming_path.write_text(
        json.dumps(_collector_test_record(dataset="contamination-canary")),
        encoding="utf-8",
    )
    monkeypatch.setattr(collect_evaluation_results, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(collect_evaluation_results, "HfApi", FakeHfApi)

    assert (
        collect_evaluation_results.upload_results_to_hf(new_results_path=incoming_path)
        is False
    )

    assert private_path.read_bytes() == private_bytes
    assert FakeHfApi.objects == {}
