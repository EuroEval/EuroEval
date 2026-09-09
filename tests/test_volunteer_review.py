"""Tests for durable volunteer-result review and promotion."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import threading
import typing as t
from pathlib import Path
from types import SimpleNamespace

import pytest

from euroeval_worker.review import (
    BrokerReservationResult,
    BucketApi,
    BucketStore,
    ReviewError,
    VolunteerReviewer,
)

STAGING = "EuroEval/private-volunteer-staging"
RESULTS = "EuroEval/results"
SUBMISSION = "submission-one"


def test_acceptance_renews_before_each_upload() -> None:
    """Long uploads renew the accepted reservation before every write."""
    renewals: list[list[dict[str, str]]] = []
    _, reviewer, _ = _reviewer(
        record_count=2,
        reserver=lambda issue, submission, outcome, reviewer, records: (
            BrokerReservationResult(
                token="reservation",
                decision_reviewer="maintainer",
                decision_created_at="2026-09-06T12:00:00Z",
            )
        ),
        renewer=lambda issue, submission, outcome, token, records: (
            renewals.append(records) or token
        ),
    )

    reviewer.decide(SUBMISSION, "accepted", "maintainer")

    assert len(renewals) == 3
    assert all(record["canonical_path"] for record in renewals[0])


class FakeHfApi:
    """In-memory subset of the HfApi bucket interface."""

    def __init__(self) -> None:
        """Initialise empty bucket contents and failure controls."""
        self.files: dict[tuple[str, str], bytes] = {}
        self.uploads: list[tuple[str, str]] = []
        self.uploaded_content: list[tuple[str, str, bytes]] = []
        self.fail_result_upload_number: int | None = None
        self.result_uploads = 0

    def batch_bucket_files(
        self, bucket_id: str, add: list[tuple[bytes, str]], token: str
    ) -> None:
        """Store exact object bytes, optionally injecting a partial failure.

        Raises:
            RuntimeError:
                When the configured interruption point is reached.
        """
        for content, path in add:
            if bucket_id == RESULTS:
                self.result_uploads += 1
                if self.result_uploads == self.fail_result_upload_number:
                    raise RuntimeError("injected upload interruption")
            self.files[(bucket_id, path)] = content
            self.uploads.append((bucket_id, path))
            self.uploaded_content.append((bucket_id, path, content))

    def bucket_info(self, bucket: str, token: str) -> SimpleNamespace:
        """Return private metadata."""
        return SimpleNamespace(private=True)

    def download_bucket_files(
        self,
        bucket: str,
        files: list[tuple[str, Path]],
        raise_on_missing_files: bool,
        token: str,
    ) -> None:
        """Write requested objects to local paths.

        Raises:
            FileNotFoundError:
                If a requested object is absent.
        """
        for remote, local in files:
            content = self.files.get((bucket, remote))
            if content is None:
                if raise_on_missing_files:
                    raise FileNotFoundError(remote)
                continue
            local.write_bytes(content)

    def get_bucket_paths_info(
        self, bucket: str, paths: list[str], token: str
    ) -> list[SimpleNamespace]:
        """Return metadata for existing paths."""
        return [
            self._entry(path, self.files[(bucket, path)])
            for path in paths
            if (bucket, path) in self.files
        ]

    @staticmethod
    def _entry(path: str, content: bytes) -> SimpleNamespace:
        return SimpleNamespace(
            type="file",
            path=path,
            size=len(content),
            xet_hash=hashlib.sha256(content).hexdigest(),
        )

    def list_bucket_tree(
        self, bucket: str, prefix: str, recursive: bool, token: str
    ) -> list[SimpleNamespace]:
        """List matching objects.

        Returns:
            Matching in-memory file metadata.
        """
        return [
            self._entry(path, content)
            for (stored_bucket, path), content in self.files.items()
            if stored_bucket == bucket
            and path.startswith(prefix)
            and (recursive or "/" not in path.removeprefix(f"{prefix}/"))
        ]


def _reviewer(
    record_count: int = 1,
    reserver: (
        t.Callable[[int, str, str, str, list[dict[str, str]]], BrokerReservationResult]
        | None
    ) = None,
    renewer: t.Callable[[int, str, str, str, list[dict[str, str]]], str] | None = None,
) -> tuple[FakeHfApi, VolunteerReviewer, list[tuple[int, str, str]]]:
    api = FakeHfApi()
    records = [_record(dataset=f"dataset-{index}") for index in range(record_count)]
    entries: list[dict[str, object]] = []
    suffixes: list[str] = []
    for index, record in enumerate(records):
        content = json.dumps(record, ensure_ascii=False).encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        path = f"volunteer/submissions/{SUBMISSION}/results/{digest}.json"
        api.files[(STAGING, path)] = content
        identity = ["org/model", f"dataset-{index}", False, False]
        entries.append(
            {
                "digest": digest,
                "identity": json.dumps(identity, separators=(",", ":")),
                "path": path,
                "warnings": [],
            }
        )
        suffixes.append(json.dumps(identity[1:], separators=(",", ":")))
    manifest: dict[str, object] = {
        "protocol_version": "volunteer-worker/v1",
        "submission_id": SUBMISSION,
        "issue_number": 12,
        "verified_contributor": "alice",
        "model": {"id": "org/model", "revision": "deadbeef"},
        "language": "da",
        "euroeval_version": "18.0.0.dev0",
        "model_profile": "llama",
        "worker_version": "18.0.0",
        "image_digest": "sha256:image",
        "created_at": "2026-09-06T10:00:00Z",
        "expected_scope": {
            "policy_version": "volunteer-scope/18.0.0.dev0",
            "language_group": "da",
            "identity_suffixes": suffixes,
            "count": record_count,
            "warnings": [],
        },
        "results": entries,
        "automated_checks": {
            "result_count": record_count,
            "identities_unique": True,
            "failed_instances": 0,
            "warnings": [],
        },
    }
    _store_manifest(api, manifest)
    calls: list[tuple[int, str, str]] = []
    store = BucketStore(
        api=t.cast(BucketApi, api), token="token", staging_bucket=STAGING
    )
    scope_policy: dict[str, object] = {
        "policy_version": "volunteer-scope/18.0.0.dev0",
        "policies": [
            {
                "euroeval_version": "18.0.0.dev0",
                "model_profile": "llama",
                "language": "da",
                "language_group": "da",
                "identity_suffixes": suffixes,
                "count": record_count,
                "warnings": [],
            }
        ],
    }
    reviewer = VolunteerReviewer(
        store=store,
        results_bucket=RESULTS,
        promoter=lambda issue, submission, outcome: calls.append(
            (issue, submission, outcome)
        ),
        reserver=reserver,
        renewer=renewer,
        scope_policy=scope_policy,
    )
    return api, reviewer, calls


def _record(dataset: str) -> dict[str, object]:
    return {
        "schema_version": "0.3.0",
        "model_info": {
            "id": "org/model",
            "revision": "deadbeef",
            "additional_details": {
                "commercially_licensed": True,
                "open": True,
                "trained_from_scratch": False,
            },
        },
        "eval_library": {
            "name": "euroeval",
            "version": "18.0.0.dev4",
            "additional_details": {
                "dataset": dataset,
                "task": "classification",
                "language": "da",
                "languages": '["da"]',
                "raw_results": "[]",
                "few_shot": False,
                "validation_split": False,
                "num_failed_instances": 0,
            },
        },
        "evaluation_results": [
            {
                "evaluation_name": "accuracy",
                "source_data": {"dataset_name": dataset},
                "metric_config": {
                    "lower_is_better": False,
                    "min_score": 0,
                    "max_score": 100,
                },
                "score_details": {"score": 80},
            }
        ],
    }


def _store_manifest(api: FakeHfApi, manifest: dict[str, object]) -> None:
    api.files[(STAGING, f"volunteer/manifests/{SUBMISSION}.json")] = json.dumps(
        manifest
    ).encode("utf-8")


def test_canonical_collision_prevents_decision_and_broker() -> None:
    """Different canonical bytes block all terminal side effects."""
    api, reviewer, broker_calls = _reviewer()
    report = reviewer.show(SUBMISSION)
    api.files[(RESULTS, report.records[0].canonical_path)] = b"different"

    with pytest.raises(ReviewError, match="collision"):
        reviewer.decide(SUBMISSION, "accepted", "maintainer")

    assert broker_calls == []
    assert (STAGING, f"volunteer/decisions/{SUBMISSION}.json") not in api.files


def test_concurrent_decisions_use_first_server_metadata() -> None:
    """Concurrent reviewers use one server-bound decision byte sequence."""
    reservation_order: list[str] = []
    bound: list[BrokerReservationResult] = []
    lock = threading.Lock()
    ready = threading.Barrier(2)
    server_times = {"alice": "2026-09-06T12:00:00Z", "bob": "2026-09-06T12:01:00Z"}

    def reserve(
        issue: int,
        submission: str,
        outcome: str,
        reviewer: str,
        records: list[dict[str, str]],
    ) -> BrokerReservationResult:
        with lock:
            reservation_order.append(reviewer)
            if not bound:
                bound.append(
                    BrokerReservationResult(
                        token="reservation",
                        decision_reviewer=reviewer,
                        decision_created_at=server_times[reviewer],
                    )
                )
        ready.wait()
        return bound[0]

    api, reviewer, broker_calls = _reviewer(reserver=reserve)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(reviewer.decide, SUBMISSION, "rejected", identity)
            for identity in ("alice", "bob")
        ]
        for future in futures:
            future.result()

    decision = api.files[(STAGING, f"volunteer/decisions/{SUBMISSION}.json")]
    decision_object = json.loads(decision)
    first_reviewer = reservation_order[0]
    assert decision_object["reviewer"] == first_reviewer
    assert decision_object["decided_at"] == server_times[first_reviewer]
    decision_path = f"volunteer/decisions/{SUBMISSION}.json"
    decision_writes = [
        content
        for bucket, path, content in api.uploaded_content
        if bucket == STAGING and path == decision_path
    ]
    assert len(decision_writes) == 2
    assert len(set(decision_writes)) == 1
    assert len(broker_calls) == 2


def test_corrupted_staged_bytes_are_rejected() -> None:
    """Exact digest validation rejects altered staged content."""
    api, reviewer, _ = _reviewer()
    path = _result_paths(api)[0]
    api.files[(STAGING, path)] += b" "

    with pytest.raises(ReviewError, match="SHA256"):
        reviewer.show(SUBMISSION)


def _result_paths(api: FakeHfApi) -> list[str]:
    return [
        path for bucket, path in api.files if bucket == STAGING and "/results/" in path
    ]


def test_manifest_scope_mismatch_is_rejected() -> None:
    """Expected and actual canonical identities must match."""
    api, reviewer, _ = _reviewer()
    manifest = _manifest(api)
    expected_scope = t.cast(dict[str, object], manifest["expected_scope"])
    expected_scope["identity_suffixes"] = ['["other",false,false]']
    _store_manifest(api, manifest)

    with pytest.raises(ReviewError, match="scope differs"):
        reviewer.show(SUBMISSION)


def _manifest(api: FakeHfApi) -> dict[str, object]:
    return json.loads(api.files[(STAGING, f"volunteer/manifests/{SUBMISSION}.json")])


def test_partial_approve_resumes_and_is_idempotent() -> None:
    """Approval resumes partial uploads and preserves its decision artifact."""
    api, reviewer, broker_calls = _reviewer(record_count=2)
    api.fail_result_upload_number = 2

    with pytest.raises(RuntimeError, match="interruption"):
        reviewer.decide(SUBMISSION, "accepted", "maintainer")
    assert len([key for key in api.files if key[0] == RESULTS]) == 1

    api.fail_result_upload_number = None
    reviewer.decide(SUBMISSION, "accepted", "maintainer")
    decision = api.files[(STAGING, f"volunteer/decisions/{SUBMISSION}.json")]
    reviewer.decide(SUBMISSION, "accepted", "another-reviewer")

    assert api.files[(STAGING, f"volunteer/decisions/{SUBMISSION}.json")] == decision
    assert len([key for key in api.files if key[0] == RESULTS]) == 2
    assert broker_calls == [(12, SUBMISSION, "accepted"), (12, SUBMISSION, "accepted")]


def test_reject_is_idempotent_and_opposite_decision_fails() -> None:
    """Rejection retries safely while the opposite outcome is forbidden."""
    api, reviewer, broker_calls = _reviewer()
    reviewer.decide(SUBMISSION, "rejected", "maintainer", ["implausible scores"])
    decision = api.files[(STAGING, f"volunteer/decisions/{SUBMISSION}.json")]
    reviewer.decide(SUBMISSION, "rejected", "maintainer", ["implausible scores"])

    assert api.files[(STAGING, f"volunteer/decisions/{SUBMISSION}.json")] == decision
    assert not [key for key in api.files if key[0] == RESULTS]
    assert broker_calls == [(12, SUBMISSION, "rejected"), (12, SUBMISSION, "rejected")]
    with pytest.raises(ReviewError, match="opposite"):
        reviewer.decide(SUBMISSION, "accepted", "maintainer")


def test_resume_uses_the_first_bound_decision_metadata() -> None:
    """A retry by another reviewer preserves the first decision metadata."""
    bound = BrokerReservationResult(
        token="reservation",
        decision_reviewer="alice",
        decision_created_at="2026-09-06T12:00:00Z",
    )
    reservation_reviewers: list[str] = []

    def reserve(
        issue: int,
        submission: str,
        outcome: str,
        reviewer: str,
        records: list[dict[str, str]],
    ) -> BrokerReservationResult:
        reservation_reviewers.append(reviewer)
        return bound

    api, reviewer, _ = _reviewer(reserver=reserve)
    reviewer.decide(SUBMISSION, "rejected", "alice")
    decision = api.files[(STAGING, f"volunteer/decisions/{SUBMISSION}.json")]
    reviewer.decide(SUBMISSION, "rejected", "bob")

    assert api.files[(STAGING, f"volunteer/decisions/{SUBMISSION}.json")] == decision
    assert reservation_reviewers == ["alice", "bob"]
