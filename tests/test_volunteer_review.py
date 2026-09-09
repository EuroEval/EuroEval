"""Tests for durable volunteer-result review and promotion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from euroeval_worker.review import BucketStore, ReviewError, VolunteerReviewer

STAGING = "EuroEval/private-volunteer-staging"
RESULTS = "EuroEval/results"
SUBMISSION = "submission-one"


class FakeHfApi:
    """In-memory subset of the HfApi bucket interface."""

    def __init__(self) -> None:
        """Initialise empty bucket contents and failure controls."""
        self.files: dict[tuple[str, str], bytes] = {}
        self.uploads: list[tuple[str, str]] = []
        self.fail_result_upload_number: int | None = None
        self.result_uploads = 0

    def bucket_info(self, bucket: str, token: str) -> SimpleNamespace:
        """Return private metadata."""
        return SimpleNamespace(private=True)

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
            if stored_bucket == bucket and path.startswith(prefix)
        ]

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

    @staticmethod
    def _entry(path: str, content: bytes) -> SimpleNamespace:
        return SimpleNamespace(
            type="file",
            path=path,
            size=len(content),
            xet_hash=hashlib.sha256(content).hexdigest(),
        )


def test_corrupted_staged_bytes_are_rejected() -> None:
    """Exact digest validation rejects altered staged content."""
    api, reviewer, _ = _reviewer()
    path = _result_paths(api)[0]
    api.files[(STAGING, path)] += b" "

    with pytest.raises(ReviewError, match="SHA256"):
        reviewer.show(SUBMISSION)


def test_manifest_scope_mismatch_is_rejected() -> None:
    """Expected and actual canonical identities must match."""
    api, reviewer, _ = _reviewer()
    manifest = _manifest(api)
    manifest["expected_scope"]["identity_suffixes"] = ['["other",false,false]']
    _store_manifest(api, manifest)

    with pytest.raises(ReviewError, match="scope differs"):
        reviewer.show(SUBMISSION)


def test_canonical_collision_prevents_decision_and_broker() -> None:
    """Different canonical bytes block all terminal side effects."""
    api, reviewer, broker_calls = _reviewer()
    report = reviewer.show(SUBMISSION)
    api.files[(RESULTS, report.records[0].canonical_path)] = b"different"

    with pytest.raises(ReviewError, match="collision"):
        reviewer.decide(SUBMISSION, "accepted", "maintainer")

    assert broker_calls == []
    assert (STAGING, f"volunteer/decisions/{SUBMISSION}.json") not in api.files


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
    reviewer.decide(SUBMISSION, "rejected", "maintainer")

    assert api.files[(STAGING, f"volunteer/decisions/{SUBMISSION}.json")] == decision
    assert not [key for key in api.files if key[0] == RESULTS]
    assert broker_calls == [(12, SUBMISSION, "rejected"), (12, SUBMISSION, "rejected")]
    with pytest.raises(ReviewError, match="opposite"):
        reviewer.decide(SUBMISSION, "accepted", "maintainer")


def _reviewer(
    record_count: int = 1,
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
    store = BucketStore(api=api, token="token", staging_bucket=STAGING)
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


def _manifest(api: FakeHfApi) -> dict[str, object]:
    return json.loads(api.files[(STAGING, f"volunteer/manifests/{SUBMISSION}.json")])


def _result_paths(api: FakeHfApi) -> list[str]:
    return [
        path for bucket, path in api.files if bucket == STAGING and "/results/" in path
    ]
