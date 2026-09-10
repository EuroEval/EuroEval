"""Tests for durable volunteer-result review and promotion."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import threading
import typing as t
import urllib.request
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
from euroeval_worker.review_transaction import renew_with_broker

STAGING = "EuroEval/private-volunteer-staging"
RESULTS = "EuroEval/results"
SUBMISSION = "submission-one"


def test_acceptance_renews_before_each_upload() -> None:
    """Long uploads renew the accepted reservation before every write."""
    renewals: list[list[dict[str, str]]] = []
    renewal_digests: list[str] = []
    api, reviewer, _ = _reviewer(
        record_count=2,
        reserver=lambda issue, submission, outcome, reviewer, records, digest: (
            BrokerReservationResult(
                token="reservation",
                decision_reviewer="maintainer",
                decision_created_at="2026-09-06T12:00:00Z",
            )
        ),
        renewer=lambda issue, submission, outcome, token, records, digest: (
            renewals.append(records) or renewal_digests.append(digest) or token
        ),
    )

    reviewer.decide(SUBMISSION, "accepted", "maintainer")

    assert len(renewals) == 3
    assert len(set(renewal_digests)) == 1
    assert renewal_digests[0] == hashlib.sha256(_decision_content(api)).hexdigest()
    assert all(record["canonical_path"] for record in renewals[0])


class FakeHfApi:
    """In-memory subset of the HfApi bucket interface."""

    def __init__(self) -> None:
        """Initialise empty bucket contents and failure controls."""
        self.files: dict[tuple[str, str], bytes] = {}
        self.uploads: list[tuple[str, str]] = []
        self.uploaded_content: list[tuple[str, str, bytes]] = []
        self.fail_result_upload_number: int | None = None
        self.fail_decision_upload = False
        self.result_uploads = 0
        self.decision_upload_barrier: threading.Barrier | None = None

    def batch_bucket_files(
        self, bucket_id: str, add: list[tuple[bytes, str]], token: str
    ) -> None:
        """Store exact object bytes, optionally injecting a partial failure.

        Raises:
            RuntimeError:
                When the configured interruption point is reached.
        """
        for content, path in add:
            if (
                bucket_id == STAGING
                and path.startswith(f"volunteer/decisions/{SUBMISSION}/")
                and self.fail_decision_upload
            ):
                raise RuntimeError("injected decision interruption")
            if bucket_id == RESULTS:
                self.result_uploads += 1
                if self.result_uploads == self.fail_result_upload_number:
                    raise RuntimeError("injected upload interruption")
            if (
                bucket_id == STAGING
                and path.startswith(f"volunteer/decisions/{SUBMISSION}/")
                and self.decision_upload_barrier
            ):
                self.decision_upload_barrier.wait()
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


def _decision_content(api: FakeHfApi) -> bytes:
    paths = _decision_paths(api)
    assert len(paths) == 1
    return api.files[(STAGING, paths[0])]


def _decision_paths(api: FakeHfApi) -> list[str]:
    return sorted(
        path
        for bucket, path in api.files
        if bucket == STAGING and path.startswith(f"volunteer/decisions/{SUBMISSION}/")
    )


def _reviewer(
    record_count: int = 1,
    reserver: (
        t.Callable[
            [int, str, str, str, list[dict[str, str]], str | None],
            BrokerReservationResult,
        ]
        | None
    ) = None,
    renewer: t.Callable[[int, str, str, str, list[dict[str, str]], str], str]
    | None = None,
    binder: t.Callable[[int, str, str, str, str, list[dict[str, str]]], str]
    | None = None,
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
        "language_group": "da",
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
        binder=binder,
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
    assert not _decision_paths(api)


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
        decision_digest: str | None,
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

    decision = _decision_content(api)
    decision_object = json.loads(decision)
    first_reviewer = reservation_order[0]
    assert decision_object["reviewer"] == first_reviewer
    assert decision_object["decided_at"] == server_times[first_reviewer]
    decision_paths = _decision_paths(api)
    decision_writes = [
        content
        for bucket, path, content in api.uploaded_content
        if bucket == STAGING and path in decision_paths
    ]
    assert len(decision_writes) == 1
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


def test_decision_persistence_failure_prevents_canonical_write() -> None:
    """Canonical results are untouched when the decision cannot be verified."""
    api, reviewer, broker_calls = _reviewer()
    api.fail_decision_upload = True

    with pytest.raises(RuntimeError, match="decision interruption"):
        reviewer.decide(SUBMISSION, "accepted", "maintainer")

    assert not [key for key in api.files if key[0] == RESULTS]
    assert broker_calls == []


def test_existing_decision_is_authoritative_after_reservation_expiry() -> None:
    """A same-outcome retry does not rewrite broker-bound decision metadata."""
    reservations: list[str] = []

    def reserve(
        issue: int,
        submission: str,
        outcome: str,
        reviewer: str,
        records: list[dict[str, str]],
        decision_digest: str | None,
    ) -> BrokerReservationResult:
        reservations.append(outcome)
        attempt = len(reservations)
        return BrokerReservationResult(
            token=f"reservation-{attempt}",
            decision_reviewer=f"reviewer-{attempt}",
            decision_created_at=f"2026-09-06T12:0{attempt}:00Z",
        )

    api, reviewer, broker_calls = _reviewer(reserver=reserve)
    reviewer.decide(SUBMISSION, "rejected", "alice")
    decision = _decision_content(api)

    reviewer.decide(SUBMISSION, "rejected", "bob")

    assert reservations == ["rejected", "rejected"]
    assert _decision_content(api) == decision
    assert broker_calls == [(12, SUBMISSION, "rejected"), (12, SUBMISSION, "rejected")]


def test_expired_reservation_cannot_change_durable_decision() -> None:
    """An expired accepted decision still blocks a new rejected reservation."""
    reservations: list[str] = []

    def reserve(
        issue: int,
        submission: str,
        outcome: str,
        reviewer: str,
        records: list[dict[str, str]],
        decision_digest: str | None,
    ) -> BrokerReservationResult:
        reservations.append(outcome)
        attempt = len(reservations)
        return BrokerReservationResult(
            token=f"reservation-{attempt}",
            decision_reviewer=f"reviewer-{attempt}",
            decision_created_at=f"2026-09-06T12:0{attempt}:00Z",
        )

    api, reviewer, broker_calls = _reviewer(record_count=2, reserver=reserve)
    api.fail_result_upload_number = 2
    with pytest.raises(RuntimeError, match="interruption"):
        reviewer.decide(SUBMISSION, "accepted", "maintainer")

    api.fail_result_upload_number = None
    uploads_before_reject = list(api.uploads)
    with pytest.raises(ReviewError, match="opposite"):
        reviewer.decide(SUBMISSION, "rejected", "maintainer")

    assert reservations == ["accepted"]
    assert api.uploads == uploads_before_reject
    reviewer.decide(SUBMISSION, "accepted", "maintainer")
    assert reservations == ["accepted", "accepted"]
    assert len([key for key in api.files if key[0] == RESULTS]) == 2
    assert broker_calls == [(12, SUBMISSION, "accepted")]


@pytest.mark.parametrize("language_group", ["sv", 42, ["da"]])
def test_manifest_forged_language_group_is_rejected(language_group: object) -> None:
    """A top-level language group must not contradict the trusted scope."""
    api, reviewer, _ = _reviewer()
    manifest = _manifest(api)
    manifest["language_group"] = language_group
    _store_manifest(api, manifest)

    with pytest.raises(ReviewError, match="language_group"):
        reviewer.show(SUBMISSION)


def _manifest(api: FakeHfApi) -> dict[str, object]:
    return json.loads(api.files[(STAGING, f"volunteer/manifests/{SUBMISSION}.json")])


def test_manifest_language_group_is_valid_in_real_flow() -> None:
    """A trusted language group survives review and promotion."""
    api, reviewer, broker_calls = _reviewer()

    report = reviewer.show(SUBMISSION)
    assert report.provenance["language_group"] == "da"
    reviewer.decide(SUBMISSION, "accepted", "maintainer")

    assert broker_calls == [(12, SUBMISSION, "accepted")]
    assert [key for key in api.files if key[0] == RESULTS]


def test_manifest_missing_language_group_is_rejected() -> None:
    """A manifest without a top-level language group is invalid."""
    api, reviewer, _ = _reviewer()
    manifest = _manifest(api)
    manifest.pop("language_group")
    _store_manifest(api, manifest)

    with pytest.raises(ReviewError, match="language_group"):
        reviewer.show(SUBMISSION)


def test_manifest_scope_mismatch_is_rejected() -> None:
    """Expected and actual canonical identities must match."""
    api, reviewer, _ = _reviewer()
    manifest = _manifest(api)
    expected_scope = t.cast(dict[str, object], manifest["expected_scope"])
    expected_scope["identity_suffixes"] = ['["other",false,false]']
    _store_manifest(api, manifest)

    with pytest.raises(ReviewError, match="scope differs"):
        reviewer.show(SUBMISSION)


def test_opposite_concurrent_decisions_fail_closed_before_canonical_writes() -> None:
    """Content-addressed races leave both immutable outcomes unpromoted."""
    reservations = iter(("accepted-token", "rejected-token"))
    bind_barrier = threading.Barrier(2)

    def reserve(
        issue: int,
        submission: str,
        outcome: str,
        reviewer: str,
        records: list[dict[str, str]],
        decision_digest: str | None,
    ) -> BrokerReservationResult:
        return BrokerReservationResult(
            token=next(reservations),
            decision_reviewer=reviewer,
            decision_created_at="2026-09-06T12:00:00Z",
        )

    def bind(
        issue: int,
        submission: str,
        outcome: str,
        token: str,
        digest: str,
        records: list[dict[str, str]],
    ) -> str:
        bind_barrier.wait()
        return token

    api, reviewer, broker_calls = _reviewer(reserver=reserve, binder=bind)
    api.decision_upload_barrier = threading.Barrier(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(reviewer.decide, SUBMISSION, outcome, "maintainer")
            for outcome in ("accepted", "rejected")
        ]
        errors = [future.exception() for future in futures]

    assert all(isinstance(error, ReviewError) for error in errors)
    decision_paths = _decision_paths(api)
    assert len(decision_paths) == 2
    outcomes = {
        json.loads(api.files[(STAGING, path)][:-1])["outcome"]
        for path in decision_paths
    }
    assert outcomes == {"accepted", "rejected"}
    assert not [key for key in api.files if key[0] == RESULTS]
    assert broker_calls == []


def test_partial_approve_resumes_and_is_idempotent() -> None:
    """Approval resumes partial uploads and preserves its decision artifact."""
    api, reviewer, broker_calls = _reviewer(record_count=2)
    api.fail_result_upload_number = 2

    with pytest.raises(RuntimeError, match="interruption"):
        reviewer.decide(SUBMISSION, "accepted", "maintainer")
    assert len([key for key in api.files if key[0] == RESULTS]) == 1

    api.fail_result_upload_number = None
    reviewer.decide(SUBMISSION, "accepted", "maintainer")
    decision = _decision_content(api)
    reviewer.decide(SUBMISSION, "accepted", "another-reviewer")

    assert _decision_content(api) == decision
    assert len([key for key in api.files if key[0] == RESULTS]) == 2
    assert broker_calls == [(12, SUBMISSION, "accepted"), (12, SUBMISSION, "accepted")]


def test_reject_is_idempotent_and_opposite_decision_fails() -> None:
    """Rejection retries safely while the opposite outcome is forbidden."""
    api, reviewer, broker_calls = _reviewer()
    reviewer.decide(SUBMISSION, "rejected", "maintainer", ["implausible scores"])
    decision = _decision_content(api)
    reviewer.decide(SUBMISSION, "rejected", "maintainer", ["implausible scores"])

    assert _decision_content(api) == decision
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
        decision_digest: str | None,
    ) -> BrokerReservationResult:
        reservation_reviewers.append(reviewer)
        return bound

    api, reviewer, _ = _reviewer(reserver=reserve)
    reviewer.decide(SUBMISSION, "rejected", "alice")
    decision = _decision_content(api)
    reviewer.decide(SUBMISSION, "rejected", "bob")

    assert _decision_content(api) == decision
    assert reservation_reviewers == ["alice", "bob"]


def test_review_renewal_round_trip_includes_bound_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send the bound decision digest in the broker renewal envelope."""
    requests: list[dict[str, object]] = []

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "protocol_version": "volunteer-worker/v1",
                    "status": "reserved",
                    "token": "reservation",
                }
            ).encode()

    def urlopen(request: urllib.request.Request, timeout: int) -> Response:
        assert timeout == 30
        requests.append(json.loads(request.data.decode()))
        return Response()

    monkeypatch.setenv("VOLUNTEER_PROMOTION_SECRET", "secret")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    digest = "a" * 64
    records = [
        {
            "identity": '["org/model","dataset",false,false]',
            "canonical_path": "org_model/dataset__test__zeroshot.json",
            "digest": "b" * 64,
        }
    ]

    assert (
        renew_with_broker(12, SUBMISSION, "accepted", "reservation", records, digest)
        == "reservation"
    )
    assert requests[0]["decision_digest"] == digest
    assert requests[0]["reservation_token"] == "reservation"


def test_terminal_response_loss_recovers_with_durable_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh reviewer recovers a completed terminal broker transition."""
    api, template, _ = _reviewer()
    reviewer = VolunteerReviewer(
        store=template.store, results_bucket=RESULTS, scope_policy=template.scope_policy
    )
    requests: list[tuple[str, dict[str, object]]] = []
    reservation = {
        "token": "reservation",
        "decision_reviewer": "alice",
        "decision_created_at": "2026-09-06T12:00:00Z",
        "status": "reserved",
    }
    lost_response = True
    bound_digest: str | None = None

    class Response:
        def __init__(self, body: dict[str, object]) -> None:
            self.body = body

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(self.body).encode()

    def urlopen(request: urllib.request.Request, timeout: int) -> Response:
        nonlocal bound_digest, lost_response
        body = json.loads(request.data.decode())
        requests.append((request.full_url, body))
        assert request.get_header("X-promotion-secret") == "secret"
        if request.full_url.endswith("/reserve"):
            if "reservation_token" not in body:
                if "decision_digest" not in body:
                    return Response(
                        {
                            "status": "reserved",
                            "token": reservation["token"],
                            "decision_reviewer": reservation["decision_reviewer"],
                            "decision_created_at": reservation["decision_created_at"],
                        }
                    )
                assert body["decision_digest"] == bound_digest
                assert reservation["status"] == "terminal"
                return Response(
                    {
                        "status": "terminal",
                        "token": reservation["token"],
                        "decision_reviewer": reservation["decision_reviewer"],
                        "decision_created_at": reservation["decision_created_at"],
                        "decision_digest": decision_digest,
                    }
                )
            assert body["reservation_token"] == reservation["token"]
            bound_digest = body["decision_digest"]
            return Response(
                {
                    "status": reservation["status"],
                    "token": reservation["token"],
                    "decision_digest": bound_digest,
                }
            )
        assert request.full_url.endswith("/promote")
        assert body["reservation_token"] == reservation["token"]
        assert body["decision_digest"] == bound_digest
        reservation["status"] = "terminal"
        if lost_response:
            lost_response = False
            raise OSError("completed response was lost")
        return Response({"status": "rejected"})

    monkeypatch.setenv("VOLUNTEER_PROMOTION_SECRET", "secret")
    monkeypatch.setenv("VOLUNTEER_BROKER_RESERVATION_URL", "https://broker/reserve")
    monkeypatch.setenv("VOLUNTEER_BROKER_PROMOTION_URL", "https://broker/promote")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    with pytest.raises(OSError, match="response was lost"):
        reviewer.decide(SUBMISSION, "rejected", "alice")
    decision_digest = hashlib.sha256(_decision_content(api)).hexdigest()
    assert bound_digest == decision_digest

    fresh_reviewer = VolunteerReviewer(
        store=template.store, results_bucket=RESULTS, scope_policy=template.scope_policy
    )
    fresh_reviewer.decide(SUBMISSION, "rejected", "bob")

    reserve_requests = [
        body
        for url, body in requests
        if url.endswith("/reserve") and "reservation_token" not in body
    ]
    assert "decision_digest" not in reserve_requests[0]
    assert reserve_requests[1]["decision_digest"] == decision_digest
    assert reserve_requests[1]["records"] == reserve_requests[0]["records"]
    assert "reservation_token" not in reserve_requests[1]
    assert len([body for url, body in requests if url.endswith("/promote")]) == 2
