"""Data contracts for volunteer-result review."""

from __future__ import annotations

import collections.abc as c
import dataclasses
import typing as t
from pathlib import Path

from leaderboards.result_identity import ResultIdentity

JsonObject: t.TypeAlias = dict[str, object]
BrokerPromoter: t.TypeAlias = t.Callable[
    [int, str, str, str, list[dict[str, str]], str], None
]
BrokerReservation: t.TypeAlias = t.Callable[
    [int, str, str, str, list[dict[str, str]]], "BrokerReservationResult"
]
BrokerRenewer: t.TypeAlias = t.Callable[[int, str, str, str, list[dict[str, str]]], str]
BrokerBinder: t.TypeAlias = t.Callable[
    [int, str, str, str, str, list[dict[str, str]]], str
]


@dataclasses.dataclass(frozen=True)
class BrokerReservationResult:
    """Reservation token and server-bound decision metadata."""

    token: str
    decision_reviewer: str
    decision_created_at: str
    decision_digest: str | None = None


class BucketEntry(t.Protocol):
    """Bucket file metadata used for exact verification."""

    type: str
    path: str
    size: int
    xet_hash: str


class BucketInfo(t.Protocol):
    """Bucket visibility metadata used by the reviewer."""

    private: bool


class BucketApi(t.Protocol):
    """HfApi bucket operations required by the reviewer."""

    def batch_bucket_files(
        self, bucket_id: str, *, add: list[tuple[bytes, str]], token: str
    ) -> object:
        """Upload selected bucket files."""
        ...

    def bucket_info(self, bucket_id: str, *, token: str) -> BucketInfo:
        """Return bucket visibility metadata."""
        ...

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[str, Path]],
        *,
        raise_on_missing_files: bool,
        token: str,
    ) -> None:
        """Download selected bucket files."""
        ...

    def get_bucket_paths_info(
        self, bucket_id: str, paths: list[str], *, token: str
    ) -> c.Iterable[BucketEntry]:
        """Return metadata for selected paths."""
        ...

    def list_bucket_tree(
        self, bucket_id: str, prefix: str, *, recursive: bool, token: str
    ) -> c.Iterable[BucketEntry]:
        """List files beneath a bucket prefix."""
        raise NotImplementedError(
            f"BucketApi implementations must handle recursive={recursive!r}"
        )


class ReviewError(RuntimeError):
    """Raised when staged evidence cannot be safely promoted."""


@dataclasses.dataclass(frozen=True)
class ValidatedRecord:
    """One byte-verified, schema-verified staged result."""

    identity: ResultIdentity
    digest: str
    staged_path: str
    canonical_path: str
    content: bytes
    scores: tuple[tuple[str, float], ...]
    warnings: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ReviewReport:
    """Independently validated evidence for one submission."""

    submission_id: str
    issue_number: int
    contributor: str
    model_id: str
    model_revision: str
    language: str
    euroeval_version: str
    provenance: JsonObject
    expected_identities: tuple[ResultIdentity, ...]
    records: tuple[ValidatedRecord, ...]
    warnings: tuple[str, ...]
    checks: tuple[str, ...]
