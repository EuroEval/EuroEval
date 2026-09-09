"""Exact-file storage operations for volunteer-result review."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from .review_models import BucketApi, ReviewError

_MANIFEST_PREFIX = "volunteer/manifests"


class BucketStore:
    """Small exact-file wrapper around Hugging Face bucket APIs."""

    def __init__(self, api: BucketApi, token: str, staging_bucket: str) -> None:
        """Initialise and verify the private staging bucket.

        Raises:
            ReviewError:
                If the staging bucket is public.
        """
        self.api = api
        self.token = token
        self.staging_bucket = staging_bucket
        info = self.api.bucket_info(staging_bucket, token=token)
        if not info.private:
            raise ReviewError("HF_STAGING_BUCKET must be private")

    def list_manifests(self) -> list[str]:
        """List durable submission manifests without consulting Redis.

        Returns:
            Sorted bucket paths for all manifests.
        """
        entries = self.api.list_bucket_tree(
            self.staging_bucket,
            prefix=_MANIFEST_PREFIX,
            recursive=True,
            token=self.token,
        )
        return sorted(
            entry.path
            for entry in entries
            if getattr(entry, "type", None) == "file" and entry.path.endswith(".json")
        )

    def read_optional(self, bucket: str, path: str) -> bytes | None:
        """Read an object when present.

        Returns:
            Exact object bytes, or ``None`` when absent.
        """
        entries = list(self.api.get_bucket_paths_info(bucket, [path], token=self.token))
        if not entries:
            return None
        return self.read(bucket=bucket, path=path)

    def read(self, bucket: str, path: str) -> bytes:
        """Download one exact bucket object.

        Returns:
            Exact object bytes.

        Raises:
            ReviewError:
                If the object cannot be found.
        """
        with tempfile.TemporaryDirectory(prefix="euroeval-review-") as directory:
            target = Path(directory) / "object"
            self.api.download_bucket_files(
                bucket, [(path, target)], raise_on_missing_files=True, token=self.token
            )
            if not target.is_file():
                raise ReviewError(f"Bucket object is missing: {bucket}/{path}")
            return target.read_bytes()

    def write_verified(self, bucket: str, path: str, content: bytes) -> None:
        """Upload bytes and verify destination metadata and content."""
        self.api.batch_bucket_files(
            bucket_id=bucket, add=[(content, path)], token=self.token
        )
        self.verify(bucket=bucket, path=path, content=content)

    def verify(self, bucket: str, path: str, content: bytes) -> None:
        """Verify an existing object's metadata, content, and digest.

        Raises:
            ReviewError:
                If the object does not match the expected bytes.
        """
        entries = list(self.api.get_bucket_paths_info(bucket, [path], token=self.token))
        if len(entries) != 1:
            raise ReviewError(f"Upload metadata is missing for {bucket}/{path}")
        entry = entries[0]
        if entry.path != path or entry.size != len(content) or not entry.xet_hash:
            raise ReviewError(f"Upload metadata is inconsistent for {bucket}/{path}")
        downloaded = self.read(bucket=bucket, path=path)
        if downloaded != content or _digest(downloaded) != _digest(content):
            raise ReviewError(f"Upload verification failed for {bucket}/{path}")


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
