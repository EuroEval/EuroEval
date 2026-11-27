"""Upload volunteer staging records to a verified private Hugging Face bucket."""

import asyncio
import re
import time

import httpx
from fastapi import HTTPException
from huggingface_hub import HfApi

from .dependencies import fetch_with_retry, require_env

_BUCKET_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_CACHE_SECONDS = 60
_bucket_cache: dict[str, tuple[bool, float]] = {}


async def upload_staging(http: httpx.AsyncClient, path: str, content: str) -> None:
    """Upload exact UTF-8 content to the configured private staging bucket.

    Args:
        http:
            Shared HTTP client used to verify bucket privacy.
        path:
            Server-owned path below volunteer/submissions/.
        content:
            Text content uploaded without transformation.

    Raises:
        HTTPException:
            If configuration, bucket privacy, lookup, path, or upload is invalid.
        asyncio.CancelledError:
            After an in-progress upload thread completes.
    """
    bucket = require_env(name="HF_STAGING_BUCKET")
    token = require_env(name="HF_TOKEN")
    if not _BUCKET_PATTERN.fullmatch(bucket):
        raise HTTPException(503, "HF_STAGING_BUCKET must be a namespace/bucket id.")
    if bucket.casefold() == "euroeval/results":
        raise HTTPException(503, "Canonical EuroEval results bucket is forbidden.")
    parts = path.split("/")
    if (
        len(parts) < 3
        or parts[:2] != ["volunteer", "submissions"]
        or any(part in {"", ".", ".."} for part in parts)
        or "\\" in path
    ):
        raise HTTPException(400, "Staging path must be under volunteer/submissions/.")

    cached = _bucket_cache.get(bucket)
    if cached is None or time.monotonic() - cached[1] > _CACHE_SECONDS:
        try:
            response = await fetch_with_retry(
                http=http,
                url=f"https://huggingface.co/api/buckets/{bucket}",
                headers={
                    "accept": "application/json",
                    "authorization": f"Bearer {token}",
                },
            )
            if not response.is_success:
                raise HTTPException(
                    502,
                    "Hugging Face bucket lookup failed with HTTP "
                    f"{response.status_code}.",
                )
            metadata: object = response.json()
            private = isinstance(metadata, dict) and metadata.get("private") is True
            _bucket_cache[bucket] = (private, time.monotonic())
            cached = _bucket_cache[bucket]
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(502, "Hugging Face bucket lookup failed.") from error
    if not cached[0]:
        raise HTTPException(503, "HF_STAGING_BUCKET is not private; refusing upload.")

    task = asyncio.create_task(
        asyncio.to_thread(
            HfApi(endpoint="https://huggingface.co", token=token).batch_bucket_files,
            bucket_id=bucket,
            add=[(content.encode("utf-8"), path)],
        )
    )
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise
    except Exception as error:
        raise HTTPException(502, "Hugging Face bucket upload failed.") from error
