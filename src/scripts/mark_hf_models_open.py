"""Sweep models not marked as open and relabel them if their HF repo exists.

This script iterates through JSONL result files in the results directory,
checks whether each model's canonical Hugging Face repository exists, and
if so, marks the model as ``open=True`` in ``model_info.additional_details``.

Only models with ``open`` missing, ``False``, or ``None`` are considered.
Models already explicitly marked ``open=True`` are preserved. Models with
no HF repo are left unchanged (``open`` is not set to ``False``).

The script uses exponential backoff when calling the Hugging Face Hub API
to handle rate limits and transient errors. It uploads only changed JSONL
files to the HF bucket.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import subprocess
import sys
import time
import typing as t
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    RepositoryNotFoundError,
)

from euroeval.jsonl_io import parse_jsonl_lines
from leaderboards.bucket_sync import RESULTS_DIR

logger = logging.getLogger(__name__)

# Regex to strip variant suffixes like (val), (zero-shot), etc.
# Matches patterns like " (val)", " (zero-shot)", " (zero-shot, val)" at end
VARIANT_SUFFIX_RE = re.compile(r"\s*\((?:zero-shot|val)(?:,\s*(?:zero-shot|val))*\)$")

# Regex to strip @revision and #param suffixes
REVISION_PARAM_RE = re.compile(r"(@[^@#]+|#.+)$")


def canonical_model_id(model_id: str) -> str:
    """Extract the canonical HF repo ID from a model identifier.

    Strips:
    - HTML anchors
    - Variant suffixes like ``(val)``, ``(zero-shot)``
    - ``@revision`` suffixes
    - ``#param`` suffixes (e.g. ``#no-thinking``, ``#thinking``)

    Preserves the ``org/repo`` slug for HF Hub lookups.

    Args:
        model_id:
            The model identifier, possibly with anchors, variant suffixes,
            revisions, or param markers.

    Returns:
        The canonical ``org/repo`` HF repo ID.
    """
    from leaderboards.constants import ANCHOR_RE, VARIANT_SUFFIX_RE  # noqa: PLC0415

    # Strip HTML anchors first
    anchor_match = ANCHOR_RE.search(model_id)
    if anchor_match:
        cleaned = anchor_match.group("inner")
    else:
        cleaned = model_id

    # Strip @revision and #param suffixes (these can appear after variant suffixes)
    # Keep stripping until no more matches (handles multiple suffixes)
    while True:
        new_cleaned = REVISION_PARAM_RE.sub("", cleaned)
        if new_cleaned == cleaned:
            break
        cleaned = new_cleaned

    # Strip variant suffixes like (val), (zero-shot)
    cleaned = VARIANT_SUFFIX_RE.sub("", cleaned)

    return cleaned.strip()


def check_hf_repo_exists(
    hf_api: HfApi,
    repo_id: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> bool | None:
    """Check whether a Hugging Face model repository exists.

    Uses exponential backoff with jitter for transient errors and rate limits.

    Args:
        hf_api:
            The Hugging Face API client.
        repo_id:
            The repository ID to check (e.g. ``"meta-llama/Llama-2-7b"``).
        max_retries (optional):
            Maximum number of retry attempts. Defaults to 5.
        base_delay (optional):
            Base delay in seconds for exponential backoff. Defaults to 1.0.
        max_delay (optional):
            Maximum delay in seconds. Defaults to 60.0.

    Returns:
        - ``True`` if the repo exists (includes ``GatedRepoError``, which
          indicates the repo exists but requires access).
        - ``False`` if the repo definitely does not exist.
        - ``None`` if the check failed after all retries (transient error).
    """
    delay = base_delay
    attempt = 0

    while attempt <= max_retries:
        try:
            # model_info raises on missing repo, returns ModelInfo on success
            # GatedRepoError means the repo exists but we can't access it
            hf_api.model_info(repo_id=repo_id)
            return True
        except GatedRepoError:
            # Repo exists but is gated - counts as existing
            return True
        except (RepositoryNotFoundError, HFValidationError):
            # Definitely doesn't exist or invalid ID
            return False
        except (HfHubHTTPError, TimeoutError, OSError) as e:
            # Transient error - retry with backoff
            attempt += 1
            if attempt > max_retries:
                logger.warning(
                    f"HF repo check for {repo_id!r} failed "
                    f"after {attempt} attempts: {e}"
                )
                return None
            # Exponential backoff with jitter
            sleep_time = min(
                delay * (2 ** (attempt - 1)) + random.uniform(0, 0.1 * delay), max_delay
            )
            logger.debug(
                f"HF repo check for {repo_id!r} failed (attempt {attempt}): {e}. "
                f"Retrying in {sleep_time:.1f}s..."
            )
            time.sleep(sleep_time)
        except Exception as e:
            # Unexpected error - log and treat as transient
            attempt += 1
            if attempt > max_retries:
                logger.warning(
                    f"Unexpected error checking {repo_id!r} "
                    f"after {attempt} attempts: {e}"
                )
                return None
            sleep_time = min(
                delay * (2 ** (attempt - 1)) + random.uniform(0, 0.1 * delay), max_delay
            )
            logger.debug(
                f"Unexpected error for {repo_id!r} (attempt {attempt}): {e}. "
                f"Retrying in {sleep_time:.1f}s..."
            )
            time.sleep(sleep_time)

    return None


def process_jsonl_file(
    jsonl_path: Path,
    hf_api: HfApi,
    existence_cache: dict[str, bool],
    dry_run: bool = False,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> tuple[int, int, int]:
    """Process a single JSONL file, marking open models as ``open=True``.

    Only mutates ``model_info.additional_details.open`` to True for records
    whose HF repo exists. Records with malformed JSON cause the function to
    raise ``ValueError`` - the file is not modified in that case.

    Args:
        jsonl_path:
            Path to the JSONL file.
        hf_api:
            The Hugging Face API client.
        existence_cache:
            Cache mapping canonical repo IDs to existence status.
        dry_run (optional):
            If True, log what would be changed without modifying the file.
        max_retries (optional):
            Maximum number of retry attempts for HF API calls.
        base_delay (optional):
            Base delay in seconds for exponential backoff.
        max_delay (optional):
            Maximum delay in seconds.

    Returns:
        Tuple of (records_checked, records_changed, retries_used).
    """
    records_checked = 0
    records_changed = 0
    retries_used = 0

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    # Use strict=True to fail on invalid JSON rather than silently dropping lines
    records = parse_jsonl_lines(lines=lines, source=str(jsonl_path), strict=True)

    modified_records: list[dict] = []

    for record in records:
        # Navigate to model_info.additional_details.open in EEE format
        model_info = record.get("model_info", {})
        additional_details = model_info.get("additional_details")

        # Treat missing key same as None - need to create dict
        if additional_details is None:
            additional_details = {}
            model_info_typed = t.cast("dict[str, object]", model_info)
            model_info_typed["additional_details"] = additional_details

        open_value = additional_details.get("open")
        model_id = ""
        if model_info:
            model_id = model_info.get("id", model_info.get("name", ""))

        records_checked += 1

        # Skip if already explicitly open=True
        if open_value is True:
            continue

        # Only consider records where open is missing, False, or None
        if open_value is False or open_value is None:
            canonical_id = canonical_model_id(model_id=model_id)

            if not canonical_id:
                logger.debug(f"Could not extract canonical ID from {model_id!r}")
                continue

            # Check cache first
            if canonical_id in existence_cache:
                exists = existence_cache[canonical_id]
            else:
                logger.debug(f"Checking HF repo: {canonical_id!r}")
                exists = check_hf_repo_exists(
                    hf_api=hf_api,
                    repo_id=canonical_id,
                    max_retries=max_retries,
                    base_delay=base_delay,
                    max_delay=max_delay,
                )
                existence_cache[canonical_id] = exists if exists is not None else False
                if exists is None:
                    retries_used += 1

            if exists:
                logger.info(f"Marking {model_id!r} -> {canonical_id!r} as open=True")
                additional_details["open"] = True
                records_changed += 1
            elif exists is False:
                logger.debug(
                    f"HF repo {canonical_id!r} not found - skipping {model_id!r}"
                )

    if records_changed > 0 and not dry_run:
        modified_records = records
        # Write back with trailing newline
        with jsonl_path.open("w", encoding="utf-8") as f:
            for record in modified_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            f"Updated {records_changed}/{records_checked} records in {jsonl_path.name}"
        )
    elif records_changed > 0 and dry_run:
        logger.info(
            f"[dry-run] Would update {records_changed}/{records_checked} records "
            f"in {jsonl_path.name}"
        )

    return records_checked, records_changed, retries_used


def upload_changed_files(
    changed_files: list[Path],
    results_dir: Path,
    hf_bucket: str = "EuroEval/results",
    dry_run: bool = False,
) -> int:
    """Upload changed JSONL files to the Hugging Face bucket using ``hf buckets cp``.

    Uses the ``hf buckets cp`` CLI to copy individual files to the bucket.
    Authentication is handled automatically via cached credentials or ``HF_TOKEN``
    environment variable - no explicit token is passed or printed.

    Args:
        changed_files:
            List of JSONL file paths to upload.
        results_dir:
            Directory containing the JSONL files (parent directory for source paths).
        hf_bucket (optional):
            The HF bucket path (e.g. ``"EuroEval/results"``).
        dry_run (optional):
            If True, log what would be uploaded without actually uploading.

    Returns:
        Number of files successfully uploaded.
    """
    uploaded = 0

    for file_path in changed_files:
        if dry_run:
            logger.info(
                f"[dry-run] Would copy {file_path.name} to hf://buckets/{hf_bucket}/"
            )
            uploaded += 1
            continue

        try:
            # Use hf buckets cp CLI to upload individual files
            # Authentication is handled via cached credentials or HF_TOKEN env var
            bucket_path = f"hf://buckets/{hf_bucket}/{file_path.name}"
            subprocess.run(
                ["hf", "buckets", "cp", str(file_path), bucket_path, "--quiet"],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Uploaded {file_path.name} to {bucket_path}")
            uploaded += 1
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"Failed to upload {file_path.name} to bucket: {e.stderr or e.stdout}"
            )
        except FileNotFoundError:
            logger.warning(
                "hf CLI not found - skipping bucket upload. "
                "Install with: pip install huggingface_hub[cli]"
            )
            return uploaded

    return uploaded


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mark models as open=True if their HF repo exists"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Directory containing JSONL result files (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be changed without modifying files or uploading",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading changed files to the HF bucket",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts for HF API calls (default: 5)",
    )
    parser.add_argument(
        "--base-delay",
        type=float,
        default=1.0,
        help="Base delay in seconds for exponential backoff (default: 1.0)",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=60.0,
        help="Maximum delay in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of JSONL files to process (for testing)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    results_dir = args.results_dir
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    jsonl_files = sorted(results_dir.glob("*.jsonl"))
    if args.limit:
        jsonl_files = jsonl_files[: args.limit]

    if not jsonl_files:
        logger.warning(f"No JSONL files found in {results_dir}")
        sys.exit(0)

    logger.info(f"Found {len(jsonl_files)} JSONL files to process")

    hf_api = HfApi()
    existence_cache: dict[str, bool] = {}

    total_checked = 0
    total_changed = 0
    total_retries = 0
    changed_files: list[Path] = []
    skipped_missing = 0

    for jsonl_path in jsonl_files:
        logger.info(f"Processing {jsonl_path.name}...")
        checked, changed, retries = process_jsonl_file(
            jsonl_path=jsonl_path,
            hf_api=hf_api,
            existence_cache=existence_cache,
            dry_run=args.dry_run,
            max_retries=args.max_retries,
            base_delay=args.base_delay,
            max_delay=args.max_delay,
        )
        total_checked += checked
        total_changed += changed
        total_retries += retries
        if changed > 0:
            changed_files.append(jsonl_path)

    # Estimate skipped missing (repos that don't exist)
    skipped_missing = sum(1 for exists in existence_cache.values() if not exists)

    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"  Records checked: {total_checked}")
    logger.info(f"  Records changed: {total_changed}")
    logger.info(f"  Files modified: {len(changed_files)}")
    logger.info(f"  Unique repos checked: {len(existence_cache)}")
    logger.info(f"  Skipped (repo not found): {skipped_missing}")
    logger.info(f"  Retry count: {total_retries}")

    if not args.no_upload and changed_files:
        logger.info("Uploading changed files to HF bucket...")
        uploaded = upload_changed_files(
            changed_files=changed_files, results_dir=results_dir, dry_run=args.dry_run
        )
        logger.info(f"  Files uploaded: {uploaded}")
    elif changed_files and args.no_upload:
        logger.info("Upload skipped (--no-upload flag set)")
    elif not changed_files:
        logger.info("No files to upload (no changes made)")

    logger.info("Done!")


if __name__ == "__main__":
    main()
