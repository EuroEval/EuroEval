"""Loading of results, to be converted into leaderboards."""

from __future__ import annotations

import logging
import typing as t
from functools import cache

from .backup import backup_results
from .bucket_sync import download_missing_bucket_files
from .constants import NEW_RESULTS_PATH, RESULTS_DIR
from .contamination_canary import is_canary_record
from .eee_validation import is_eee_record
from .jsonl_io import load_records_from_jsonl_files, load_records_from_result_tree
from .result_identity import dedup_newer_record, identity_from_eee_record

logger = logging.getLogger(__name__)

_EXCLUDED_MODELS: frozenset[str] = frozenset()
_HIDE_AUXILIARY_RESULTS = False


def configure_leaderboard_result_filter(*, excluded_models: set[str]) -> None:
    """Configure filtering for later leaderboard reads in this process.

    Args:
        excluded_models:
            Base model IDs that the maintainer chose to remove.
    """
    global _EXCLUDED_MODELS, _HIDE_AUXILIARY_RESULTS  # noqa: PLW0603
    _EXCLUDED_MODELS = frozenset(excluded_models)
    _HIDE_AUXILIARY_RESULTS = True
    load_raw_results.cache_clear()


def reset_leaderboard_result_filter() -> None:
    """Expose source records for a fresh processing pass."""
    global _EXCLUDED_MODELS, _HIDE_AUXILIARY_RESULTS  # noqa: PLW0603
    _EXCLUDED_MODELS = frozenset()
    _HIDE_AUXILIARY_RESULTS = False
    load_raw_results.cache_clear()


@cache
def load_raw_results() -> list[dict[str, t.Any]]:
    """Load all EEE-format results from the results directory.

    Syncs the EuroEval/results bucket into RESULTS_DIR, then reads every
    per-record JSON file from the tree structure
    (``results/<model>/<dataset>__<split>__<shot>.json``). Records staged in
    NEW_RESULTS_PATH (a transient JSONL file) are folded in (and the staging
    file removed) so manually added results are picked up exactly once. No
    distinction is made between raw and processed results; leaderboard consumers
    receive the EEE records exactly as stored.

    Deduplicates ordinary records by canonical storage identity using
    :func:`.result_identity.dedup_newer_record`. Auxiliary canary records remain
    distinct until private scoring has rejected immutable conflicts or selected the
    newest mutable snapshot.

    Only the EEE envelope structure is validated here. The "precious" metadata
    fields (commercially_licensed, open, trained_from_scratch) are intentionally
    not required at load time: results posted from compute VMs can't supply them,
    so they're filled in later by the interactive ``add_missing_entries`` step
    and enforced when the processed records are written back out.

    Returns:
        All evaluation results in EEE format. Ordinary records are deduplicated by
        storage identity; auxiliary records retain candidate snapshots for scoring.

    Raises:
        ValueError:
            If a loaded record is not an EEE-format record.
    """
    _sync_results_from_bucket()

    records = load_records_from_result_tree(results_dir=RESULTS_DIR)
    logger.info(f"Loaded {len(records):,} existing results from {RESULTS_DIR}.")

    if NEW_RESULTS_PATH.exists():
        new_records = load_records_from_jsonl_files(paths=[NEW_RESULTS_PATH])
        logger.info(f"Loaded {len(new_records):,} new results from staging file.")
        records.extend(new_records)
        NEW_RESULTS_PATH.unlink()

    # Canary duplicates must reach private scoring so immutable conflicts cannot be
    # hidden by ordinary storage-identity deduplication.
    canary_records = [record for record in records if is_canary_record(record)]
    ordinary_records = [record for record in records if not is_canary_record(record)]
    records = [*_dedup_by_storage_identity(records=ordinary_records), *canary_records]

    for idx, record in enumerate(records, start=1):
        if not is_eee_record(record=record):
            raise ValueError(f"raw results record {idx:,} is not an EEE-format record.")

    if _HIDE_AUXILIARY_RESULTS:
        records = [
            record
            for record in records
            if not is_canary_record(record)
            and not _model_is_excluded(identity_from_eee_record(record)[0])
        ]
    return records


def _model_is_excluded(model_name: str) -> bool:
    """Return whether a stored identity belongs to an excluded base model."""
    return any(
        model_name == model_id
        or model_name.startswith(f"{model_id}@")
        or model_name.startswith(f"{model_id}#")
        for model_id in _EXCLUDED_MODELS
    )


def _dedup_by_storage_identity(
    records: list[dict[str, t.Any]],
) -> list[dict[str, t.Any]]:
    """Deduplicate records by canonical storage identity.

    Groups records by ``(model_id, dataset, validation_split, few_shot)`` and
    keeps only the newest record per identity using
    :func:`.result_identity.dedup_newer_record`.

    Args:
        records:
            List of EEE-format records, potentially with duplicates.

    Returns:
        Deduplicated list with one record per unique storage identity.
    """
    # Group by identity
    by_identity: dict[tuple[str, str, bool | None, bool | None], dict[str, t.Any]] = {}

    for record in records:
        identity = identity_from_eee_record(record=record)
        existing = by_identity.get(identity)
        if existing is None:
            by_identity[identity] = record
        else:
            by_identity[identity] = dedup_newer_record(
                record_a=existing, record_b=record
            )

    return list(by_identity.values())


def _sync_results_from_bucket() -> None:
    """Sync the HF results bucket into RESULTS_DIR and back it up.

    Performs an incremental fetch of the single EuroEval/results bucket into
    RESULTS_DIR (downloading only files not already present locally) and creates
    a backup of the synced per-record JSON files.

    Raises:
        FileNotFoundError:
            If sync fails and no local files exist.
    """
    download_missing_bucket_files()

    file_count = len(list(RESULTS_DIR.glob("*/*.json")))
    if file_count == 0:
        raise FileNotFoundError(
            "No results available. Sync failed and no local cache exists."
        )
    logger.info(f"Synced {file_count:,} result files to {RESULTS_DIR}.")

    backup_path = backup_results()
    if backup_path:
        logger.info(f"Backup created at {backup_path}.")
