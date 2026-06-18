"""Loading of results, to be converted into leaderboards."""

from __future__ import annotations

import logging
import typing as t
from functools import cache

from .backup import backup_results
from .bucket_sync import sync_bucket
from .constants import NEW_RESULTS_PATH, RESULTS_DIR
from .eee_validation import is_eee_record
from .records import load_records_from_jsonl_files

logger = logging.getLogger(__name__)


@cache
def load_raw_results() -> list[dict[str, t.Any]]:
    """Load all EEE-format results from the results directory.

    Syncs the EuroEval/results bucket into RESULTS_DIR, then reads every
    per-model JSONL file. Records staged in NEW_RESULTS_PATH are folded in
    (and the staging file removed) so manually added results are picked up
    exactly once. No distinction is made between raw and processed results;
    leaderboard consumers receive the EEE records exactly as stored.

    Only the EEE envelope structure is validated here. The "precious" metadata
    fields (commercially_licensed, open, trained_from_scratch) are intentionally
    not required at load time: results posted from compute VMs can't supply them,
    so they're filled in later by the interactive ``add_missing_entries`` step
    and enforced when the processed records are written back out.

    Returns:
        All evaluation results in EEE format.

    Raises:
        ValueError:
            If a loaded record is not an EEE-format record.
    """
    _sync_results_from_bucket()

    model_files = sorted(RESULTS_DIR.glob("*.jsonl"))
    records = load_records_from_jsonl_files(paths=model_files)
    logger.info(f"Loaded {len(records):,} existing results from {RESULTS_DIR}.")

    if NEW_RESULTS_PATH.exists():
        new_records = load_records_from_jsonl_files(paths=[NEW_RESULTS_PATH])
        records.extend(new_records)
        logger.info(f"Loaded {len(new_records):,} new results.")
        NEW_RESULTS_PATH.unlink()

    for idx, record in enumerate(records, start=1):
        if not is_eee_record(record=record):
            raise ValueError(f"raw results record {idx:,} is not an EEE-format record.")

    return records


def _sync_results_from_bucket() -> None:
    """Sync the HF results bucket into RESULTS_DIR and back it up.

    Syncs the single EuroEval/results bucket to RESULTS_DIR and creates a
    backup of the synced per-model JSONL files.

    Raises:
        FileNotFoundError:
            If sync fails and no local files exist.
    """
    sync_bucket()

    file_count = len(list(RESULTS_DIR.glob("*.jsonl")))
    if file_count == 0:
        raise FileNotFoundError(
            "No results available. Sync failed and no local cache exists."
        )
    logger.info(f"Synced {file_count:,} model files to {RESULTS_DIR}.")

    backup_path = backup_results()
    if backup_path:
        logger.info(f"Backup created at {backup_path}.")
