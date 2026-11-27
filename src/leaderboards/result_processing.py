"""Process EuroEval records into per-model result files.

This module is the orchestrator: it loads raw records, deduplicates them,
repairs and validates their metadata, and writes the processed results as one
JSON file per logical result, both locally in RESULTS_DIR and to the Hugging
Face results bucket.
"""

from __future__ import annotations

import json
import logging
import re
import typing as t
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import HfApi
from tqdm.auto import tqdm

from .cache import Cache
from .canonical_publication import BucketApi, CanonicalPublisher
from .constants import HF_RESULTS_BUCKET, RESULTS_DIR
from .contamination_canary import process_contamination_canaries
from .evaluation_common import resolve_hf_token
from .model_metadata import add_missing_entries, fix_metadata, record_is_valid
from .record_fields import deduplicate_records
from .records import get_model_name
from .result_conflict_prompt import resolve_result_interactively
from .result_conflicts import ConflictChoice, ConflictDecision, resolve_result_conflict
from .result_identity import (
    ResultIdentity,
    UnresolvedResultConflictError,
    identity_from_eee_record,
    raise_on_collision,
    record_relative_path,
)
from .result_loading import (
    configure_leaderboard_result_filter,
    load_raw_results,
    reset_leaderboard_result_filter,
)

logger = logging.getLogger(__name__)


def configure_canary_result_filter() -> None:
    """Score stored canary evidence and configure leaderboard-only filtering.

    This is used when callers intentionally skip ordinary result processing but still
    need auxiliary records and positively detected models excluded from generated
    leaderboards.
    """
    reset_leaderboard_result_filter()
    records = load_raw_results()
    _, _, excluded_models, canary_report = process_contamination_canaries(
        records=records
    )
    if canary_report.get("status") == "unavailable":
        logger.warning(
            "Contamination-canary scoring unavailable at stage %s (%s).",
            canary_report.get("stage", "unknown"),
            canary_report.get("reason", "unknown_error"),
        )
    configure_leaderboard_result_filter(excluded_models=excluded_models)
    load_raw_results.cache_clear()


def process_results(
    min_version: str,
    min_number_of_model_records: int,
    banned_versions: list[str],
    banned_model_patterns: list[re.Pattern],
    api_model_patterns: list[re.Pattern],
    trained_from_scratch_patterns: list[re.Pattern],
    upload_to_bucket: bool = False,
    conflict_resolver: Callable[..., ConflictDecision] | None = None,
) -> list[dict[str, t.Any]]:
    """Process EuroEval records from a JSONL file.

    Args:
        min_version:
            The minimum EuroEval version to include.
        min_number_of_model_records:
            The minimum number of records for a model to be included.
        banned_versions:
            A list of banned EuroEval versions to filter out.
        banned_model_patterns:
            A list of regex patterns to filter out models that should not be
            included.
        api_model_patterns:
            A list of regex patterns for API inference models.
        trained_from_scratch_patterns:
            A list of regex patterns for trained-from-scratch models.
        upload_to_bucket (optional):
            Whether to sync the processed per-model files to the Hugging Face
            results bucket. Defaults to False.
        conflict_resolver (optional):
            Terminal adapter for unresolved duplicate result identities. Defaults to
            automatic-only policy for library callers.

    Returns:
        Reconciled public records eligible for leaderboard generation. These are the
        exact winners written and optionally published; canary-excluded models are
        omitted, although their ordinary records remain in storage.
    """
    # Load raw results first so per-model files in RESULTS_DIR are synced.
    reset_leaderboard_result_filter()
    records = load_raw_results()
    records, canary_records, excluded_models, canary_report = (
        process_contamination_canaries(records=records)
    )
    if canary_report.get("status") == "unavailable":
        logger.warning(
            "Contamination-canary scoring unavailable at stage %s (%s).",
            canary_report.get("stage", "unknown"),
            canary_report.get("reason", "unknown_error"),
        )

    # Build the metadata cache from the synced per-model result files.
    cache = Cache.from_results_dir(RESULTS_DIR)
    num_raw_records = len(records)

    records = deduplicate_records(records=records)
    num_duplicates = num_raw_records - len(records)
    if num_duplicates:
        logger.info(f"Removed {num_duplicates:,} duplicates.")

    fixed_records = [
        fix_metadata(record=record)
        for record in tqdm(records, desc="Fixing metadata in records")
    ]

    processed_records = [
        record
        for record in fixed_records
        if record_is_valid(
            record=record,
            min_version=min_version,
            banned_versions=banned_versions,
            banned_model_patterns=banned_model_patterns,
            api_model_patterns=api_model_patterns,
        )
    ]

    # Remove records for models with few records
    counter = Counter([get_model_name(record) for record in processed_records])
    processed_records = [
        record
        for record in processed_records
        if counter[get_model_name(record)] >= min_number_of_model_records
    ]

    num_invalid_records = num_raw_records - num_duplicates - len(processed_records)
    if num_invalid_records > 0:
        logger.info(f"Removed {num_invalid_records:,} invalid records.")

    processed_records = [
        add_missing_entries(
            record=record,
            trained_from_scratch_patterns=trained_from_scratch_patterns,
            cache=cache,
        )
        for record in tqdm(processed_records, desc="Adding missing entries")
    ]
    upload_options: dict[str, t.Any] = {}
    if conflict_resolver is not None:
        upload_options["conflict_resolver"] = conflict_resolver
    reconciled_records = _upload_per_model_files(
        processed_records=processed_records,
        upload_to_bucket=upload_to_bucket,
        **upload_options,
    )
    # Completion-bearing evidence remains local/private and is never uploaded to
    # the public canonical results bucket.
    _upload_per_model_files(
        processed_records=canary_records,
        upload_to_bucket=False,
        local_only_label="private contamination-canary",
    )
    configure_leaderboard_result_filter(excluded_models=excluded_models)
    load_raw_results.cache_clear()
    logger.info("Cleared load_raw_results cache.")
    return [
        record
        for record in reconciled_records
        if identity_from_eee_record(record)[0] not in excluded_models
    ]


def _upload_per_model_files(
    processed_records: list[dict[str, t.Any]],
    upload_to_bucket: bool = False,
    local_only_label: str | None = None,
    conflict_resolver: Callable[..., ConflictDecision] | None = None,
) -> list[dict[str, t.Any]]:
    """Write one JSON file per logical result and optionally sync to the HF bucket.

    Each record is written to
    ``results/<sanitise(model_id)>/<dataset>__<split>__<shot>.json``.
    If two records resolve to the same path, the canonical result-conflict policy
    selects a compatible metadata superset or newer evaluation, and unresolved
    conflicts fail closed. Records with unresolvable model identities are dropped
    with a log line.

    Args:
        processed_records:
            The processed records to upload.
        upload_to_bucket (optional):
            Whether to sync the written files to the Hugging Face results
            bucket. If False, files are only written locally to RESULTS_DIR,
            and no HF_TOKEN is required. Defaults to False.
        local_only_label (optional):
            Safe description for local-only files, such as private canary
            results, included in the non-upload log message.
        conflict_resolver (optional):
            Terminal adapter for unresolved records; omitted callers fail closed.

    Returns:
        The final records selected for each written path, in path insertion order.

    Raises:
        RuntimeError:
            If HF_TOKEN is not set or bucket sync fails, when uploading.
    """
    hf_token = resolve_hf_token() if upload_to_bucket else None
    if upload_to_bucket and not hf_token:
        raise RuntimeError(
            "HF_TOKEN not set. Cannot upload results to Hugging Face bucket. "
            "Run 'hf auth login' or set the HF_TOKEN environment variable."
        )

    # The generation entry point uses the terminal adapter directly. Give it one
    # choice cache for this processing run so a model decision is reused.
    if conflict_resolver is not None:
        if conflict_resolver is resolve_result_interactively:
            model_choices: dict[str, ConflictChoice] = {}

            def batched_resolver(
                *,
                current: dict[str, t.Any],
                candidate: dict[str, t.Any],
                current_bytes: bytes | None = None,
                candidate_bytes: bytes | None = None,
            ) -> ConflictDecision:
                return resolve_result_interactively(
                    current=current,
                    candidate=candidate,
                    current_bytes=current_bytes,
                    candidate_bytes=candidate_bytes,
                    model_choices=model_choices,
                )

            conflict_resolver = batched_resolver

    # Capture canonical predecessors before deciding collisions or writing files.
    source_bytes = (
        _capture_local_sources(records=processed_records, results_dir=RESULTS_DIR)
        if upload_to_bucket
        else {}
    )

    # Resolve input contenders first; canonical predecessor reconciliation happens
    # only after each identity has one final payload.
    records_by_path: dict[Path, tuple[dict[str, t.Any], ResultIdentity]] = {}
    input_decisions: dict[ResultIdentity, list[ConflictDecision]] = {}
    dropped_count = 0

    for record in processed_records:
        try:
            identity = identity_from_eee_record(record)
        except ValueError as e:
            logger.warning(f"Dropping record with unresolvable identity: {e}")
            dropped_count += 1
            continue

        relative_path = record_relative_path(
            model_id=identity[0],
            dataset=identity[1],
            validation_split=identity[2],
            few_shot=identity[3],
        )

        if relative_path in records_by_path:
            # Path collision - check if identities match
            existing_record, existing_identity = records_by_path[relative_path]
            # raise_on_collision will raise ValueError if identities differ
            raise_on_collision(existing_identity, identity)
            # Identities match - keep the newer record based on version/timestamp
            selected, decision = _select_collision_record(
                existing_record=existing_record,
                candidate=record,
                identity=identity,
                conflict_resolver=conflict_resolver,
            )
            if decision is not None:
                input_decisions.setdefault(identity, []).append(decision)
            records_by_path[relative_path] = (selected, identity)
        else:
            records_by_path[relative_path] = (record, identity)

    hf_results_bucket = f"hf://buckets/{HF_RESULTS_BUCKET}"

    final_records: dict[Path, tuple[dict[str, t.Any], ResultIdentity]] = {}
    decisions: dict[ResultIdentity, ConflictDecision] = {}
    for relative_path, (record, identity) in records_by_path.items():
        source = source_bytes.get(identity)
        if source is not None:
            record, decision = _reconcile_canonical_record(
                candidate=record,
                source=source,
                identity=identity,
                prior_decisions=input_decisions.get(identity, []),
                conflict_resolver=conflict_resolver,
            )
            if decision is not None:
                decisions[identity] = decision
        final_records[relative_path] = (record, identity)

    payloads = [
        _serialise_record(record=record) for record, _ in final_records.values()
    ]
    publisher = None
    if upload_to_bucket:
        publisher = CanonicalPublisher(
            api=t.cast(BucketApi, HfApi()),
            token=hf_token or "",
            bucket=HF_RESULTS_BUCKET,
        )
        publisher.prepare(
            records=payloads, sources=source_bytes, decisions=list(decisions.values())
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written_files: list[tuple[str | Path | bytes, str]] = []

    logger.info("Writing result files...")
    for relative_path, (record, _) in final_records.items():
        result = _write_record_file(
            relative_path=relative_path, record=record, results_dir=RESULTS_DIR
        )
        if result is not None:
            written_files.append(result)

    if upload_to_bucket:
        assert publisher is not None
        with publisher.session() as session:
            session.publish(
                records=payloads,
                sources=source_bytes,
                decisions=list(decisions.values()),
            )
        logger.info(f"Published {len(payloads):,} result files to {hf_results_bucket}.")
    else:
        if local_only_label is not None:
            logger.info(
                f"Wrote {len(written_files):,} {local_only_label} result file(s) "
                "locally; not uploaded to the public results bucket."
            )
        else:
            logger.info(
                f"Skipped uploading {len(written_files):,} result file(s) to "
                f"{hf_results_bucket} (upload_to_bucket=False)."
            )

    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count:,} records with unresolvable identities.")

    return [record for record, _ in final_records.values()]


def _capture_local_sources(
    *, records: list[dict[str, t.Any]], results_dir: Path
) -> dict[ResultIdentity, bytes]:
    """Read each existing canonical predecessor once before local writes.

    Args:
        records: Processed records whose predecessor files may exist.
        results_dir: Local canonical results directory.

    Returns:
        Existing record bytes keyed by logical identity.
    """
    sources: dict[ResultIdentity, bytes] = {}
    for record in records:
        try:
            identity = identity_from_eee_record(record)
        except ValueError:
            continue
        path = results_dir / record_relative_path(
            model_id=identity[0],
            dataset=identity[1],
            validation_split=identity[2],
            few_shot=identity[3],
        )
        if identity not in sources and path.is_file():
            sources[identity] = path.read_bytes()
    return sources


def _reconcile_canonical_record(
    *,
    candidate: dict[str, t.Any],
    source: bytes,
    identity: ResultIdentity,
    prior_decisions: list[ConflictDecision],
    conflict_resolver: Callable[..., ConflictDecision] | None,
) -> tuple[dict[str, t.Any], ConflictDecision | None]:
    """Reconcile a final input winner against its exact local predecessor.

    Args:
        candidate: Final record selected from processed input contenders.
        source: Captured canonical predecessor bytes.
        identity: Record identity.
        prior_decisions: Earlier explicit contender choices for this identity.
        conflict_resolver: Optional terminal adapter for unresolved conflicts.

    Returns:
        The final payload and an incoming explicit choice that authorises it.
    """
    stored = json.loads(source)
    candidate_bytes = _serialise_record(record=candidate)
    decision = resolve_result_conflict(
        current=stored,
        candidate=candidate,
        current_bytes=source,
        candidate_bytes=candidate_bytes,
    )
    if not decision.needs_choice:
        if decision.selected is ConflictChoice.STORED:
            return stored, None
        return candidate, None

    expected = resolve_result_conflict(
        current=stored,
        candidate=candidate,
        current_bytes=source,
        candidate_bytes=candidate_bytes,
        choice=ConflictChoice.INCOMING,
    )
    for prior in reversed(prior_decisions):
        if (
            prior.reason.value == "explicit_choice"
            and prior.selected is ConflictChoice.INCOMING
            and prior.evidence == expected.evidence
        ):
            decision = prior
            break
    else:
        decision = _resolve_with_adapter(
            current=stored,
            candidate=candidate,
            current_bytes=source,
            candidate_bytes=candidate_bytes,
            identity=identity,
            conflict_resolver=conflict_resolver,
        )

    if decision.selected is ConflictChoice.STORED:
        return stored, None
    return candidate, decision


def _resolve_with_adapter(
    *,
    current: dict[str, t.Any],
    candidate: dict[str, t.Any],
    current_bytes: bytes,
    candidate_bytes: bytes,
    identity: ResultIdentity,
    conflict_resolver: Callable[..., ConflictDecision] | None,
) -> ConflictDecision:
    """Resolve an ambiguous exact record pair through the supplied adapter.

    Args:
        current: Stored-side record in this choice pair.
        candidate: Incoming-side record in this choice pair.
        current_bytes: Exact stored-side bytes.
        candidate_bytes: Exact incoming-side bytes.
        identity: Pair identity.
        conflict_resolver: Optional owner-choice adapter.

    Returns:
        A policy decision bound to this exact pair.

    Raises:
        UnresolvedResultConflictError: If no adapter is available.
    """
    if conflict_resolver is None:
        raise UnresolvedResultConflictError(
            f"Unresolved result conflict for {identity}: "
            f"{_describe_differences(current, candidate)}"
        )
    decision = conflict_resolver(
        current=current,
        candidate=candidate,
        current_bytes=current_bytes,
        candidate_bytes=candidate_bytes,
    )
    if (
        decision.evidence is None
        or decision.reason.value != "explicit_choice"
        or decision.selected not in (ConflictChoice.STORED, ConflictChoice.INCOMING)
    ):
        raise UnresolvedResultConflictError(
            f"Conflict resolver did not provide an explicit choice for {identity}"
        )
    expected = resolve_result_conflict(
        current=current,
        candidate=candidate,
        current_bytes=current_bytes,
        candidate_bytes=candidate_bytes,
        choice=decision.selected,
    )
    if decision.evidence != expected.evidence:
        raise UnresolvedResultConflictError(
            f"Conflict resolver choice is not bound to the exact pair for {identity}"
        )
    return decision


def _describe_differences(
    existing_record: dict[str, t.Any], candidate: dict[str, t.Any]
) -> str:
    """Describe conflicting fields without exposing full result records.

    Args:
        existing_record: First record for an identity.
        candidate: Conflicting record.

    Returns:
        Comma-separated names of differing top-level fields.
    """
    differences = sorted(
        key
        for key in existing_record.keys() | candidate.keys()
        if existing_record.get(key) != candidate.get(key)
    )
    return ", ".join(differences) or "unknown differences"


def _serialise_record(*, record: dict[str, t.Any]) -> bytes:
    """Encode records consistently for storage and byte-bound conflict proofs.

    Args:
        record: Processed result record.

    Returns:
        Canonical JSON bytes used for both local payloads and publication proofs.
    """
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _select_collision_record(
    *,
    existing_record: dict[str, t.Any],
    candidate: dict[str, t.Any],
    identity: ResultIdentity,
    conflict_resolver: Callable[..., ConflictDecision] | None,
) -> tuple[dict[str, t.Any], ConflictDecision | None]:
    """Select the winner between two processed contenders for one identity.

    Args:
        existing_record: Current processed winner.
        candidate: Next processed contender.
        identity: Shared canonical storage identity.
        conflict_resolver: Optional terminal adapter for unresolved conflicts.

    Returns:
        The selected record and any explicit choice evidence.
    """
    current_bytes = _serialise_record(record=existing_record)
    candidate_bytes = _serialise_record(record=candidate)
    decision = resolve_result_conflict(
        current=existing_record,
        candidate=candidate,
        current_bytes=current_bytes,
        candidate_bytes=candidate_bytes,
    )
    if decision.needs_choice:
        decision = _resolve_with_adapter(
            current=existing_record,
            candidate=candidate,
            current_bytes=current_bytes,
            candidate_bytes=candidate_bytes,
            identity=identity,
            conflict_resolver=conflict_resolver,
        )
    selected = (
        existing_record if decision.selected is ConflictChoice.STORED else candidate
    )
    return (selected, decision if decision.reason.value == "explicit_choice" else None)


def _write_record_file(
    relative_path: Path, record: dict[str, t.Any], results_dir: Path
) -> tuple[Path, str] | None:
    """Write a single record file if it doesn't exist or content differs.

    Args:
        relative_path:
            Relative path for the record file.
        record:
            The record dict to serialize.
        results_dir:
            The results directory.

    Returns:
        Tuple of (file_path, relative_path_str) if written, None if skipped.
    """
    file_path = results_dir / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Use canonical JSON serialization for consistent comparison
    canonical_content = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )

    # Only write if file doesn't exist or content differs
    if file_path.exists():
        try:
            existing_content = file_path.read_text(encoding="utf-8").strip()
            existing_record = json.loads(existing_content)
            canonical_existing = json.dumps(
                existing_record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            if canonical_existing == canonical_content:
                return None  # Skip unchanged files
        except (json.JSONDecodeError, OSError):
            pass  # If we can't parse existing, overwrite it

    file_path.write_text(canonical_content + "\n", encoding="utf-8")
    # Skip empty files
    if file_path.stat().st_size > 0:
        return (file_path, str(relative_path))
    return None
