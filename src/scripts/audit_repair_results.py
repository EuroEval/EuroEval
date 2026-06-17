"""Audit and safely repair EuroEval result stores to EEE format."""

from __future__ import annotations

import argparse
import collections.abc as c
import datetime as dt
import logging
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from leaderboards.constants import (
    BACKUPS_DIR,
    HF_RESULTS_BUCKET,
    RESULTS_DIR,
    RESULTS_PATH,
)
from leaderboards.eee_validation import (
    PreciousMetadataCache,
    build_precious_metadata_cache,
    dump_jsonl_records,
    is_eee_record,
    load_records_from_jsonl_files,
    load_records_from_results_archive,
    normalise_record_to_eee,
    validate_eee_record,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("audit_repair_results")


@dataclass
class StoreAudit:
    """Audit result for one result store."""

    name: str
    path: Path
    total_records: int = 0
    old_format_records: int = 0
    invalid_records: list[str] = field(default_factory=list)
    repairable_records: int = 0
    unrecoverable_records: int = 0

    @property
    def needs_repair(self) -> bool:
        """Return whether the store contains records needing repair."""
        return self.old_format_records > 0 or bool(self.invalid_records)


def main() -> None:
    """Run the audit and optional guarded local repair."""
    args = _parse_args()
    _validate_apply_guard(args=args)

    stores = _load_local_stores(
        results_dir=args.results_dir,
        results_archive=args.results_archive,
        include_backups=args.include_backups,
    )
    if args.audit_hf:
        stores.extend(_load_hf_store())

    all_records = [record for store in stores for record in store.records]
    precious_metadata_cache = build_precious_metadata_cache(records=all_records)

    audits = [
        _audit_store(store=store, precious_metadata_cache=precious_metadata_cache)
        for store in stores
    ]
    _log_audits(audits=audits)

    if not args.apply:
        logger.info(
            "Dry run only. Re-run with --apply and the guard flag to write fixes."
        )
        return

    for store in stores:
        if store.kind == "hf-bucket":
            logger.warning(
                "HF bucket repair is local-only: writing repaired mirror to %s, "
                "but not uploading to %s.",
                args.hf_output_dir,
                HF_RESULTS_BUCKET,
            )
            _write_results_dir(
                target=args.hf_output_dir,
                records=_repair_records(
                    records=store.records,
                    precious_metadata_cache=precious_metadata_cache,
                    source_name=store.name,
                ),
                backup=False,
            )
        elif store.kind == "results-dir" and store.path == args.results_dir:
            _write_results_dir(
                target=store.path,
                records=_repair_records(
                    records=store.records,
                    precious_metadata_cache=precious_metadata_cache,
                    source_name=store.name,
                ),
                backup=True,
            )
        elif store.kind == "archive" and store.path == args.results_archive:
            _write_results_archive(
                target=store.path,
                records=_repair_records(
                    records=store.records,
                    precious_metadata_cache=precious_metadata_cache,
                    source_name=store.name,
                ),
            )
        else:
            logger.info("Skipping backup/reference store %s during apply.", store.name)


@dataclass
class ResultStore:
    """A loaded result source."""

    name: str
    kind: str
    path: Path
    records: list[dict[str, object]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--results-archive", type=Path, default=RESULTS_PATH)
    parser.add_argument(
        "--include-backups",
        action="store_true",
        help="Use backup archives as read-only metadata recovery sources.",
    )
    parser.add_argument(
        "--audit-hf",
        action="store_true",
        help="Download EuroEval/results to a temporary directory and audit it.",
    )
    parser.add_argument(
        "--hf-output-dir",
        type=Path,
        default=Path("repaired_hf_results"),
        help="Local output directory for repaired HF bucket files; never uploaded.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write local repairs after taking backups. Dry-run is the default.",
    )
    parser.add_argument(
        "--i-understand-this-rewrites-local-results",
        action="store_true",
        help="Required together with --apply.",
    )
    return parser.parse_args()


def _validate_apply_guard(args: argparse.Namespace) -> None:
    if args.apply and not args.i_understand_this_rewrites_local_results:
        raise SystemExit(
            "--apply requires --i-understand-this-rewrites-local-results. "
            "The default dry run performs no writes."
        )


def _load_local_stores(
    results_dir: Path, results_archive: Path, include_backups: bool
) -> list[ResultStore]:
    stores: list[ResultStore] = []
    if results_dir.exists():
        stores.append(
            ResultStore(
                name="results/",
                kind="results-dir",
                path=results_dir,
                records=load_records_from_jsonl_files(
                    paths=sorted(results_dir.glob("*.jsonl"))
                ),
            )
        )
    else:
        logger.warning("Results directory not found: %s", results_dir)

    if results_archive.exists():
        stores.append(
            ResultStore(
                name="results.tar.gz",
                kind="archive",
                path=results_archive,
                records=load_records_from_results_archive(path=results_archive),
            )
        )
    else:
        logger.warning("Results archive not found: %s", results_archive)

    if include_backups and BACKUPS_DIR.exists():
        for backup_path in sorted(BACKUPS_DIR.glob("results*.tar.gz")):
            try:
                stores.append(
                    ResultStore(
                        name=f"backup:{backup_path.name}",
                        kind="backup-archive",
                        path=backup_path,
                        records=load_records_from_results_archive(path=backup_path),
                    )
                )
            except (FileNotFoundError, tarfile.TarError, ValueError) as exc:
                logger.warning("Skipping unreadable backup %s: %s", backup_path, exc)
    return stores


def _load_hf_store() -> list[ResultStore]:
    with tempfile.TemporaryDirectory(prefix="euroeval-hf-results-") as tmp_dir:
        target = Path(tmp_dir)
        logger.info("Syncing %s to temporary audit dir %s", HF_RESULTS_BUCKET, target)
        result = subprocess.run(
            ["hf", "sync", f"hf://buckets/{HF_RESULTS_BUCKET}/", str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("HF sync failed; skipping HF audit: %s", result.stderr)
            return []
        records = load_records_from_jsonl_files(paths=sorted(target.glob("*.jsonl")))
        return [
            ResultStore(
                name=f"hf://buckets/{HF_RESULTS_BUCKET}",
                kind="hf-bucket",
                path=target,
                records=records,
            )
        ]


def _audit_store(
    store: ResultStore, precious_metadata_cache: PreciousMetadataCache
) -> StoreAudit:
    audit = StoreAudit(
        name=store.name, path=store.path, total_records=len(store.records)
    )
    for idx, record in enumerate(store.records, start=1):
        if not is_eee_record(record=record):
            audit.old_format_records += 1
        repaired_record = normalise_record_to_eee(
            record=record, precious_metadata_cache=precious_metadata_cache
        )
        try:
            validate_eee_record(record=repaired_record, context=f"{store.name} #{idx}")
        except ValueError as exc:
            audit.invalid_records.append(str(exc))
            audit.unrecoverable_records += 1
        else:
            if repaired_record != record:
                audit.repairable_records += 1
    return audit


def _log_audits(audits: list[StoreAudit]) -> None:
    for audit in audits:
        logger.info(
            "%s: %,d records, %,d old-format, %,d repairable, %,d unrecoverable.",
            audit.name,
            audit.total_records,
            audit.old_format_records,
            audit.repairable_records,
            audit.unrecoverable_records,
        )
        for error in audit.invalid_records[:20]:
            logger.warning("  %s", error)
        if len(audit.invalid_records) > 20:
            logger.warning("  ... %,d more errors", len(audit.invalid_records) - 20)


def _repair_records(
    records: list[dict[str, object]],
    precious_metadata_cache: PreciousMetadataCache,
    source_name: str,
) -> list[dict[str, object]]:
    repaired_records = [
        normalise_record_to_eee(
            record=record, precious_metadata_cache=precious_metadata_cache
        )
        for record in records
    ]
    for idx, record in enumerate(repaired_records, start=1):
        validate_eee_record(record=record, context=f"{source_name} repaired #{idx}")
    return repaired_records


def _write_results_dir(
    target: Path, records: list[dict[str, object]], backup: bool
) -> None:
    if backup and target.exists():
        backup_target = _timestamped_backup_path(path=target)
        shutil.copytree(src=target, dst=backup_target)
        logger.info("Backed up %s to %s", target, backup_target)
    target.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        model_info = record.get("model_info", {})
        if not isinstance(model_info, c.Mapping):
            raise ValueError("Repaired EEE record lacks model_info.")
        typed_model_info = dict[str, object](model_info)
        model_id = str(
            typed_model_info.get("id") or typed_model_info.get("name") or "unknown"
        )
        filename = model_id.replace("/", "_") + ".jsonl"
        grouped.setdefault(filename, []).append(record)
    for filename, model_records in grouped.items():
        (target / filename).write_text(
            dump_jsonl_records(records=model_records), encoding="utf-8"
        )


def _write_results_archive(target: Path, records: list[dict[str, object]]) -> None:
    if target.exists():
        backup_target = _timestamped_backup_path(path=target)
        shutil.copy2(src=target, dst=backup_target)
        logger.info("Backed up %s to %s", target, backup_target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            content = dump_jsonl_records(records=records).encode(encoding="utf-8")
            tarinfo = tarfile.TarInfo(name="results/results.jsonl")
            tarinfo.size = len(content)
            with tempfile.TemporaryFile() as buffer:
                buffer.write(content)
                buffer.seek(0)
                tar.addfile(tarinfo=tarinfo, fileobj=buffer)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _timestamped_backup_path(path: Path) -> Path:
    stamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d_%H%M%S")
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        return BACKUPS_DIR / f"{path.name}_{stamp}"
    return BACKUPS_DIR / f"{path.name}.{stamp}.bak"


if __name__ == "__main__":
    main()
