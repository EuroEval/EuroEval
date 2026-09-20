"""Score contamination-canary records in a benchmark JSONL file."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from euroeval.canary_evidence import CANARY_PRIVATE_DIR_ENV
from leaderboards.contamination_canary import (
    CANARY_KEY_ENV,
    CANARY_REPORT_PATH_ENV,
    is_canary_record,
    score_canary_records,
)
from leaderboards.jsonl_io import load_records_from_jsonl_files

_REPORT_LOGGER = logging.getLogger(f"{__name__}.report")
DEFAULT_RESULTS_PATH = Path("euroeval_benchmark_results.jsonl")
DEFAULT_KEY_PATH = Path("~/.config/euroeval/watermark-audit-v1.key")
DEFAULT_PRIVATE_DIR = Path("~/.local/share/euroeval/private-canary-v5")
DEFAULT_REPORT_PATH = Path("~/.local/state/euroeval/canary/report.json")


def main(argv: list[str] | None = None) -> int:
    """Score every canary record and emit the report as JSON.

    Args:
        argv:
            Optional command-line arguments.

    Returns:
        Zero after the report has been emitted. Invalid or missing input is handled by
        ``argparse`` as a non-zero command-line error.
    """
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    _configure_logging()
    results_path = arguments.results_file.expanduser()
    try:
        records = load_records_from_jsonl_files(paths=[results_path])
    except FileNotFoundError:
        parser.error(f"results file does not exist: {results_path}")
    except (OSError, ValueError) as error:
        parser.error(f"could not read valid JSONL from {results_path}: {error}")

    _configure_scorer(
        key_path=arguments.key,
        private_dir=arguments.private_dir,
        report_path=arguments.report_path,
    )
    canary_records = [record for record in records if is_canary_record(record=record)]
    report = score_canary_records(records=canary_records)
    _log_report(report=report)
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_file",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"benchmark results JSONL file (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=DEFAULT_KEY_PATH,
        help=f"private canary key (default: {DEFAULT_KEY_PATH})",
    )
    parser.add_argument(
        "--private-dir",
        type=Path,
        default=DEFAULT_PRIVATE_DIR,
        help=f"private canary records directory (default: {DEFAULT_PRIVATE_DIR})",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"owner-only report path (default: {DEFAULT_REPORT_PATH})",
    )
    return parser


def _configure_logging() -> None:
    """Route ordinary diagnostics to standard error."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr
    )


def _configure_scorer(*, key_path: Path, private_dir: Path, report_path: Path) -> None:
    """Configure the environment-backed private scorer from command-line paths."""
    os.environ[CANARY_KEY_ENV] = str(key_path.expanduser())
    os.environ[CANARY_PRIVATE_DIR_ENV] = str(private_dir.expanduser())
    os.environ[CANARY_REPORT_PATH_ENV] = str(report_path.expanduser())


def _log_report(*, report: dict[str, object]) -> None:
    """Render one stable, readable JSON report to standard output via logging."""
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _REPORT_LOGGER.setLevel(logging.INFO)
    _REPORT_LOGGER.propagate = False
    _REPORT_LOGGER.addHandler(handler)
    try:
        _REPORT_LOGGER.info(
            "%s", json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
        )
    finally:
        _REPORT_LOGGER.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
