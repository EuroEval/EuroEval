"""Analyse private grouped contamination-canary results."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import typing as t
from pathlib import Path

from euroeval.private_canary import (
    EXPOSURE_LEVELS,
    analyse_canary_results,
    repository_root,
    validate_arm_payload,
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments, analyse all five arms and write one private report.

    Raises:
        ValueError: If the output path is inside the repository or results are invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results_dir = args.results_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    repository = repository_root()
    if any(path.is_relative_to(repository) for path in (results_dir, output)):
        raise ValueError("canary results and reports must be outside the repository")
    arms: dict[int, list[dict[str, object]]] = {}
    fingerprints: set[str] = set()
    for dose in EXPOSURE_LEVELS:
        arm_path = results_dir / f"arm-{dose}.json"
        value = json.loads(arm_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("canary arm result is malformed")
        rows = validate_arm_payload(value, expected_dose=dose, filename=arm_path.name)
        arms[dose] = rows
        fingerprint = value["fingerprint"]
        assert isinstance(fingerprint, str)
        fingerprints.add(fingerprint)
    if len(fingerprints) != 1:
        raise ValueError("canary arms have different run fingerprints")
    row_ids = {t.cast(str, row["row_id"]) for row in arms[0]}
    if any(
        {t.cast(str, row["row_id"]) for row in rows} != row_ids
        for rows in arms.values()
    ):
        raise ValueError("canary arms have incomplete row IDs")
    report = analyse_canary_results(arms)
    report["fingerprint"] = fingerprints.pop()
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output.expanduser().resolve().parent, 0o700)
    _atomic_json(output, report)
    LOGGER.info("private canary report written")


def _atomic_json(path: Path, value: object) -> None:
    """Write a private report atomically."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == "__main__":
    main()
