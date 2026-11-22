"""Collect CPU-only runtime inventories from an immutable native candidate image."""

import argparse
import json
import re
from pathlib import Path

from scripts.release_volunteer_worker import IMAGE, command, require

INVENTORY_PROGRAM = """
import importlib.metadata
import json
import subprocess
print(json.dumps({
    "python": sorted((d.metadata['Name'], d.version)
                     for d in importlib.metadata.distributions()),
    "dpkg": sorted(subprocess.check_output(
        ['dpkg-query', '-W', '-f=${Package}=${Version}\\n'],
        text=True).splitlines()),
}))
"""


def main() -> None:
    """Write attempt-qualified candidate metadata for the image workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--index-digest", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--run-attempt", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    output.write_text(json.dumps(collect(**args), sort_keys=True))


def collect(
    *,
    architecture: str,
    index_digest: str,
    source_sha: str,
    run_id: int,
    run_attempt: int,
) -> dict:
    """Execute only standard-library metadata queries; no inference or GPU access.

    Args:
        architecture: Native build architecture.
        index_digest: Original attestation-bearing candidate index.
        source_sha: Full trusted source SHA.
        run_id: GitHub workflow run identity.
        run_attempt: Exact workflow attempt.

    Returns:
        Small public runtime metadata artifact.
    """
    require(
        architecture in ("amd64", "arm64")
        and re.fullmatch(r"sha256:[0-9a-f]{64}", index_digest)
        and re.fullmatch(r"[0-9a-f]{40}", source_sha),
        "invalid inventory identity",
    )
    raw = command(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--platform",
            f"linux/{architecture}",
            "--entrypoint",
            "python",
            f"{IMAGE}@{index_digest}",
            "-c",
            INVENTORY_PROGRAM,
        ]
    )
    return {
        "architecture": architecture,
        "index_digest": index_digest,
        "source_sha": source_sha,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "inventory": json.loads(raw),
    }


if __name__ == "__main__":
    main()
