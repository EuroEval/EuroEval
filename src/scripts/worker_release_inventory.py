"""Capture CPU-only installed package metadata for worker release provenance.

The CPU-only inventory steps in .github/workflows/worker-image.yaml query installed
Python package and dpkg versions inside each native, immutable candidate image via
Docker, without GPU access or model evaluation. The JSON artifact binds that inventory
to its architecture, image index digest, source SHA, and workflow run/attempt.

Installed versions are recorded because mutable apt repositories can change them
without source changes. Inventories describe each published image; they do not require
a reviewed baseline or physical GPU acceptance. This is maintainer release tooling,
not the volunteer startup script.
"""

import json
import re
from pathlib import Path

import click

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


@click.command(help=__doc__)
@click.option("--architecture", required=True)
@click.option("--index-digest", required=True)
@click.option("--source-sha", required=True)
@click.option("--run-id", type=int, required=True)
@click.option("--run-attempt", type=int, required=True)
@click.option(
    "--output", type=click.Path(path_type=Path, dir_okay=False), required=True
)
def main(
    architecture: str,
    index_digest: str,
    source_sha: str,
    run_id: int,
    run_attempt: int,
    output: Path,
) -> None:
    """Write attempt-qualified candidate metadata for the image workflow.

    Args:
        architecture: Native build architecture.
        index_digest: Original attestation-bearing candidate index.
        source_sha: Full trusted source SHA.
        run_id: GitHub workflow run identity.
        run_attempt: Exact workflow attempt.
        output: Destination JSON artifact.
    """
    inventory = collect(
        architecture=architecture,
        index_digest=index_digest,
        source_sha=source_sha,
        run_id=run_id,
        run_attempt=run_attempt,
    )
    output.write_text(json.dumps(inventory, sort_keys=True))


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
