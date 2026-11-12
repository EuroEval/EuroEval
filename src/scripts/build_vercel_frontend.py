"""Build the frontend against the version configured for Vercel Production."""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

POLICY_PATHS = (
    Path("api/worker/scope-policy.json"),
    Path("api/worker/_lib/scope-policy.generated.ts"),
)


def main() -> int:
    """Build with production-versioned policies while preserving source artifacts.

    Returns:
        Exit status from the Vercel build, or one when configuration is invalid.
    """
    environment_file = Path(".vercel/.env.production.local")
    try:
        subprocess.run(
            [
                "vercel",
                "env",
                "pull",
                str(environment_file),
                "--environment=production",
                "--yes",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        values = dotenv_values(environment_file)
        production_version = values.get("EUROEVAL_VERSION")
        requested_version = os.environ.get("EUROEVAL_VERSION")
        if not production_version:
            print(
                "Vercel Production EUROEVAL_VERSION is missing; "
                "frontend build stopped.",
                file=sys.stderr,
            )
            return 1
        if requested_version and requested_version != production_version:
            print(
                "EUROEVAL_VERSION override does not match Vercel Production; "
                "frontend build stopped.",
                file=sys.stderr,
            )
            return 1
        version = requested_version or production_version
    except (OSError, subprocess.CalledProcessError) as error:
        print(
            f"Vercel production configuration refresh failed: {error}", file=sys.stderr
        )
        return 1

    original_policies = [path.read_bytes() for path in POLICY_PATHS]
    try:
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "src/scripts/generate_volunteer_scope_policy.py",
                "--version",
                version,
            ],
            check=True,
        )
        build = subprocess.run(["vercel", "build", "--prod", "--yes"], check=False)
        return build.returncode
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Vercel frontend build preparation failed: {error}", file=sys.stderr)
        return 1
    finally:
        for path, contents in zip(POLICY_PATHS, original_policies, strict=True):
            path.write_bytes(contents)


if __name__ == "__main__":
    raise SystemExit(main())
