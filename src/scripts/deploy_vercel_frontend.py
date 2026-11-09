"""Deploy a prebuilt frontend with narrowly scoped upload retries."""

import logging
import subprocess
import sys
import time

LOGGER = logging.getLogger(__name__)
DEPLOY_COMMAND = [
    "vercel",
    "deploy",
    "--prebuilt",
    "--prod",
    "--yes",
    "--non-interactive",
]
MAX_ATTEMPTS = 3
RETRY_DELAYS = (2, 4)
ERROR_SIGNATURE = "Unexpected token 'I'"
ERROR_BODY_SIGNATURE = '"Internal S"'
ERROR_SUFFIX_SIGNATURE = "is not valid JSON"


def main(argv: list[str] | None = None) -> int:
    """Run the prebuilt production deploy, retrying only the known upload failure.

    Args:
        argv (optional):
            Unused command-line arguments, retained for script testability.

    Returns:
        The Vercel CLI exit status.
    """
    del argv
    for attempt in range(MAX_ATTEMPTS):
        error_output, status = _run_deploy()
        if status == 0:
            return 0
        if not _is_known_upload_failure(output=error_output):
            return status
        if attempt == MAX_ATTEMPTS - 1:
            LOGGER.error(
                "Vercel upload failure persisted after %s attempts.", MAX_ATTEMPTS
            )
            return status
        delay = RETRY_DELAYS[attempt]
        LOGGER.warning("Known Vercel upload failure; retrying in %s seconds.", delay)
        time.sleep(delay)
    return 1


def _is_known_upload_failure(output: str) -> bool:
    """Identify only the verified plain-text upload JSON parse failure.

    Args:
        output:
            Bounded tail of streamed Vercel output.

    Returns:
        Whether output contains all distinctive parts of the known error.
    """
    return all(
        signature in output
        for signature in (ERROR_SIGNATURE, ERROR_BODY_SIGNATURE, ERROR_SUFFIX_SIGNATURE)
    )


def _run_deploy() -> tuple[str, int]:
    """Stream one Vercel deploy and retain only its bounded diagnostic tail.

    Returns:
        The final 1,024 characters of output and the CLI exit status.
    """
    process = subprocess.Popen(
        DEPLOY_COMMAND,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_tail = ""
    if process.stdout is not None:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            output_tail = (output_tail + line)[-1024:]
    return output_tail, process.wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
