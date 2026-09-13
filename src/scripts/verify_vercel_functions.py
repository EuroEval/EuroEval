"""Verify the functions emitted by a Vercel production build."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

LOGGER = logging.getLogger(__name__)
DEFAULT_FUNCTIONS_DIRECTORY = Path(".vercel/output/functions")
_NODE_RUNTIME = re.compile(r"nodejs(?:\d+\.x)?\Z")
EXPECTED_RUNTIMES = {
    "api/hall-of-fame": "edge",
    "api/issues": "edge",
    "api/submit-evaluation": "edge",
    "api/worker/auth/poll": "edge",
    "api/worker/auth/revoke": "edge",
    "api/worker/auth/start": "edge",
    "api/worker/claim": "edge",
    "api/worker/coordinator-lock": "edge",
    "api/worker/coordinator-release": "edge",
    "api/worker/coordinator-renew": "edge",
    "api/worker/heartbeat": "edge",
    "api/worker/promote": "edge",
    "api/worker/promotion-lock": "edge",
    "api/worker/promotion-reserve": "edge",
    "api/worker/release": "edge",
    "api/worker/finalise": "nodejs",
    "api/worker/result": "nodejs",
}


def main(argv: list[str] | None = None) -> int:
    """Verify a Vercel output directory and return a process status.

    Args:
        argv:
            Optional command-line arguments.

    Returns:
        Zero when verification succeeds, otherwise one.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--functions-directory",
        type=Path,
        default=DEFAULT_FUNCTIONS_DIRECTORY,
        help=f"Vercel functions directory (default: {DEFAULT_FUNCTIONS_DIRECTORY})",
    )
    arguments = parser.parse_args(argv)
    try:
        verify_vercel_functions(functions_directory=arguments.functions_directory)
    except VerificationError as error:
        LOGGER.error("%s", error)
        return 1
    LOGGER.info("Vercel function routes and runtimes are verified.")
    return 0


def verify_vercel_functions(
    functions_directory: Path = DEFAULT_FUNCTIONS_DIRECTORY,
) -> None:
    """Verify the exact routes and runtimes in a Vercel output directory.

    Args:
        functions_directory:
            Vercel Build Output API functions directory.

    Raises:
        VerificationError:
            If the output is missing, malformed, or differs from the allow-list.
    """
    if not functions_directory.is_dir():
        raise VerificationError(
            f"Vercel functions directory is missing: {functions_directory}"
        )

    function_directories = sorted(
        path for path in functions_directory.rglob("*.func") if path.is_dir()
    )
    for function_directory in function_directories:
        config_path = function_directory / ".vc-config.json"
        if not config_path.is_file():
            raise VerificationError(f"Missing function config: {config_path}")

    config_paths = sorted(functions_directory.rglob(".vc-config.json"))
    discovered: dict[str, tuple[str, Path]] = {}
    for config_path in config_paths:
        route = _route_from_config_path(
            config_path=config_path, functions_directory=functions_directory
        )
        if route in discovered:
            raise VerificationError(f"Duplicate function route: {route}")
        discovered[route] = (_read_runtime(config_path=config_path), config_path)

    expected_routes = set(EXPECTED_RUNTIMES)
    discovered_routes = set(discovered)
    missing = sorted(expected_routes - discovered_routes)
    extra = sorted(discovered_routes - expected_routes)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing routes: {', '.join(missing)}")
        if extra:
            details.append(f"extra routes: {', '.join(extra)}")
        message = "; ".join(details)
        raise VerificationError(f"Vercel function route mismatch ({message})")

    mismatches = []
    for route, (runtime, _) in sorted(discovered.items()):
        expected = EXPECTED_RUNTIMES[route]
        if not _runtime_matches(runtime=runtime, expected=expected):
            mismatches.append(f"{route}: expected {expected}, found {runtime}")
    if mismatches:
        message = "; ".join(mismatches)
        raise VerificationError(f"Vercel function runtime mismatch ({message})")


class VerificationError(ValueError):
    """Raised when a Vercel function output is not the expected deployment."""


def _read_runtime(*, config_path: Path) -> str:
    """Read and validate the runtime field from one Vercel config.

    Args:
        config_path:
            Path to the function config.

    Returns:
        The emitted runtime name.

    Raises:
        VerificationError:
            If the config is not valid JSON with a string runtime.
    """
    try:
        config = json.loads(
            config_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise VerificationError(f"Malformed function config: {config_path}") from error
    if not isinstance(config, dict) or not isinstance(config.get("runtime"), str):
        raise VerificationError(f"Malformed function config: {config_path}")
    return config["runtime"]


def _route_from_config_path(*, config_path: Path, functions_directory: Path) -> str:
    """Derive a route from a Vercel function config path.

    Args:
        config_path:
            Path to the function config.
        functions_directory:
            Root Vercel functions directory.

    Returns:
        The emitted function route.

    Raises:
        VerificationError:
            If the config is not directly inside a function directory.
    """
    relative = config_path.relative_to(functions_directory)
    if len(relative.parts) < 2 or not relative.parts[-2].endswith(".func"):
        raise VerificationError(f"Malformed function config path: {config_path}")
    function_name = relative.parts[-2][: -len(".func")]
    if not function_name:
        raise VerificationError(f"Malformed function config path: {config_path}")
    route_parts = (*relative.parts[:-2], function_name)
    return "/".join(route_parts)


def _runtime_matches(*, runtime: str, expected: str) -> bool:
    """Return whether an emitted runtime satisfies an expected runtime class."""
    if expected == "edge":
        return runtime == "edge"
    return bool(_NODE_RUNTIME.fullmatch(runtime))


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON object keys instead of silently choosing one.

    Args:
        pairs:
            JSON object pairs supplied by the decoder.

    Returns:
        The decoded object with unique keys.

    Raises:
        ValueError:
            If a key occurs more than once.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
