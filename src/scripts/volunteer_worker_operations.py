"""Safe maintainer operations for the EuroEval volunteer GPU broker.

This command deliberately keeps cloud mutations small, explicit, and injectable.  It
never reads dotenv files except for the explicitly confirmed local secret
initialisation operation.
"""

from __future__ import annotations

import argparse
import dataclasses
import http.client
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
import typing as t
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY = "EuroEval/EuroEval"
PRODUCTION_BASE_URL = "https://euroeval.com"
BROKER_BASE_URL = f"{PRODUCTION_BASE_URL}/api/worker"
RESULTS_BUCKET = "EuroEval/results"
IMAGE_REPOSITORY = "ghcr.io/euroeval/euroeval-worker"
LABELS = ("model evaluation request", "community-review-ready", "results-ready")
REQUIRED_ENVIRONMENT = (
    "GITHUB_TOKEN",
    "GITHUB_OAUTH_CLIENT_ID",
    "GITHUB_OAUTH_CLIENT_SECRET",
    "EUROEVAL_VERSION",
    "VOLUNTEER_WORKER_VERSION",
    "VOLUNTEER_WORKER_IMAGE_DIGEST",
    "VOLUNTEER_MARKER_SECRET",
    "UPSTASH_REDIS_REST_URL",
    "UPSTASH_REDIS_REST_TOKEN",
    "HF_STAGING_BUCKET",
    "HF_TOKEN",
    "WORKER_COORDINATOR_SECRET",
    "VOLUNTEER_PROMOTION_SECRET",
)
PUBLIC_CONFIG = {
    "EUROEVAL_VERSION",
    "VOLUNTEER_WORKER_VERSION",
    "VOLUNTEER_WORKER_IMAGE_DIGEST",
    "HF_STAGING_BUCKET",
}
VERCEL_KV_SOURCE_VARIABLES = ("KV_REST_API_URL", "KV_REST_API_TOKEN")
VERCEL_KV_OUTPUT_MARKER = "__EUROEVAL_VERCEL_KV_PAYLOAD__:"
VERCEL_KV_ALIASES = {
    "KV_REST_API_URL": "UPSTASH_REDIS_REST_URL",
    "KV_REST_API_TOKEN": "UPSTASH_REDIS_REST_TOKEN",
}
BASIC_ENVIRONMENT = {
    "PATH",
    "HOME",
    "USER",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "UV_CACHE_DIR",
    "UV_TOOL_DIR",
}
TOOL_AUTH_ENVIRONMENT = {
    "gh": {"GH_TOKEN", "GITHUB_TOKEN", "GH_HOST"},
    "hf": {"HF_TOKEN", "HF_HOME"},
    "vercel": {
        "VERCEL_TOKEN",
        "VERCEL_ORG_ID",
        "VERCEL_PROJECT_ID",
        "VERCEL_PROJECT_NAME",
    },
}
LOCAL_SECRET_NAMES = (
    "VOLUNTEER_MARKER_SECRET",
    "WORKER_COORDINATOR_SECRET",
    "VOLUNTEER_PROMOTION_SECRET",
)
ROUTES = (
    "auth/start",
    "auth/poll",
    "auth/revoke",
    "claim",
    "heartbeat",
    "result",
    "finalise",
    "release",
    "coordinator-lock",
    "coordinator-renew",
    "coordinator-release",
    "promotion-lock",
    "promotion-reserve",
    "promote",
)


def main(argv: list[str] | None = None) -> int:
    """Run a volunteer worker operations command.

    Returns:
        Process exit status.
    """
    arguments = parse_arguments(argv)
    environment = dict(os.environ)
    if arguments.local_secrets and arguments.command != "apply":
        print("--local-secrets requires apply; no changes were made.")
        return 2
    if arguments.env_file != Path(".env") and not arguments.local_secrets:
        print("--env-file requires --local-secrets; no changes were made.")
        return 2
    if arguments.reuse_vercel_kv and arguments.command != "apply":
        print("--reuse-vercel-kv requires apply; no changes were made.")
        return 2
    if arguments.command == "plan":
        print_plan(environment=environment)
        return 0
    if arguments.command == "check":
        diagnostics = check(
            environment=environment,
            components=selected_components(arguments),
            route_probes=arguments.routes,
        )
        print_diagnostics(diagnostics)
        return int(any(item.failed for item in diagnostics))
    if arguments.command == "apply":
        return apply(
            environment=environment,
            components=selected_components(arguments, explicit=True),
            confirmed=arguments.yes,
            hf_region=arguments.hf_region,
            reuse_vercel_kv=arguments.reuse_vercel_kv,
            local_secrets=arguments.local_secrets,
            env_file=arguments.env_file,
        )
    diagnostics = smoke(
        base_url=arguments.base_url,
        routes=arguments.route or list(ROUTES),
        environment=environment,
    )
    print_diagnostics(diagnostics)
    return int(any(item.failed for item in diagnostics))


def apply(
    *,
    environment: dict[str, str],
    components: set[str],
    confirmed: bool,
    hf_region: str = "eu",
    reuse_vercel_kv: bool = False,
    local_secrets: bool = False,
    env_file: Path = Path(".env"),
) -> int:
    """Apply only explicitly confirmed, narrowly scoped setup.

    Returns:
        Process exit status.
    """
    if local_secrets:
        if components or reuse_vercel_kv:
            print("--local-secrets cannot be combined with another apply scope.")
            return 2
        if not confirmed:
            print("--local-secrets requires --yes; no changes were made.")
            return 2
        diagnostics = apply_local_secrets(env_file=env_file)
        print_diagnostics(diagnostics)
        return int(any(item.failed for item in diagnostics))
    if reuse_vercel_kv:
        if components:
            print("--reuse-vercel-kv cannot be combined with another apply scope.")
            return 2
        if not confirmed:
            print("--reuse-vercel-kv requires --yes; no changes were made.")
            return 2
        diagnostics = apply_reuse_vercel_kv(environment=environment)
        print_diagnostics(diagnostics)
        return int(any(item.failed for item in diagnostics))
    if not components or not components <= {"github", "hf", "vercel"}:
        print(
            "apply requires explicit --github, --hf, or --vercel, or "
            "--reuse-vercel-kv (no default/all path)."
        )
        return 2
    if not confirmed:
        print("apply requires --yes; no changes were made.")
        return 2
    diagnostics: list[Diagnostic] = []
    if "github" in components:
        diagnostics.extend(apply_github())
    if "hf" in components:
        diagnostics.extend(apply_hf(environment=environment, region=hf_region))
    if "vercel" in components:
        diagnostics.extend(apply_vercel(environment=environment))
    print_diagnostics(diagnostics)
    return int(any(item.failed for item in diagnostics))


_LOCAL_ENV_ASSIGNMENT = re.compile(
    r"[ \t]*(?:export[ \t]+)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)[ \t]*="
    r"(?P<value>.*?)(?:\r?\n)?\Z"
)


def _json_object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Decode a JSON object while rejecting duplicate keys.

    Args:
        pairs:
            Object key-value pairs from the JSON decoder.

    Returns:
        The decoded object.

    Raises:
        ValueError:
            If an object key occurs more than once.
    """
    decoded: dict[str, object] = {}
    for name, value in pairs:
        if name in decoded:
            raise ValueError("duplicate JSON key")
        decoded[name] = value
    return decoded


@dataclasses.dataclass(frozen=True)
class Diagnostic:
    """One concise, classified diagnostic."""

    component: str
    category: str
    message: str
    failed: bool = False


def apply_github() -> list[Diagnostic]:
    """Create only missing canonical labels.

    Returns:
        GitHub diagnostics.
    """
    listing = run_command(["gh", "api", f"repos/{REPOSITORY}/labels", "--paginate"])
    if listing.returncode:
        return [Diagnostic("github", "service failure", "cannot inspect labels", True)]
    names = _json_names(listing.stdout)
    diagnostics = []
    for label in LABELS:
        if label in names:
            diagnostics.append(
                Diagnostic("github", "ok", f"label {label} already exists")
            )
            continue
        created = run_command(["gh", "label", "create", label, "--repo", REPOSITORY])
        creation_state = "created" if created.returncode == 0 else "creation failed"
        diagnostics.append(
            Diagnostic(
                "github",
                "ok" if created.returncode == 0 else "service failure",
                f"label {label}: {creation_state}",
                created.returncode != 0,
            )
        )
    return diagnostics


def _json_names(value: str) -> set[str]:
    """Extract names from a GitHub JSON response.

    Returns:
        Names found in the response.
    """
    return {str(item.get("name", "")) for item in _json_list(value)}


def _json_list(value: str) -> list[dict[str, object]]:
    """Decode a JSON list of objects.

    Returns:
        Decoded object list.
    """
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []
    if isinstance(decoded, dict) and isinstance(decoded.get("envs"), list):
        decoded = decoded["envs"]
    return (
        [item for item in decoded if isinstance(item, dict)]
        if isinstance(decoded, list)
        else []
    )


@dataclasses.dataclass(frozen=True)
class CommandResult:
    """Captured result of an external command."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


def run_command(
    command: list[str],
    *,
    input_text: str | None = None,
    environment: dict[str, str] | None = None,
) -> CommandResult:
    """Run a tool with a least-privilege environment and capture its output.

    ``environment`` overrides the process environment before the allowlist is applied.

    Returns:
        Captured command result.
    """
    source = {
        name: value for name, value in os.environ.items() if name in BASIC_ENVIRONMENT
    }
    if environment is None:
        source.update(os.environ)
    else:
        source.update(environment)
    tool = command[0] if command else ""
    if tool == "uv" and len(command) > 2:
        tool = command[2]
    allowed = BASIC_ENVIRONMENT | TOOL_AUTH_ENVIRONMENT.get(tool, set())
    child_environment = {name: source[name] for name in allowed if source.get(name)}
    if "DOCKER_CONFIG" in source and tool == "docker":
        child_environment["DOCKER_CONFIG"] = source["DOCKER_CONFIG"]
    try:
        completed = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
            env=child_environment,
        )
    except OSError:
        return CommandResult(127)
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def apply_hf(*, environment: dict[str, str], region: str = "eu") -> list[Diagnostic]:
    """Create a missing private bucket in the selected region, or verify a bucket.

    Returns:
        Hugging Face diagnostics.
    """
    bucket = environment.get("HF_STAGING_BUCKET")
    if not bucket or not environment.get("HF_TOKEN"):
        return [
            Diagnostic(
                "hf",
                "missing config",
                "HF_TOKEN and HF_STAGING_BUCKET are required",
                True,
            )
        ]
    auth = run_command(["uv", "run", "hf", "auth", "whoami"])
    if auth.returncode:
        return [Diagnostic("hf", "auth", "Hugging Face authentication failed", True)]
    info_command = ["uv", "run", "hf", "buckets", "info", bucket, "--json"]
    current = run_command(info_command)
    if current.returncode:
        created = run_command(
            [
                "uv",
                "run",
                "hf",
                "buckets",
                "create",
                bucket,
                "--private",
                "--region",
                region,
                "--exist-ok",
            ]
        )
        if created.returncode:
            category, message = _classify_hf_failure(created)
            return [Diagnostic("hf", category, message, True)]
        current = run_command(info_command)
    data = _json_object(current.stdout)
    if current.returncode or not data:
        if current.returncode:
            category, message = _classify_hf_failure(current)
            return [Diagnostic("hf", category, message, True)]
        return [
            Diagnostic("hf", "service failure", "bucket metadata is malformed", True)
        ]
    return _hf_metadata_diagnostics(data)


def _classify_hf_failure(result: CommandResult) -> tuple[str, str]:
    """Classify HF CLI failures without exposing command output.

    Returns:
        Failure category and safe message.
    """
    text = f"{result.stdout} {result.stderr}".lower()
    missing = any(value in text for value in ("404", "not found", "does not exist"))
    if any(value in text for value in ("401", "403", "unauthor")):
        return "auth", "Hugging Face bucket operation is not authorised"
    if missing and any(value in text for value in ("private", "permission", "auth")):
        return (
            "missing or inaccessible",
            "configured staging bucket is missing or inaccessible",
        )
    if any(value in text for value in ("auth", "token", "permission")):
        return "auth", "Hugging Face bucket operation is not authorised"
    if any(value in text for value in ("timeout", "network", "connection", "dns")):
        return "network", "Hugging Face bucket operation could not be reached"
    if missing:
        return "missing", "configured staging bucket does not exist"
    return "service failure", "Hugging Face bucket operation failed"


def _hf_metadata_diagnostics(data: dict[str, object]) -> list[Diagnostic]:
    """Validate bucket visibility and report the region verification limitation.

    Returns:
        Hugging Face metadata diagnostics.
    """
    visibility = data.get("visibility")
    private = data.get("private") is True or visibility == "private"
    if visibility == "public" or data.get("private") is False:
        privacy = Diagnostic("hf", "drift", "staging bucket exists but is public", True)
    elif private:
        privacy = Diagnostic("hf", "ok", "staging bucket privacy is private")
    else:
        privacy = Diagnostic(
            "hf", "service failure", "bucket privacy cannot be verified", True
        )
    region_diagnostic = Diagnostic(
        "hf",
        "manual",
        "existing bucket region cannot be verified from metadata; confirm it manually",
    )
    return [privacy, region_diagnostic]


def _json_object(value: str) -> dict[str, object]:
    """Decode a JSON object, returning an empty object on command failure.

    Returns:
        Decoded object or an empty object.
    """
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def apply_local_secrets(*, env_file: Path = Path(".env")) -> list[Diagnostic]:
    """Initialise missing broker secrets in a local dotenv file.

    Args:
        env_file:
            The local dotenv file to update.

    Returns:
        Safe diagnostics describing generated and preserved variable names.
    """
    try:
        env_file = Path(env_file)
        original_stat = _validate_local_secret_path(env_file)
        original_content = _read_local_secret_file(
            env_file=env_file, expected_stat=original_stat
        )
        present = _local_secret_values(original_content)
        preserved = [name for name in LOCAL_SECRET_NAMES if present.get(name, False)]
        generated = [name for name in LOCAL_SECRET_NAMES if name not in preserved]

        if generated:
            additions = "".join(
                f"{name}={secrets.token_urlsafe(32)}\n" for name in generated
            ).encode()
            separator = (
                b"\n"
                if original_content and not original_content.endswith(b"\n")
                else b""
            )
            _atomic_write_local_secret_file(
                env_file=env_file,
                content=original_content + separator + additions,
                expected_stat=original_stat,
            )
        elif original_stat is not None and stat.S_IMODE(original_stat.st_mode) != 0o600:
            _tighten_local_secret_mode(env_file=env_file, expected_stat=original_stat)

        generated_names = ", ".join(generated) or "none"
        preserved_names = ", ".join(preserved) or "none"
        return [
            Diagnostic(
                "local-secrets",
                "ok",
                f"generated: {generated_names}; preserved: {preserved_names}; "
                "permissions: 0600",
            )
        ]
    except (OSError, UnicodeError, ValueError):
        return [
            Diagnostic(
                "local-secrets",
                "safety",
                "local secret file was rejected or could not be updated",
                True,
            )
        ]


def _atomic_write_local_secret_file(
    *, env_file: Path, content: bytes, expected_stat: os.stat_result | None
) -> None:
    """Replace a local dotenv file atomically and durably.

    Raises:
        ValueError:
            If the destination changes before the replacement.
    """
    if not _target_matches(env_file, expected_stat):
        raise ValueError("dotenv destination changed during validation")

    temporary_path: Path | None = None
    file_descriptor = -1
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{env_file.name}.", dir=env_file.parent
        )
        temporary_path = Path(temporary_name)
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "wb") as stream:
            file_descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if not _target_matches(env_file, expected_stat):
            raise ValueError("dotenv destination changed during validation")
        os.replace(temporary_path, env_file)
        temporary_path = None
        directory_descriptor = os.open(
            env_file.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _target_matches(env_file: Path, expected_stat: os.stat_result | None) -> bool:
    """Check that the destination still names the file that was inspected.

    Returns:
        Whether the destination still has the expected identity.
    """
    try:
        current_stat = os.lstat(env_file)
    except FileNotFoundError:
        return expected_stat is None
    return bool(
        expected_stat is not None
        and stat.S_ISREG(current_stat.st_mode)
        and current_stat.st_dev == expected_stat.st_dev
        and current_stat.st_ino == expected_stat.st_ino
    )


def _local_secret_values(content: bytes) -> dict[str, bool]:
    """Return whether each managed dotenv variable has a non-empty value."""
    text = content.decode("utf-8")
    values = {name: False for name in LOCAL_SECRET_NAMES}
    for line in text.splitlines(keepends=True):
        match = _LOCAL_ENV_ASSIGNMENT.fullmatch(line)
        if match and match.group("name") in values:
            values[match.group("name")] |= _non_empty_env_value(match.group("value"))
    return values


def _non_empty_env_value(value: str) -> bool:
    """Determine emptiness without expanding or evaluating dotenv syntax.

    Returns:
        Whether the assignment has a non-empty value.
    """
    stripped = value.strip()
    if not stripped or stripped.startswith("#"):
        return False
    if stripped[0] not in ('"', "'"):
        return True
    quote = stripped[0]
    escaped = False
    for index, character in enumerate(stripped[1:], start=1):
        if character == quote and not escaped:
            return bool(stripped[1:index])
        escaped = character == "\\" and not escaped
    return True


def _read_local_secret_file(
    *, env_file: Path, expected_stat: os.stat_result | None
) -> bytes:
    """Read the dotenv file while keeping the checked inode fixed.

    Returns:
        The original file bytes, or empty bytes for a new file.

    Raises:
        ValueError:
            If the destination changed while it was being opened.
    """
    if expected_stat is None:
        return b""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    file_descriptor = os.open(env_file, os.O_RDONLY | nofollow)
    try:
        opened_stat = os.fstat(file_descriptor)
        if (
            opened_stat.st_dev != expected_stat.st_dev
            or opened_stat.st_ino != expected_stat.st_ino
            or not stat.S_ISREG(opened_stat.st_mode)
        ):
            raise ValueError("dotenv file changed during validation")
        with os.fdopen(file_descriptor, "rb") as stream:
            file_descriptor = -1
            return stream.read()
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)


def _tighten_local_secret_mode(
    *, env_file: Path, expected_stat: os.stat_result
) -> None:
    """Tighten an unchanged existing dotenv file without rewriting its content.

    Raises:
        ValueError:
            If the destination changes before its mode is tightened.
    """
    if not _target_matches(env_file, expected_stat):
        raise ValueError("dotenv destination changed during validation")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    file_descriptor = os.open(env_file, os.O_WRONLY | nofollow)
    try:
        current_stat = os.fstat(file_descriptor)
        if (
            current_stat.st_dev != expected_stat.st_dev
            or current_stat.st_ino != expected_stat.st_ino
        ):
            raise ValueError("dotenv destination changed during validation")
        os.fchmod(file_descriptor, 0o600)
    finally:
        os.close(file_descriptor)


def _validate_local_secret_path(env_file: Path) -> os.stat_result | None:
    """Validate the dotenv path without following a symlink.

    Returns:
        The existing file's metadata, or ``None`` when it does not exist.

    Raises:
        ValueError:
            If the parent or destination is unsafe.
    """
    parent_stat = os.lstat(env_file.parent)
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise ValueError("dotenv parent is not a directory")
    if stat.S_ISLNK(parent_stat.st_mode):
        raise ValueError("dotenv parent is a symlink")
    if parent_stat.st_uid != os.getuid() or stat.S_IMODE(parent_stat.st_mode) & 0o022:
        raise ValueError("dotenv parent is unsafe")

    try:
        file_stat = os.lstat(env_file)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(file_stat.st_mode) or stat.S_ISLNK(file_stat.st_mode):
        raise ValueError("dotenv path is not a regular file")
    if file_stat.st_uid != os.getuid() or file_stat.st_nlink != 1:
        raise ValueError("dotenv file is unsafe")
    return file_stat


def apply_reuse_vercel_kv(*, environment: dict[str, str]) -> list[Diagnostic]:
    """Copy the existing Vercel KV credentials to broker aliases.

    The source values are read by a short-lived child of ``vercel env run`` and are
    passed to Vercel's add command through stdin only. Existing source variables are
    never modified.

    Returns:
        Vercel diagnostics.
    """
    identity, verified = _check_vercel_project(environment=environment)
    if not verified:
        return identity
    vercel_environment = _vercel_command_environment(environment)
    source = run_command(_vercel_kv_read_command(), environment=vercel_environment)
    if source.returncode:
        return [
            Diagnostic(
                "vercel", "service failure", "source KV variables: read failed", True
            )
        ]
    values = _vercel_kv_values(source.stdout)
    missing = [name for name in VERCEL_KV_SOURCE_VARIABLES if name not in values]
    if missing:
        return [
            Diagnostic(
                "vercel",
                "missing config",
                "source KV variables missing: " + ", ".join(missing),
                True,
            )
        ]

    diagnostics: list[Diagnostic] = []
    for source_name in VERCEL_KV_SOURCE_VARIABLES:
        alias = VERCEL_KV_ALIASES[source_name]
        added = run_command(
            [
                "vercel",
                "env",
                "add",
                alias,
                "production",
                "--force",
                "--yes",
                "--sensitive",
            ],
            input_text=values[source_name] + "\n",
            environment=vercel_environment,
        )
        diagnostics.append(
            Diagnostic(
                "vercel",
                "ok" if added.returncode == 0 else "service failure",
                f"Production variable {alias}: "
                f"{'updated' if added.returncode == 0 else 'update failed'}",
                added.returncode != 0,
            )
        )
        if added.returncode:
            return diagnostics

    return diagnostics + _check_vercel_kv_aliases(environment=vercel_environment)


def _check_vercel_kv_aliases(*, environment: dict[str, str]) -> list[Diagnostic]:
    """Verify only the two broker aliases and never inspect their values.

    Returns:
        Diagnostics for the two broker aliases.
    """
    result = run_command(
        ["vercel", "env", "ls", "production", "--format=json"], environment=environment
    )
    if result.returncode:
        return [
            Diagnostic(
                "vercel",
                "service failure",
                "broker KV aliases: metadata read failed",
                True,
            )
        ]
    metadata = _json_list(result.stdout)
    by_name = {str(item.get("key", item.get("name", ""))): item for item in metadata}
    diagnostics: list[Diagnostic] = []
    for alias in VERCEL_KV_ALIASES.values():
        item = by_name.get(alias)
        target = item.get("target", item.get("targets")) if item else None
        variable_type = str(item.get("type", "")) if item else ""
        valid = (
            item is not None
            and isinstance(target, list)
            and "production" in target
            and variable_type in {"sensitive", "secret"}
        )
        diagnostics.append(
            Diagnostic(
                "vercel",
                "ok" if valid else "drift",
                f"Production variable {alias}: "
                f"{'present' if valid else 'missing or wrong type/target'}",
                not valid,
            )
        )
    return diagnostics


def _check_vercel_project(
    *, environment: dict[str, str]
) -> tuple[list[Diagnostic], bool]:
    """Verify the local link and remote project before any Vercel mutation.

    Returns:
        Diagnostics and whether the identity is verified.
    """
    link_path = Path(".vercel/project.json")
    try:
        link = _json_object(link_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError):
        link = {}
    linked_project = link.get("projectId")
    linked_scope = link.get("orgId")
    linked_name = link.get("projectName")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (linked_project, linked_scope, linked_name)
    ):
        return [
            Diagnostic(
                "vercel",
                "missing config",
                "local Vercel link is missing projectId, orgId, or projectName",
                True,
            )
        ], False
    assert isinstance(linked_project, str)
    assert isinstance(linked_scope, str)
    assert isinstance(linked_name, str)
    for name, linked_value in (
        ("VERCEL_PROJECT_ID", linked_project),
        ("VERCEL_ORG_ID", linked_scope),
        ("VERCEL_PROJECT_NAME", linked_name),
    ):
        supplied_value = environment.get(name)
        if supplied_value and supplied_value != linked_value:
            return [
                Diagnostic(
                    "vercel", "drift", f"{name} disagrees with the local link", True
                )
            ], False
    project = run_command(
        ["vercel", "project", "inspect", "--format", "json"],
        environment=_vercel_command_environment(environment),
    )
    if project.returncode:
        return [
            Diagnostic(
                "vercel", "auth", "linked Vercel project cannot be inspected", True
            )
        ], False
    data = _json_object(project.stdout)
    actual_project = str(data.get("id", data.get("projectId", "")))
    actual_name = str(data.get("name", data.get("projectName", "")))
    remote_scopes = [
        str(data[name]) for name in ("accountId", "teamId", "orgId") if data.get(name)
    ]
    identity_ok = (
        actual_project == linked_project
        and actual_name == linked_name
        and all(scope == linked_scope for scope in remote_scopes)
    )
    if not identity_ok:
        return [
            Diagnostic(
                "vercel",
                "drift",
                "linked Vercel project identity or scope does not match",
                True,
            )
        ], False
    return [
        Diagnostic("vercel", "ok", "linked Vercel project identity and scope verified")
    ], True


def _vercel_command_environment(environment: dict[str, str]) -> dict[str, str]:
    """Keep Vercel subprocesses free of unrelated injected application secrets.

    Returns:
        Allowlisted environment values for Vercel commands.
    """
    allowed = BASIC_ENVIRONMENT | TOOL_AUTH_ENVIRONMENT["vercel"]
    return {name: environment[name] for name in allowed if environment.get(name)}


def _vercel_kv_read_command() -> list[str]:
    """Build the value-free command used inside Vercel's production environment.

    Returns:
        Command arguments for the nested Vercel environment process.
    """
    names = ", ".join(repr(name) for name in VERCEL_KV_SOURCE_VARIABLES)
    script = (
        "import json, os; "
        f"print({VERCEL_KV_OUTPUT_MARKER!r} + "
        f"json.dumps({{name: os.environ.get(name) for name in ({names},)}}, "
        "separators=(',', ':')))"
    )
    return [
        "vercel",
        "env",
        "run",
        "--environment",
        "production",
        "--",
        sys.executable,
        "-c",
        script,
    ]


def _vercel_kv_values(output: str) -> dict[str, str]:
    """Decode the uniquely marked, non-empty source values.

    Returns:
        Non-empty source values, keyed by their Vercel variable names.
    """
    marked_payloads = [
        line[len(VERCEL_KV_OUTPUT_MARKER) :]
        for line in output.splitlines()
        if line.startswith(VERCEL_KV_OUTPUT_MARKER)
    ]
    if len(marked_payloads) != 1:
        return {}
    try:
        decoded = json.loads(
            marked_payloads[0], object_pairs_hook=_json_object_without_duplicates
        )
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(decoded, dict):
        return {}
    return {
        name: value
        for name, value in decoded.items()
        if name in VERCEL_KV_SOURCE_VARIABLES
        and isinstance(value, str)
        and value.strip()
    }


def apply_vercel(*, environment: dict[str, str]) -> list[Diagnostic]:
    """Add or update Production variables atomically using stdin.

    Returns:
        Vercel diagnostics.
    """
    identity, verified = _check_vercel_project(environment=environment)
    if not verified:
        return identity
    euroeval_version, worker_version = source_versions()
    values = dict(environment)
    values.setdefault("EUROEVAL_VERSION", euroeval_version)
    values.setdefault("VOLUNTEER_WORKER_VERSION", worker_version)
    missing = [name for name in REQUIRED_ENVIRONMENT if not values.get(name)]
    if missing:
        return [
            Diagnostic(
                "vercel",
                "missing config",
                "missing required values: " + ", ".join(missing),
                True,
            )
        ]
    diagnostics: list[Diagnostic] = []
    for name in REQUIRED_ENVIRONMENT:
        variable_type = "plain" if name in PUBLIC_CONFIG else "sensitive"
        added = run_command(
            [
                "vercel",
                "env",
                "add",
                name,
                "production",
                "--force",
                "--yes",
                "--type",
                variable_type,
            ],
            input_text=values[name] + "\n",
            environment=environment,
        )
        diagnostics.append(
            Diagnostic(
                "vercel",
                "ok" if added.returncode == 0 else "service failure",
                f"Production variable {name}: "
                f"{'updated' if added.returncode == 0 else 'update failed'}",
                added.returncode != 0,
            )
        )
    if any(item.failed for item in diagnostics):
        return diagnostics
    verification = check_vercel(environment=environment)
    return diagnostics + verification


def check_vercel(*, environment: dict[str, str]) -> list[Diagnostic]:
    """Check the exact linked Vercel project and Production variable metadata.

    Returns:
        Vercel diagnostics.
    """
    result, identity_ok = _check_vercel_project(environment=environment)
    if not identity_ok:
        return result
    variables = run_command(["vercel", "env", "ls", "production", "--format=json"])
    if variables.returncode:
        result.append(
            Diagnostic(
                "vercel",
                "service failure",
                "Production environment metadata cannot be read",
                True,
            )
        )
        return result
    try:
        decoded_metadata = json.loads(variables.stdout)
    except json.JSONDecodeError:
        result.append(
            Diagnostic(
                "vercel",
                "service failure",
                "Production environment metadata is malformed",
                True,
            )
        )
        return result
    if not (
        isinstance(decoded_metadata, list)
        or isinstance(decoded_metadata, dict)
        and isinstance(decoded_metadata.get("envs"), list)
    ):
        result.append(
            Diagnostic(
                "vercel",
                "service failure",
                "Production environment metadata is malformed",
                True,
            )
        )
        return result
    metadata = _json_list(variables.stdout)
    by_name = {str(item.get("key", item.get("name", ""))): item for item in metadata}
    for name in REQUIRED_ENVIRONMENT:
        item = by_name.get(name)
        target = item.get("target", item.get("targets")) if item else None
        variable_type = str(item.get("type", "")) if item else ""
        target_ok = isinstance(target, list) and "production" in target
        type_ok = (
            variable_type == "plain"
            if name in PUBLIC_CONFIG
            else variable_type in {"sensitive", "secret"}
        )
        valid = item is not None and target_ok and type_ok
        variable_state = "present" if valid else "missing or wrong type/target"
        result.append(
            Diagnostic(
                "vercel",
                "ok" if valid else "drift",
                f"Production variable {name}: {variable_state}",
                not valid,
            )
        )
    return result


def source_versions() -> tuple[str, str]:
    """Read EuroEval and worker versions from tracked source files.

    Returns:
        EuroEval and worker versions.

    Raises:
        RuntimeError:
            If the worker source version is missing.
    """
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project_metadata = t.cast(dict[str, object], project["project"])
    euroeval = str(project_metadata["version"])
    worker_source = Path("src/euroeval_worker/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)', worker_source)
    if match is None:
        raise RuntimeError("worker source version is missing")
    return euroeval, match.group(1)


def check(
    *, environment: dict[str, str], components: set[str], route_probes: bool = False
) -> list[Diagnostic]:
    """Run read-only diagnostics for selected components.

    Returns:
        Classified diagnostics.
    """
    diagnostics: list[Diagnostic] = []
    if "tools" in components:
        diagnostics.extend(check_tools())
    if "policy" in components:
        diagnostics.extend(check_policy(environment=environment))
    if "github" in components:
        diagnostics.extend(check_github(environment=environment))
    if "vercel" in components:
        diagnostics.extend(check_vercel(environment=environment))
    if "hf" in components:
        diagnostics.extend(check_hf(environment=environment))
    if "redis" in components:
        diagnostics.extend(check_redis(environment=environment))
    if "ghcr" in components:
        diagnostics.extend(check_ghcr(environment=environment))
    if route_probes or "routes" in components:
        diagnostics.extend(
            smoke(
                base_url=PRODUCTION_BASE_URL,
                routes=list(ROUTES),
                environment=environment,
            )
        )
    return diagnostics


def check_ghcr(*, environment: dict[str, str]) -> list[Diagnostic]:
    """Check the public package and exact platform manifest without pulling it.

    Returns:
        GHCR diagnostics.
    """
    visibility = run_command(
        [
            "gh",
            "api",
            "/orgs/EuroEval/packages/container/euroeval-worker",
            "--jq",
            ".visibility",
        ]
    )
    if visibility.returncode:
        category = _classify_registry_failure(visibility)
        visibility_diagnostic = Diagnostic(
            "ghcr", category, "GHCR package visibility cannot be checked", True
        )
    elif visibility.stdout.strip() == "public":
        visibility_diagnostic = Diagnostic(
            "ghcr", "ok", "GHCR package visibility is public"
        )
    elif visibility.stdout.strip() == "private":
        visibility_diagnostic = Diagnostic(
            "ghcr", "drift", "GHCR package is private", True
        )
    else:
        visibility_diagnostic = Diagnostic(
            "ghcr", "malformed", "GHCR package visibility response is malformed", True
        )
    result = [visibility_diagnostic]
    digest = environment.get("VOLUNTEER_WORKER_IMAGE_DIGEST")
    if not digest:
        return result + [
            Diagnostic(
                "ghcr",
                "missing config",
                "VOLUNTEER_WORKER_IMAGE_DIGEST is not configured",
                True,
            )
        ]
    if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
        return result + [
            Diagnostic("ghcr", "drift", "configured image digest is malformed", True)
        ]
    reference = f"{IMAGE_REPOSITORY}@{digest}"
    with tempfile.TemporaryDirectory(prefix="euroeval-ghcr-") as directory:
        inspect = run_command(
            [
                "docker",
                "buildx",
                "imagetools",
                "inspect",
                reference,
                "--format",
                "{{json .}}",
            ],
            environment={"DOCKER_CONFIG": directory},
        )
    if inspect.returncode:
        category = _classify_registry_failure(inspect)
        return result + [
            Diagnostic(
                "ghcr", category, "anonymous image manifest inspection failed", True
            )
        ]
    if not _manifest_matches(inspect.stdout, digest):
        return result + [
            Diagnostic(
                "ghcr",
                "malformed",
                "manifest does not prove the configured linux/amd64 digest",
                True,
            )
        ]
    return result + [
        Diagnostic(
            "ghcr", "ok", "anonymous manifest proves the configured linux/amd64 digest"
        )
    ]


def _classify_registry_failure(result: CommandResult) -> str:
    """Classify registry inspection failures without returning registry output.

    Returns:
        Safe failure category.
    """
    text = f"{result.stdout} {result.stderr}".lower()
    if any(
        value in text for value in ("unauthor", "denied", "authentication", "login")
    ):
        return "auth"
    if any(
        value in text for value in ("not found", "manifest unknown", "name unknown")
    ):
        return "missing"
    if any(value in text for value in ("timeout", "connection", "network", "dns")):
        return "network"
    return "service failure"


def _manifest_matches(output: str, digest: str) -> bool:
    """Return whether manifest output proves digest and linux/amd64."""
    try:
        decoded: object = json.loads(output)
    except json.JSONDecodeError:
        decoded = None
    if decoded is not None:
        return _manifest_json_matches(decoded, digest)
    digest_match = re.search(rf"Digest:\s*{re.escape(digest)}", output)
    platform_match = re.search(r"Platform:\s*linux/amd64", output)
    return digest_match is not None and platform_match is not None


def _manifest_json_matches(value: object, digest: str) -> bool:
    """Return whether structured output proves the index and amd64 child.

    Returns:
        Whether the top-level digest and a child platform are both proven.
    """
    if not isinstance(value, dict):
        return False
    index_digest = value.get("digest", value.get("Digest"))
    if index_digest is None:
        descriptor = value.get("descriptor", value.get("Descriptor"))
        if isinstance(descriptor, dict):
            index_digest = descriptor.get("digest", descriptor.get("Digest"))
    if index_digest != digest:
        return False
    manifests = value.get("manifests", value.get("Manifests"))
    if not isinstance(manifests, list):
        return False
    return any(_is_amd64_descriptor(item) for item in manifests)


def _is_amd64_descriptor(value: object) -> bool:
    """Return whether a manifest child is a linux/amd64 descriptor."""
    if not isinstance(value, dict):
        return False
    platform = value.get("platform", value.get("Platform"))
    if not isinstance(platform, dict):
        return False
    if platform.get("os", platform.get("OS")) != "linux":
        return False
    if platform.get("architecture", platform.get("Architecture")) != "amd64":
        return False
    child_digest = value.get("digest", value.get("Digest"))
    if not isinstance(child_digest, str) or not child_digest:
        descriptor = value.get("descriptor", value.get("Descriptor"))
        if isinstance(descriptor, dict):
            child_digest = descriptor.get("digest", descriptor.get("Digest"))
    return isinstance(child_digest, str) and bool(child_digest)


def check_github(*, environment: dict[str, str]) -> list[Diagnostic]:
    """Check GitHub authentication, repository access, and queue labels.

    Returns:
        GitHub diagnostics.
    """
    del environment
    result: list[Diagnostic] = []
    auth = run_command(["gh", "auth", "status"])
    if auth.returncode:
        result.append(Diagnostic("github", "auth", "gh is not authenticated", True))
        return result
    repo = run_command(["gh", "repo", "view", REPOSITORY])
    if repo.returncode:
        result.append(
            Diagnostic(
                "github", "service failure", "repository is not accessible", True
            )
        )
        return result
    labels = run_command(["gh", "api", f"repos/{REPOSITORY}/labels", "--paginate"])
    if labels.returncode:
        result.append(
            Diagnostic("github", "auth", "repository labels cannot be inspected", True)
        )
        return result
    names = _json_names(labels.stdout)
    for label in LABELS:
        result.append(
            Diagnostic(
                "github",
                "ok" if label in names else "drift",
                f"label {label}: {'present' if label in names else 'missing'}",
                label not in names,
            )
        )
    result.append(
        Diagnostic(
            "github",
            "manual",
            "issue-write permission is manual/unverified; no token mutation "
            "was attempted",
        )
    )
    return result


def check_hf(*, environment: dict[str, str]) -> list[Diagnostic]:
    """Check HF authentication and configured staging bucket privacy.

    Returns:
        Hugging Face diagnostics.
    """
    if not environment.get("HF_TOKEN") or not environment.get("HF_STAGING_BUCKET"):
        return [
            Diagnostic(
                "hf",
                "missing config",
                "HF_TOKEN and HF_STAGING_BUCKET are required",
                True,
            )
        ]
    bucket = environment["HF_STAGING_BUCKET"]
    auth = run_command(["uv", "run", "hf", "auth", "whoami"])
    if auth.returncode:
        return [Diagnostic("hf", "auth", "Hugging Face authentication failed", True)]
    info = run_command(["uv", "run", "hf", "buckets", "info", bucket, "--json"])
    if info.returncode:
        category, message = _classify_hf_failure(info)
        return [Diagnostic("hf", category, message, True)]
    data = _json_object(info.stdout)
    if not data:
        return [
            Diagnostic("hf", "service failure", "bucket metadata is malformed", True)
        ]
    return _hf_metadata_diagnostics(data)


def check_policy(*, environment: dict[str, str]) -> list[Diagnostic]:
    """Check source/policy versions and generated policy freshness.

    Returns:
        Policy diagnostics.
    """
    euroeval_version, worker_version = source_versions()
    result: list[Diagnostic] = [
        Diagnostic(
            "policy",
            "ok",
            f"source versions: EuroEval {euroeval_version}, worker {worker_version}",
        )
    ]
    configured = environment.get("EUROEVAL_VERSION")
    if configured and normalise_version(configured) != normalise_version(
        euroeval_version
    ):
        result.append(
            Diagnostic("policy", "drift", "EUROEVAL_VERSION differs from source", True)
        )
    policy_path = Path("api/worker/scope-policy.json")
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        policy_versions = {
            str(entry["euroeval_version"]) for entry in policy["policies"]
        }
        policy_versions.add(
            str(policy["policy_version"]).removeprefix("volunteer-scope/")
        )
        if {normalise_version(value) for value in policy_versions} != {
            normalise_version(euroeval_version)
        }:
            result.append(
                Diagnostic(
                    "policy",
                    "drift",
                    "generated policy version differs from source",
                    True,
                )
            )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        result.append(
            Diagnostic(
                "policy",
                "missing config",
                "generated scope policy is missing or invalid",
                True,
            )
        )
    freshness = run_command(
        [
            "uv",
            "run",
            "python",
            "src/scripts/generate_volunteer_scope_policy.py",
            "--check",
        ]
    )
    if freshness.returncode:
        result.append(
            Diagnostic(
                "policy", "drift", "generated scope policy is stale or missing", True
            )
        )
    return result


def normalise_version(value: str) -> str:
    """Normalise the broker's accepted trailing development notation.

    Returns:
        Normalised version.
    """
    return value[:-4] + ".dev0" if value.endswith(".dev") else value


def check_redis(*, environment: dict[str, str]) -> list[Diagnostic]:
    """Perform an Upstash data-plane PING when both values are configured.

    Returns:
        Redis diagnostics.
    """
    url, token = (
        environment.get("UPSTASH_REDIS_REST_URL"),
        environment.get("UPSTASH_REDIS_REST_TOKEN"),
    )
    if not url or not token:
        return [
            Diagnostic(
                "redis", "missing config", "Upstash URL and token are required", True
            )
        ]
    response = request_http(
        url,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "content-type": "application/json",
        },
        body=b'["PING"]',
    )
    if response.status == 0:
        return [
            Diagnostic("redis", "network", "Upstash PING could not be reached", True)
        ]
    if response.status != 200:
        return [
            Diagnostic(
                "redis",
                "service failure",
                f"Upstash PING returned HTTP {response.status}",
                True,
            )
        ]
    try:
        payload = json.loads(response.body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return [
            Diagnostic(
                "redis", "service failure", "Upstash PING returned malformed JSON", True
            )
        ]
    if not isinstance(payload, dict) or payload.get("result") != "PONG":
        return [
            Diagnostic(
                "redis",
                "service failure",
                "Upstash PING returned an invalid response",
                True,
            )
        ]
    return [Diagnostic("redis", "ok", "Upstash PING succeeded")]


@dataclasses.dataclass(frozen=True)
class HttpResult:
    """Captured HTTP response without exposing request credentials."""

    status: int
    body: bytes


def request_http(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> HttpResult:
    """Make one HTTP request, returning HTTP errors as ordinary responses.

    Returns:
        Captured HTTP result; status zero denotes a network failure.
    """
    try:
        request = urllib.request.Request(
            url, data=body, headers=headers or {}, method=method
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            return HttpResult(response.status, response.read())
    except urllib.error.HTTPError as error:
        try:
            body = error.read()
        except (OSError, http.client.HTTPException):
            body = b""
        return HttpResult(error.code, body)
    except (
        ValueError,
        TypeError,
        TimeoutError,
        OSError,
        http.client.HTTPException,
        urllib.error.URLError,
    ):
        return HttpResult(0, b"")


def check_tools() -> list[Diagnostic]:
    """Check tools needed by the corresponding operations.

    Returns:
        Tool diagnostics.
    """
    return [
        Diagnostic(
            "tools",
            "missing config" if shutil.which(tool) is None else "ok",
            f"{tool} is {'available' if shutil.which(tool) else 'not installed'}",
            shutil.which(tool) is None,
        )
        for tool in ("git", "uv", "gh", "vercel", "hf", "docker")
    ]


def smoke(
    *, base_url: str, routes: list[str], environment: dict[str, str]
) -> list[Diagnostic]:
    """Probe only harmless method guards and unauthenticated protected routes.

    Returns:
        Route diagnostics.
    """
    base = base_url.rstrip("/")
    diagnostics: list[Diagnostic] = []
    for route in routes:
        url = f"{base}/api/worker/{route}"
        get = request_http(url)
        expected = get.status == 405 and get.body == b'{"error":"Method not allowed"}'
        get_state = (
            "405 exact method error"
            if expected
            else (f"HTTP {get.status} or unexpected body")
        )
        diagnostics.append(
            Diagnostic(
                "routes",
                "ok" if expected else "service failure",
                f"GET /{route}: {get_state}",
                not expected,
            )
        )
        options = request_http(url, method="OPTIONS")
        expected_options = options.status == 204 and options.body == b""
        options_state = (
            "204 empty body"
            if expected_options
            else (f"HTTP {options.status} or non-empty body")
        )
        diagnostics.append(
            Diagnostic(
                "routes",
                "ok" if expected_options else "service failure",
                f"OPTIONS /{route}: {options_state}",
                not expected_options,
            )
        )
    unauthenticated = request_http(
        f"{base}/api/worker/claim",
        method="POST",
        headers={"content-type": "application/json"},
        body=b"{}",
    )
    claim_ok = unauthenticated.status == 401 and _has_error_body(unauthenticated.body)
    diagnostics.append(
        Diagnostic(
            "routes",
            "ok" if claim_ok else "service failure",
            "POST /claim without credentials: 401 authentication error"
            if claim_ok
            else "POST /claim did not return a valid 401 error",
            not claim_ok,
        )
    )
    for route in ("coordinator-lock", "promotion-lock"):
        response = request_http(
            f"{base}/api/worker/{route}",
            method="POST",
            headers={"content-type": "application/json"},
            body=b'{"protocol_version":"volunteer-worker/v1","issue_number":1}',
        )
        valid_401 = response.status == 401 and _has_error_body(response.body)
        valid_503 = response.status == 503 and _has_error_body(response.body)
        expected_probe = valid_401 or valid_503
        classification = (
            "configured authentication rejected"
            if valid_401
            else (
                "deployed but missing configuration"
                if valid_503
                else "invalid response"
            )
        )
        diagnostics.append(
            Diagnostic(
                "routes",
                "ok" if expected_probe else "service failure",
                f"POST /{route} without credentials: {classification}",
                not expected_probe,
            )
        )
    return diagnostics


def _has_error_body(body: bytes) -> bool:
    """Validate the JSON error contract used by protected broker routes.

    Returns:
        Whether the body contains a non-empty error string.
    """
    try:
        decoded = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(decoded, dict)
        and isinstance(decoded.get("error"), str)
        and bool(decoded["error"])
    )


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check", "apply", "smoke"))
    parser.add_argument(
        "--component",
        action="append",
        choices=(
            "tools",
            "policy",
            "github",
            "vercel",
            "hf",
            "redis",
            "ghcr",
            "routes",
        ),
        help="Limit check/apply to a component; may be repeated.",
    )
    parser.add_argument("--github", action="store_true", help="Apply GitHub labels.")
    parser.add_argument("--vercel", action="store_true", help="Apply Vercel variables.")
    parser.add_argument(
        "--hf", action="store_true", help="Apply the HF staging bucket."
    )
    parser.add_argument(
        "--hf-region",
        choices=("eu", "us"),
        default="eu",
        help="Region for a newly created HF bucket (apply --hf --yes only).",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Confirm an apply operation."
    )
    parser.add_argument(
        "--local-secrets",
        action="store_true",
        help="Initialise missing local broker secrets in .env (apply --yes only).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Local dotenv file for --local-secrets (default: .env).",
    )
    parser.add_argument(
        "--reuse-vercel-kv",
        action="store_true",
        help="Copy existing Vercel KV credentials to broker aliases.",
    )
    parser.add_argument(
        "--routes", action="store_true", help="Include deployed route probes in check."
    )
    parser.add_argument(
        "--base-url", default=PRODUCTION_BASE_URL, help="Deployed broker base URL."
    )
    parser.add_argument("--route", action="append", help="Smoke only this route.")
    return parser.parse_args(argv)


def print_diagnostics(diagnostics: list[Diagnostic]) -> None:
    """Print classified diagnostics without command output or secret values."""
    for item in diagnostics:
        status = "FAIL" if item.failed else "OK"
        print(f"{status} [{item.category}] {item.component}: {item.message}")


def print_plan(*, environment: dict[str, str]) -> None:
    """Print the immutable workflow and names of required inputs."""
    euroeval_version, worker_version = source_versions()
    print("1. AUTOMATED CHECK: inspect tools, source versions, and generated policy.")
    print(
        "2. AUTOMATED CHECK: inspect GitHub, Vercel, HF, Redis, and GHCR configuration."
    )
    print(
        "3. CONFIRMED AUTOMATION: apply only explicitly selected labels, bucket, "
        "or variables."
    )
    print(
        "4. MANUAL: publish/canary/promote the immutable GPU image and deploy Vercel."
    )
    print("5. AUTOMATED CHECK: run safe GET/OPTIONS and unauthenticated POST probes.")
    print(f"Defaults: repository={REPOSITORY}; base_url={PRODUCTION_BASE_URL};")
    print(f"  results_bucket={RESULTS_BUCKET}; image={IMAGE_REPOSITORY};")
    print(f"  euroeval_version={euroeval_version}; worker_version={worker_version}")
    print("  hf_region=eu (HF bucket creation default)")
    print(
        "EU creation requires an eligible organisation plan; HF metadata cannot "
        "verify an existing bucket's region."
    )
    print(
        "US requires an explicit data-residency decision; use apply --hf --yes "
        "--hf-region us only then."
    )
    print("Required environment variable names (values are never printed):")
    for name in REQUIRED_ENVIRONMENT:
        state = "present" if environment.get(name) else "missing"
        print(f"  {name} ({state})")
    print(
        "Manual inputs still needed: maintainer credentials, project choice, image "
        "digest,"
    )
    print("physical Linux amd64 GPU canary, image promotion, and deployment approval.")


def selected_components(
    arguments: argparse.Namespace, *, explicit: bool = False
) -> set[str]:
    """Return requested components, keeping apply's default deliberately empty."""
    flags = {
        name for name in ("github", "vercel", "hf") if getattr(arguments, name, False)
    }
    requested = set(arguments.component or ()) | flags
    if explicit:
        return requested
    return requested or {"tools", "policy", "github", "vercel", "hf", "redis", "ghcr"}


if __name__ == "__main__":
    raise SystemExit(main())
