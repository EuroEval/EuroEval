"""Safe maintainer operations for the EuroEval volunteer GPU broker.

This command deliberately keeps cloud mutations small, explicit, and injectable.  It
never reads dotenv files: credentials must already be present in the process
environment.
"""

from __future__ import annotations

import argparse
import dataclasses
import http.client
import json
import os
import re
import shutil
import subprocess
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
        )
    diagnostics = smoke(
        base_url=arguments.base_url,
        routes=arguments.route or list(ROUTES),
        environment=environment,
    )
    print_diagnostics(diagnostics)
    return int(any(item.failed for item in diagnostics))


def apply(*, environment: dict[str, str], components: set[str], confirmed: bool) -> int:
    """Apply only explicitly confirmed, narrowly scoped setup.

    Returns:
        Process exit status.
    """
    if not components or not components <= {"github", "hf", "vercel"}:
        print(
            "apply requires explicit --github, --hf, or --vercel (no default/all path)."
        )
        return 2
    if not confirmed:
        print("apply requires --yes; no changes were made.")
        return 2
    diagnostics: list[Diagnostic] = []
    if "github" in components:
        diagnostics.extend(apply_github())
    if "hf" in components:
        diagnostics.extend(apply_hf(environment=environment))
    if "vercel" in components:
        diagnostics.extend(apply_vercel(environment=environment))
    print_diagnostics(diagnostics)
    return int(any(item.failed for item in diagnostics))


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


def apply_hf(*, environment: dict[str, str]) -> list[Diagnostic]:
    """Create a missing private EU bucket, or verify an existing bucket.

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
    current = run_command(["uv", "run", "hf", "buckets", "info", bucket, "--json"])
    if current.returncode == 0:
        if _json_object(current.stdout):
            return check_hf(environment=environment)
        return [
            Diagnostic("hf", "service failure", "bucket metadata is malformed", True)
        ]
    category, _ = _classify_hf_failure(current)
    if category != "missing":
        return [
            Diagnostic("hf", category, "existing bucket could not be verified", True)
        ]
    created = run_command(
        ["uv", "run", "hf", "buckets", "create", bucket, "--private", "--region", "eu"]
    )
    if created.returncode:
        return [
            Diagnostic(
                "hf",
                "service failure",
                "private EU staging bucket creation failed",
                True,
            )
        ]
    return check_hf(environment=environment)


def _classify_hf_failure(result: CommandResult) -> tuple[str, str]:
    """Classify HF CLI failures without exposing command output.

    Returns:
        Failure category and safe message.
    """
    text = f"{result.stdout} {result.stderr}".lower()
    if any(value in text for value in ("401", "403", "unauthor", "token")):
        return "auth", "Hugging Face bucket metadata is not authorised"
    if any(value in text for value in ("timeout", "network", "connection", "dns")):
        return "network", "Hugging Face bucket metadata could not be reached"
    if any(value in text for value in ("404", "not found", "does not exist")):
        return "missing", "configured staging bucket does not exist"
    return "service failure", "Hugging Face bucket metadata could not be read"


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
    region = data.get("region")
    if region is None:
        region_diagnostic = Diagnostic(
            "hf", "manual", "existing bucket region is manual/unverifiable"
        )
    elif str(region).lower() == "eu":
        region_diagnostic = Diagnostic("hf", "ok", "existing bucket region is EU")
    else:
        region_diagnostic = Diagnostic(
            "hf", "drift", "existing bucket region is not EU", True
        )
    return [privacy, region_diagnostic]


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
    project = run_command(["vercel", "project", "inspect", "--format", "json"])
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


def print_diagnostics(diagnostics: list[Diagnostic]) -> None:
    """Print classified diagnostics without command output or secret values."""
    for item in diagnostics:
        status = "FAIL" if item.failed else "OK"
        print(f"{status} [{item.category}] {item.component}: {item.message}")


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
        "--yes", action="store_true", help="Confirm an apply operation."
    )
    parser.add_argument(
        "--routes", action="store_true", help="Include deployed route probes in check."
    )
    parser.add_argument(
        "--base-url", default=PRODUCTION_BASE_URL, help="Deployed broker base URL."
    )
    parser.add_argument("--route", action="append", help="Smoke only this route.")
    return parser.parse_args(argv)


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
