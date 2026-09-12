"""Safe maintainer operations for the EuroEval volunteer GPU broker.

This command deliberately keeps cloud mutations small, explicit, and injectable.  It
never reads dotenv files: credentials must already be present in the process
environment.
"""

from __future__ import annotations

import argparse
import dataclasses
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
    "WORKER_COORDINATOR_LOGIN",
    "UPSTASH_REDIS_REST_URL",
    "UPSTASH_REDIS_REST_TOKEN",
    "HF_STAGING_BUCKET",
    "HF_TOKEN",
    "WORKER_COORDINATOR_SECRET",
    "VOLUNTEER_PROMOTION_SECRET",
)
DURABLE_SECRETS = {
    "VOLUNTEER_MARKER_SECRET",
    "WORKER_COORDINATOR_SECRET",
    "VOLUNTEER_PROMOTION_SECRET",
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


@dataclasses.dataclass(frozen=True)
class CommandResult:
    """Captured result of an external command."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclasses.dataclass(frozen=True)
class HttpResult:
    """Captured HTTP response without exposing request credentials."""

    status: int
    body: bytes


@dataclasses.dataclass(frozen=True)
class Diagnostic:
    """One concise, classified diagnostic."""

    component: str
    category: str
    message: str
    failed: bool = False


def run_command(
    command: list[str],
    *,
    input_text: str | None = None,
    environment: dict[str, str] | None = None,
) -> CommandResult:
    """Run a command and capture it without displaying its output.

    Returns:
        Captured command result.
    """
    try:
        completed = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
    except OSError:
        return CommandResult(127)
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


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
    request = urllib.request.Request(
        url, data=body, headers=headers or {}, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return HttpResult(response.status, response.read())
    except urllib.error.HTTPError as error:
        return HttpResult(error.code, error.read())
    except urllib.error.URLError:
        return HttpResult(0, b"")


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


def check_github(*, environment: dict[str, str]) -> list[Diagnostic]:
    """Check GitHub authentication, labels, and coordinator permission.

    Returns:
        GitHub diagnostics.
    """
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
    login = environment.get("WORKER_COORDINATOR_LOGIN")
    if not login:
        result.append(
            Diagnostic(
                "github",
                "missing config",
                "WORKER_COORDINATOR_LOGIN is not configured",
                True,
            )
        )
    else:
        permission = run_command(
            ["gh", "api", f"repos/{REPOSITORY}/collaborators/{login}/permission"]
        )
        data = _json_object(permission.stdout)
        allowed = str(data.get("permission", "")) in {
            "admin",
            "maintain",
            "push",
            "triage",
        }
        permission_state = "sufficient" if allowed else "insufficient"
        result.append(
            Diagnostic(
                "github",
                "ok" if allowed else "service failure",
                f"coordinator collaborator permission: {permission_state}",
                not allowed,
            )
        )
    return result


def check_vercel(*, environment: dict[str, str]) -> list[Diagnostic]:
    """Check the linked Vercel project and Production variable metadata.

    Returns:
        Vercel diagnostics.
    """
    result: list[Diagnostic] = []
    link_path = Path(".vercel/project.json")
    try:
        link = _json_object(link_path.read_text(encoding="utf-8"))
    except OSError:
        link = {}
    if not link.get("projectId"):
        result.append(
            Diagnostic(
                "vercel",
                "missing config",
                "repository is not linked to a Vercel project",
                True,
            )
        )
    project = run_command(["vercel", "project", "inspect", "--json"])
    project_data = _json_object(project.stdout)
    project_name = str(project_data.get("name", ""))
    identity_ok = project.returncode == 0 and project_name.lower() in {
        "euroeval",
        "euro-eval",
    }
    if not identity_ok:
        result.append(
            Diagnostic(
                "vercel",
                "auth" if project.returncode else "drift",
                "linked project identity is unavailable or is not EuroEval",
                True,
            )
        )
    else:
        result.append(Diagnostic("vercel", "ok", "linked project identity is EuroEval"))
    variables = run_command(["vercel", "env", "ls", "production", "--format=json"])
    if variables.returncode:
        result.append(
            Diagnostic(
                "vercel", "auth", "Production environment metadata cannot be read", True
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
        type_ok = variable_type in {"plain", "secret", "encrypted", "sensitive"}
        present = item is not None
        valid = present and target_ok and type_ok
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
    auth = run_command(["uv", "run", "hf", "whoami"])
    if auth.returncode:
        return [Diagnostic("hf", "auth", "Hugging Face authentication failed", True)]
    info = run_command(
        [
            "uv",
            "run",
            "hf",
            "buckets",
            "info",
            environment["HF_STAGING_BUCKET"],
            "--json",
        ]
    )
    data = _json_object(info.stdout)
    private = data.get("private") is True or data.get("visibility") == "private"
    return [
        Diagnostic(
            "hf",
            "ok" if info.returncode == 0 and private else "drift",
            "EU staging bucket exists and is private"
            if private
            else "staging bucket is missing or not private",
            not (info.returncode == 0 and private),
        )
    ]


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
    if response.status != 200:
        return [
            Diagnostic(
                "redis",
                "network" if response.status == 0 else "service failure",
                f"Upstash PING returned HTTP {response.status}",
                True,
            )
        ]
    return [Diagnostic("redis", "ok", "Upstash PING succeeded")]


def check_ghcr(*, environment: dict[str, str]) -> list[Diagnostic]:
    """Check public package visibility and anonymously inspect/pull a digest.

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
    public = visibility.stdout.strip() == "public"
    visibility_state = "public" if public else "not public"
    result = [
        Diagnostic(
            "ghcr",
            "ok" if public else ("auth" if visibility.returncode else "drift"),
            f"GHCR package visibility: {visibility_state}",
            visibility.returncode != 0 or not public,
        )
    ]
    digest = environment.get("VOLUNTEER_WORKER_IMAGE_DIGEST")
    if not digest:
        result.append(
            Diagnostic(
                "ghcr",
                "missing config",
                "VOLUNTEER_WORKER_IMAGE_DIGEST is not configured",
                True,
            )
        )
        return result
    if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
        result.append(
            Diagnostic(
                "ghcr",
                "drift",
                "configured image digest is not sha256 plus 64 hex characters",
                True,
            )
        )
        return result
    reference = f"{IMAGE_REPOSITORY}@{digest}"
    with tempfile.TemporaryDirectory(prefix="euroeval-ghcr-") as directory:
        isolated = {**environment, "DOCKER_CONFIG": directory}
        inspect = run_command(
            ["docker", "manifest", "inspect", "--verbose", reference],
            environment=isolated,
        )
        pull = run_command(["docker", "pull", reference], environment=isolated)
    failed = inspect.returncode != 0 or pull.returncode != 0
    result.append(
        Diagnostic(
            "ghcr",
            "network" if failed else "ok",
            "anonymous digest inspect and pull succeeded"
            if not failed
            else "anonymous digest inspect or pull failed",
            failed,
        )
    )
    return result


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


def apply_hf(*, environment: dict[str, str]) -> list[Diagnostic]:
    """Create or verify the configured private EU staging bucket.

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
            "eu",
            "--exist-ok",
        ]
    )
    if created.returncode:
        return [
            Diagnostic(
                "hf", "service failure", "private EU staging bucket setup failed", True
            )
        ]
    return check_hf(environment=environment)


def apply_vercel(*, environment: dict[str, str]) -> list[Diagnostic]:
    """Add or update Production variables using stdin, never command arguments.

    Returns:
        Vercel diagnostics.
    """
    euroeval_version, worker_version = source_versions()
    values = dict(environment)
    if not values.get("EUROEVAL_VERSION"):
        values["EUROEVAL_VERSION"] = euroeval_version
    if not values.get("VOLUNTEER_WORKER_VERSION"):
        values["VOLUNTEER_WORKER_VERSION"] = worker_version
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
        if name in DURABLE_SECRETS:
            # Durable secrets are accepted when explicitly supplied, never generated.
            pass
        removed = run_command(["vercel", "env", "rm", name, "production", "--yes"])
        if removed.returncode not in (0, 1):
            diagnostics.append(
                Diagnostic(
                    "vercel",
                    "service failure",
                    f"could not prepare Production variable {name}",
                    True,
                )
            )
            continue
        added = run_command(
            ["vercel", "env", "add", name, "production"], input_text=values[name] + "\n"
        )
        update_state = "updated" if added.returncode == 0 else "update failed"
        diagnostics.append(
            Diagnostic(
                "vercel",
                "ok" if added.returncode == 0 else "service failure",
                f"Production variable {name}: {update_state}",
                added.returncode != 0,
            )
        )
    return diagnostics


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
    expected_claim = unauthenticated.status == 401
    claim_state = "401" if expected_claim else f"HTTP {unauthenticated.status}"
    diagnostics.append(
        Diagnostic(
            "routes",
            "ok" if expected_claim else "service failure",
            f"POST /claim without credentials: {claim_state}",
            not expected_claim,
        )
    )
    for route, secret_name in (
        ("coordinator-lock", "WORKER_COORDINATOR_SECRET"),
        ("promotion-lock", "VOLUNTEER_PROMOTION_SECRET"),
    ):
        headers = {"content-type": "application/json"}
        response = request_http(
            f"{base}/api/worker/{route}",
            method="POST",
            headers=headers,
            body=b'{"protocol_version":"volunteer-worker/v1","issue_number":1}',
        )
        expected_status = 401 if environment.get(secret_name) else 503
        expected_probe = response.status == expected_status
        probe_state = (
            f"expected {expected_status}"
            if expected_probe
            else f"unexpected HTTP {response.status}"
        )
        diagnostics.append(
            Diagnostic(
                "routes",
                "ok" if expected_probe else "service failure",
                f"POST /{route} without credentials: {probe_state}",
                not expected_probe,
            )
        )
    return diagnostics


def print_diagnostics(diagnostics: list[Diagnostic]) -> None:
    """Print classified diagnostics without command output or secret values."""
    for item in diagnostics:
        status = "FAIL" if item.failed else "OK"
        print(f"{status} [{item.category}] {item.component}: {item.message}")


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


def normalise_version(value: str) -> str:
    """Normalise the broker's accepted trailing development notation.

    Returns:
        Normalised version.
    """
    return value[:-4] + ".dev0" if value.endswith(".dev") else value


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


def _json_names(value: str) -> set[str]:
    """Extract names from a GitHub JSON response.

    Returns:
        Names found in the response.
    """
    return {str(item.get("name", "")) for item in _json_list(value)}


if __name__ == "__main__":
    raise SystemExit(main())
