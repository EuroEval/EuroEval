"""Two-phase worker publication; all remote observations cross injectable adapters.

No website build or benchmark data is transported by this module. Activation is
separate from installation: reviewed policy plus explicit CI or local setup gate it.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import subprocess
import tempfile
import tomllib
import typing as t
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import click
from packaging.version import Version

REPOSITORY = "EuroEval/EuroEval"
IMAGE = "ghcr.io/euroeval/euroeval-worker"
ARCHES = ("amd64", "arm64")
POLICY = Path("src/scripts/worker-release-policy.json")
ACTIVATION = Path(".vercel/worker-release-activation.json")
PROJECT = Path(".vercel/project.json")
STATE = Path(".vercel/worker-release-state.json")
MAX_PROOF_BYTES = 262144
POLICIES = ("api/worker/scope-policy.json", "api/worker/_lib/scope-policy.generated.ts")


def main(argv: list[str] | None = None) -> int:
    """Run real release operations; disabled hooks do not contact any service.

    Args:
        argv: Optional CLI arguments.

    Returns:
        Process exit status.
    """
    try:
        return cli.main(args=argv, standalone_mode=False) or 0
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.exceptions.Exit as exc:
        return exc.exit_code


class GitHubAdapter(t.Protocol):
    """Injectable GitHub observations and narrowly scoped effects."""

    def artifact(self, run: dict, name: str) -> dict:
        """Read a unique unexpired artifact from the selected attempt.

        Args:
            run: Observed trusted run.
            name: Attempt-qualified artifact name.

        Returns:
            JSON metadata from the artifact.
        """
        ...

    def dispatch(self, plan: dict, deployment: str) -> None:
        """Request finalization after the ordinary local deploy succeeds.

        Args:
            plan: Staged plan.
            deployment: Observed READY deployment identity.
        """
        ...

    def head(self) -> str:
        """Return the current trusted main commit."""
        ...

    def pending(self) -> dict | None:
        """Read the newest immutable public staged plan, if any.

        Returns:
            The public staged plan, or None.
        """
        ...

    def proof_asset(self, asset_id: int) -> bytes:
        """Read exact retained proof bytes from the fixed repository.

        Args:
            asset_id: Positive release asset identity.

        Returns:
            Bounded raw JSON bytes.
        """
        ...

    def publish(self, plan: dict) -> None:
        """Publish only public JSON, never image or leaderboard archives.

        Args:
            plan: Immutable staged plan.
        """
        ...

    def runs(self, workflow: str, sha: str) -> list[dict]:
        """Observe exact-SHA runs for one trusted workflow.

        Args:
            workflow: Workflow filename.
            sha: Full source identity.

        Returns:
            Workflow runs.
        """
        ...


class RegistryAdapter(t.Protocol):
    """Injectable Registry observations and narrowly scoped effects."""

    def identities(self, indexes: dict[str, str]) -> tuple[list[dict], dict]:
        """Resolve platform/config identities and preserve all attestations.

        Args:
            indexes: Architecture to original build index digest.

        Returns:
            Complete descriptor union and platform/config identities.
        """
        ...

    def manifest(self, reference: str) -> tuple[bytes, dict]:
        """Fetch and verify exact bytes anonymously from the fixed repository.

        Args:
            reference: Digest or repository-local tag.

        Returns:
            Original bytes and parsed manifest.
        """
        ...

    def put(self, tag: str, raw: bytes) -> str:
        """Publish exact index bytes through oras, retaining every descriptor.

        Args:
            tag: Destination tag.
            raw: Serialized OCI index.

        Returns:
            Published digest.

        Raises:
            HTTPError: On registry errors other than an absent immutable tag.
        """
        ...


class VercelAdapter(t.Protocol):
    """Injectable Vercel observations and narrowly scoped effects."""

    def config(self) -> list[dict]:
        """Observe environment metadata; never persist secret values.

        Returns:
            Observed environment entries.
        """
        ...

    def discovery(self, arch: str) -> dict:
        """Read public architecture-specific broker discovery anonymously.

        Args:
            arch: Requested architecture.

        Returns:
            Public broker metadata.
        """
        ...

    def production(self) -> dict:
        """Observe alias and its actual READY production deployment.

        Returns:
            The verified deployment metadata.
        """
        ...

    def resolve_deployment(self, hostname: str) -> dict:
        """Resolve the CLI's immutable deployment URL.

        Args:
            hostname: Deployment hostname.

        Returns:
            Deployment metadata.
        """
        ...

    def update(self, entry: dict, value: str) -> None:
        """Update only an existing exact-Production plain variable.

        Args:
            entry: Observed metadata.
            value: New public value.
        """
        ...


class GitHub:
    """Trusted-repository GitHub CLI adapter, including artifact byte retrieval."""

    def artifact(self, run: dict, name: str) -> dict:
        """Read a unique unexpired artifact from the selected attempt.

        Args:
            run: Observed trusted run.
            name: Attempt-qualified artifact name.

        Returns:
            JSON metadata from the artifact.
        """
        artifacts = self.api(f"actions/runs/{run['id']}/artifacts?per_page=100")[
            "artifacts"
        ]
        matches = [a for a in artifacts if a["name"] == name and not a["expired"]]
        require(len(matches) == 1, "missing or ambiguous attempt artifact")
        item = matches[0]
        require(
            item["workflow_run"]["head_sha"] == run["head_sha"],
            "artifact source mismatch",
        )
        raw = command(
            ["gh", "api", f"repos/{REPOSITORY}/actions/artifacts/{item['id']}/zip"]
        )
        require(item.get("digest") == digest(raw), "artifact archive digest mismatch")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            require(len(archive.namelist()) == 1, "unexpected artifact contents")
            value = json.loads(archive.read(archive.namelist()[0]))
        value["artifact_id"] = item["id"]
        return value

    def api(self, path: str, data: dict | None = None) -> dict:
        """Call the fixed repository, using stdin for request bodies.

        Args:
            path: Repository-relative endpoint.
            data: Optional JSON body.

        Returns:
            Decoded response.
        """
        args = ["gh", "api", f"repos/{REPOSITORY}/{path}"]
        if data is not None:
            args += ["--method", "POST", "--input", "-"]
        raw = command(args, json.dumps(data).encode() if data is not None else None)
        value = json.loads(raw) if raw else {}
        return value if isinstance(value, dict) else {"items": value}

    def dispatch(self, plan: dict, deployment: str) -> None:
        """Request finalization after the ordinary local deploy succeeds.

        Args:
            plan: Staged plan.
            deployment: Observed READY deployment identity.
        """
        self.api(
            "actions/workflows/worker-release.yaml/dispatches",
            {
                "ref": "main",
                "inputs": {
                    "phase": "finalize",
                    "release_tag": plan["tag"],
                    "deployment": deployment,
                },
            },
        )

    def head(self) -> str:
        """Return the current trusted main commit."""
        return self.api("commits/main")["sha"]

    def pending(self) -> dict | None:
        """Read the newest immutable public staged plan, if any.

        Returns:
            The public staged plan, or None.
        """
        releases = self.api("releases?per_page=100")["items"]
        for release in releases:
            if (
                release["tag_name"].startswith("worker-release-")
                and not release["draft"]
            ):
                assets = [a for a in release["assets"] if a["name"] == "release.json"]
                require(len(assets) == 1, "staged metadata missing")
                raw = command(
                    [
                        "gh",
                        "api",
                        "-H",
                        "Accept: application/octet-stream",
                        f"repos/{REPOSITORY}/releases/assets/{assets[0]['id']}",
                    ]
                )
                return json.loads(raw)
        return None

    def proof_asset(self, asset_id: int) -> bytes:
        """Retrieve bounded retained evidence only from the fixed repository.

        Args:
            asset_id: Positive GitHub release asset identity, never a URL.

        Returns:
            Exact retained JSON bytes.
        """
        require(type(asset_id) is int and asset_id > 0, "invalid proof asset id")
        metadata = self.api(f"releases/assets/{asset_id}")
        require(
            metadata.get("id") == asset_id
            and type(metadata.get("size")) is int
            and 0 < metadata["size"] <= MAX_PROOF_BYTES,
            "invalid proof asset size",
        )
        raw = command(
            [
                "gh",
                "api",
                "-H",
                "Accept: application/octet-stream",
                f"repos/{REPOSITORY}/releases/assets/{asset_id}",
            ]
        )
        require(len(raw) == metadata["size"], "proof asset size changed")
        return raw

    def publish(self, plan: dict) -> None:
        """Publish only public JSON, never image or leaderboard archives.

        Args:
            plan: Immutable staged plan.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release.json"
            path.write_text(json.dumps(plan, sort_keys=True))
            command(
                [
                    "gh",
                    "release",
                    "create",
                    plan["tag"],
                    str(path),
                    "--repo",
                    REPOSITORY,
                    "--target",
                    plan["source_sha"],
                    "--prerelease",
                    "--title",
                    plan["tag"],
                    "--notes",
                    "Staged worker; production and latest unchanged.",
                ]
            )

    def runs(self, workflow: str, sha: str) -> list[dict]:
        """Observe exact-SHA runs for one trusted workflow.

        Args:
            workflow: Workflow filename.
            sha: Full source identity.

        Returns:
            Workflow runs.
        """
        return self.api(
            f"actions/workflows/{workflow}/runs?head_sha={sha}&per_page=100"
        )["workflow_runs"]


def command(args: list[str], data: bytes | None = None) -> bytes:
    """Run a tool without exposing captured credentials or response bodies.

    Args:
        args: Tool arguments, never credentials.
        data: Optional stdin.

    Returns:
        Captured stdout.
    """
    result = subprocess.run(args, input=data, capture_output=True, check=False)
    require(result.returncode == 0, f"{args[0]} operation failed (output redacted)")
    return result.stdout


@click.command()
@click.argument(
    "operation",
    type=click.Choice(
        (
            "configure-local",
            "disable-local",
            "stage",
            "prepare",
            "predeploy",
            "finish",
            "finalize",
        )
    ),
)
@click.option("--release-tag", default="")
@click.option("--deployment", default="")
def cli(operation: str, release_tag: str, deployment: str) -> int:
    """Run gated release operations with existing injectable adapter seams.

    Args:
        operation: Release phase.
        release_tag: Immutable staged release tag.
        deployment: Observed deployment identity.

    Returns:
        Process exit status (also reflected by Click).

    Raises:
        click.ClickException: When release observations fail validation.
    """
    try:
        if operation == "disable-local":
            ACTIVATION.unlink(missing_ok=True)
            return 0
        if operation == "configure-local":
            require(
                os.environ.get("GITHUB_ACTIONS") != "true"
                and os.environ.get("CI") != "true",
                "local setup only",
            )
            marker = _local_activation(policy=json.loads(POLICY.read_text()))
            ACTIVATION.parent.mkdir(parents=True, exist_ok=True)
            temporary = ACTIVATION.with_suffix(".tmp")
            temporary.write_text(json.dumps(marker, sort_keys=True) + "\n")
            temporary.replace(ACTIVATION)
            return 0
        policy = load_activation(ci_only=operation in ("stage", "finalize"))
        if policy is None:
            return 0
        for key, variable in (
            ("acceptance", "WORKER_RELEASE_ACCEPTANCE_JSON"),
            ("baseline", "WORKER_RELEASE_BASELINE_JSON"),
        ):
            if os.environ.get(variable):
                policy[key] = json.loads(os.environ[variable])
        github, registry = GitHub(), Registry()
        if operation == "stage":
            stage(github, registry, policy, source())
        elif operation == "prepare":
            # No approved pending release must leave the normal deploy unchanged.
            plan = github.pending()
            if plan and digest(registry.manifest("latest")[0]) != plan["image_digest"]:
                project = json.loads(Path(".vercel/project.json").read_text())
                require(
                    project["projectId"] == policy["project_id"]
                    and project["orgId"] == policy["team_id"],
                    "local Vercel project linkage differs from release policy",
                )
                prepare(github, registry, Vercel(policy), policy, source())
        elif operation == "predeploy":
            if STATE.exists():
                state = json.loads(STATE.read_text())
                if state["status"] != "finalization-dispatched":
                    require(state["status"] == "prepared", "preparation incomplete")
                    vercel = Vercel(policy)
                    require(
                        public_config(vercel.config()) == state["prepared_config"],
                        "config changed before deployment",
                    )
                    require(
                        vercel.production()["id"] == state["previous_deployment"],
                        "alias changed before deployment",
                    )
                    require(
                        github.head() == state["plan"]["source_sha"],
                        "main advanced before deployment",
                    )
                    require(
                        all(
                            state["plan"][key] == value
                            for key, value in source().items()
                        ),
                        "local source changed before deployment",
                    )
        elif operation == "finish":
            if STATE.exists():
                finish(github, Vercel(policy))
        else:
            finalize(github, registry, Vercel(policy), release_tag, deployment)
    except (ValueError, KeyError, OSError, TypeError):
        logging.error(
            "Worker release stopped; inspect nonsecret recovery state. "
            "Remote response bodies suppressed."
        )
        raise click.ClickException("Worker release validation failed") from None
    return 0


def require(condition: object, message: str) -> None:
    """Reject missing or contradictory observed evidence.

    Args:
        condition: Required invariant.
        message: Nonsecret diagnostic.

    Raises:
        ValueError: When the invariant is not satisfied.
    """
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    """Return the content address of bytes.

    Args:
        data: Exact serialized content.

    Returns:
        SHA256 OCI digest.
    """
    return "sha256:" + hashlib.sha256(data).hexdigest()


class Registry:
    """GHCR adapter using authenticated writes and anonymous OCI reads."""

    def identities(self, indexes: dict[str, str]) -> tuple[list[dict], dict]:
        """Resolve platform/config identities and preserve all attestations.

        Args:
            indexes: Architecture to original build index digest.

        Returns:
            Complete descriptor union and platform/config identities.
        """
        descriptors = []
        identities = {}
        for arch in ARCHES:
            _, index = self.manifest(indexes[arch])
            children = index["manifests"]
            platforms = [
                m
                for m in children
                if m.get("platform", {}).get("architecture") == arch
                and m["platform"]["os"] == "linux"
            ]
            require(len(platforms) == 1, "missing or ambiguous platform")
            _, manifest = self.manifest(platforms[0]["digest"])
            identities[arch] = {
                "manifest": platforms[0]["digest"],
                "config": manifest["config"]["digest"],
            }
            for child in children:
                if child not in descriptors:
                    descriptors.append(child)
        return descriptors, identities

    def manifest(self, reference: str) -> tuple[bytes, dict]:
        """Fetch and verify exact bytes anonymously from the fixed repository.

        Args:
            reference: Digest or repository-local tag.

        Returns:
            Original bytes and parsed manifest.
        """
        require(
            re.fullmatch(r"sha256:[0-9a-f]{64}|[A-Za-z0-9_.-]+", reference),
            "untrusted registry reference",
        )
        token = json.loads(
            urllib.request.urlopen(
                "https://ghcr.io/token?scope=repository:euroeval/euroeval-worker:pull&service=ghcr.io",
                timeout=30,
            ).read()
        )["token"]
        request = urllib.request.Request(
            f"https://ghcr.io/v2/euroeval/euroeval-worker/manifests/{reference}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": (
                    "application/vnd.oci.image.index.v1+json, "
                    "application/vnd.oci.image.manifest.v1+json"
                ),
            },
        )
        raw = urllib.request.urlopen(request, timeout=30).read()
        if reference.startswith("sha256:"):
            require(digest(raw) == reference, "registry digest mismatch")
        return raw, json.loads(raw)

    def put(self, tag: str, raw: bytes) -> str:
        """Publish exact index bytes through oras, retaining every descriptor.

        Args:
            tag: Destination tag.
            raw: Serialized OCI index.

        Returns:
            Published digest.

        Raises:
            HTTPError: On registry errors other than an absent immutable tag.
        """
        if tag != "latest":
            try:
                existing, _ = self.manifest(tag)
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
            else:
                require(existing == raw, "immutable release tag already differs")
                return digest(raw)
        command(["oras", "manifest", "push", f"{IMAGE}:{tag}", "-"], raw)
        observed, _ = self.manifest(tag)
        require(observed == raw, "anonymous published index mismatch")
        return digest(raw)


class Vercel:
    """Use the existing local Vercel login or protected environment token."""

    def __init__(self, policy: dict) -> None:
        """Load authentication without printing it.

        Args:
            policy: Reviewed project and alias identities.
        """
        require(
            re.fullmatch(r"[a-z0-9.-]+", policy["alias"])
            and re.fullmatch(r"[A-Za-z0-9_]+", policy["project_id"])
            and re.fullmatch(r"[A-Za-z0-9_]+", policy["team_id"]),
            "invalid reviewed Vercel identities",
        )
        self.policy = policy
        self.token = os.environ.get("VERCEL_TOKEN")
        for path in (
            Path.home() / ".local/share/com.vercel.cli/auth.json",
            Path.home() / "Library/Application Support/com.vercel.cli/auth.json",
        ):
            if not self.token and path.exists():
                self.token = json.loads(path.read_text())["token"]
        require(self.token, "existing Vercel login required")

    def config(self) -> list[dict]:
        """Observe environment metadata; never persist secret values.

        Returns:
            Observed environment entries.
        """
        return self.api(f"/v10/projects/{self.policy['project_id']}/env")["envs"]

    def api(self, path: str, data: dict | None = None, method: str = "GET") -> dict:
        """Call Vercel with redacted failures.

        Args:
            path: Fixed API path.
            data: Request body.
            method: HTTP method.

        Returns:
            Parsed response.

        Raises:
            ValueError: On a redacted HTTP failure.
        """
        separator = "&" if "?" in path else "?"
        request = urllib.request.Request(
            "https://api.vercel.com"
            + path
            + separator
            + "teamId="
            + self.policy["team_id"],
            data=json.dumps(data).encode() if data is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            return json.loads(urllib.request.urlopen(request, timeout=30).read())
        except urllib.error.HTTPError as error:
            raise ValueError(f"Vercel HTTP {error.code} (body redacted)") from None

    def discovery(self, arch: str) -> dict:
        """Read public architecture-specific broker discovery anonymously.

        Args:
            arch: Requested architecture.

        Returns:
            Public broker metadata.
        """
        request = urllib.request.Request(
            f"https://{self.policy['alias']}/api/worker/release-info?architecture={arch}",
            headers={"Cache-Control": "no-cache"},
        )
        response = urllib.request.urlopen(request, timeout=30)
        require(response.url == request.full_url, "release-info redirected")
        return json.loads(response.read())

    def production(self) -> dict:
        """Observe alias and its actual READY production deployment.

        Returns:
            The verified deployment metadata.
        """
        alias = self.api(f"/v4/aliases/{self.policy['alias']}")
        deployment = self.api(f"/v13/deployments/{alias['deployment']['id']}")
        require(
            deployment["readyState"] == "READY"
            and deployment["target"] == "production"
            and deployment["projectId"] == self.policy["project_id"],
            "alias is not a READY project production deployment",
        )
        return deployment

    def resolve_deployment(self, hostname: str) -> dict:
        """Resolve the CLI receipt to its actual Vercel deployment identity.

        Args:
            hostname: Immutable deployment hostname.

        Returns:
            Observed deployment metadata.
        """
        require(
            re.fullmatch(r"[A-Za-z0-9.-]+\.vercel\.app", hostname),
            "invalid deployment receipt hostname",
        )
        deployment = self.api(f"/v13/deployments/{hostname}")
        require(
            deployment["projectId"] == self.policy["project_id"]
            and deployment["readyState"] == "READY"
            and deployment["target"] == "production",
            "local deployment receipt is not READY production",
        )
        return deployment

    def update(self, entry: dict, value: str) -> None:
        """Update only an existing exact-Production plain variable.

        Args:
            entry: Observed metadata.
            value: New public value.
        """
        require(
            entry["target"] == ["production"] and entry["type"] == "plain",
            "release variable must be exact Production plain",
        )
        self.api(
            f"/v9/projects/{self.policy['project_id']}/env/{entry['id']}",
            {"value": value, "type": "plain", "target": ["production"]},
            "PATCH",
        )


def _local_activation(policy: dict, project_path: Path = PROJECT) -> dict:
    """Validate local linkage and return the exact nonsecret marker contract.

    Args:
        policy: Checked-in capability policy.
        project_path: Vercel CLI project linkage JSON.

    Returns:
        Project-bound schema-1 activation marker.
    """
    require(policy.get("enabled") is True, "release capability disabled")
    require(
        all(
            isinstance(policy.get(k), str) and policy[k]
            for k in ("project_id", "team_id", "alias")
        ),
        "release project identity missing",
    )
    project = json.loads(project_path.read_text())
    require(
        project["projectId"] == policy["project_id"]
        and project["orgId"] == policy["team_id"],
        "local Vercel project linkage differs from release policy",
    )
    return {"schema": 1, **{k: policy[k] for k in ("project_id", "team_id", "alias")}}


def finalize(
    github: GitHubAdapter,
    registry: RegistryAdapter,
    vercel: VercelAdapter,
    tag: str,
    deployment: str,
) -> None:
    """Promote latest only after independently observing deployed production.

    Args:
        github: Trusted GitHub adapter.
        registry: OCI adapter.
        vercel: VercelAdapter adapter.
        tag: Expected immutable stage tag.
        deployment: Local deployment identity.

    Raises:
        ValueError: When no staged plan exists.
    """
    plan = github.pending()
    if plan is None:
        raise ValueError("no pending plan")
    require(
        plan["tag"] == tag and github.head() == plan["source_sha"],
        "stale or replayed plan",
    )
    for workflow, run_id, attempt in (
        ("ci.yaml", plan["ci_run_id"], plan["ci_run_attempt"]),
        ("worker-image.yaml", plan["run_id"], plan["run_attempt"]),
    ):
        observations = github.runs(workflow, plan["source_sha"])
        require(observations, "release workflow observations missing")
        run = max(
            observations, key=lambda item: (item["run_number"], item["run_attempt"])
        )
        eligible(run, plan["source_sha"], workflow)
        require(
            run["id"] == run_id and run["run_attempt"] == attempt,
            "workflow was superseded after staging",
        )
    raw, index = registry.manifest(plan["image_digest"])
    require(index["manifests"] == plan["descriptors"], "descriptor union changed")
    descriptors, identities = registry.identities(plan["indexes"])
    require(
        descriptors == plan["descriptors"] and identities == plan["identities"],
        "platform/config identity changed",
    )
    latest, _ = registry.manifest("latest")
    require(
        digest(latest) == plan["previous_digest"],
        "latest changed or finalization replayed",
    )
    validate_acceptance(
        github=github, records=plan["acceptance"], basis=plan["acceptance_basis"]
    )
    verify_production(vercel, plan, deployment)
    require(github.head() == plan["source_sha"], "main advanced before promotion")
    require(vercel.production()["id"] == deployment, "alias changed before promotion")
    require(
        digest(registry.manifest("latest")[0]) == plan["previous_digest"],
        "latest raced finalization",
    )
    registry.put("latest", raw)
    observed, index = registry.manifest("latest")
    require(
        digest(observed) == plan["image_digest"] and index["manifests"] == descriptors,
        "anonymous latest descriptor verification failed",
    )
    require(
        registry.identities(plan["indexes"])[1] == identities,
        "anonymous config verification failed",
    )


def eligible(run: dict, sha: str, workflow: str) -> None:
    """Validate server-observed workflow provenance, not caller booleans.

    Args:
        run: GitHub run response.
        sha: Current trusted main.
        workflow: Expected workflow filename.
    """
    require(
        run["head_sha"] == sha
        and run["head_branch"] == "main"
        and run["event"] == "push"
        and run["head_repository"]["full_name"] == REPOSITORY
        and run["repository"]["full_name"] == REPOSITORY
        and run["path"] == f".github/workflows/{workflow}"
        and run["conclusion"] == "success"
        and run["status"] == "completed"
        and not run["pull_requests"],
        "untrusted or unsuccessful workflow",
    )


def validate_acceptance(github: GitHubAdapter, records: dict, basis: dict) -> None:
    """Consume both reviewed retained proofs before release effects.

    Per-architecture records: record_id, reviewer, accepted=true, index_digest,
    proof_asset_id (positive GitHub release asset ID), proof_sha256.
    Baseline fields: accepted_architectures, acceptance (same records), source_sha,
    indexes, identities, run_id, run_attempt, runtime_fingerprint, guide_sha256,
    inventories. These are ORIGINAL accepted build identities, not new candidate
    identities. Operator review is not remote hardware attestation.

    Args:
        github: Fixed-repository byte retrieval adapter.
        records: Protected decisions by native architecture.
        basis: Original observed build and runtime metadata.
    """
    require(set(records) == set(ARCHES), "both retained physical proofs required")
    require(re.fullmatch(r"[0-9a-f]{40}", basis["source_sha"]), "invalid proof source")
    for key in ("run_id", "run_attempt"):
        require(type(basis[key]) is int and basis[key] > 0, "invalid image run")
    for arch in ARCHES:
        record = records[arch]
        require(
            record.get("accepted") is True
            and isinstance(record.get("record_id"), str)
            and record["record_id"]
            and isinstance(record.get("reviewer"), str)
            and record["reviewer"]
            and record.get("index_digest") == basis["indexes"][arch]
            and type(record.get("proof_asset_id")) is int
            and record["proof_asset_id"] > 0,
            "reviewed retained physical acceptance required",
        )
        expected = {
            "source_sha": basis["source_sha"],
            "architecture": arch,
            "platform": {"os": "linux", "architecture": arch},
            "index_digest": basis["indexes"][arch],
            "manifest_digest": basis["identities"][arch]["manifest"],
            "config_digest": basis["identities"][arch]["config"],
            "run_id": basis["run_id"],
            "run_attempt": basis["run_attempt"],
            "runtime_fingerprint": basis["runtime_fingerprint"],
            "guide_sha256": basis["guide_sha256"],
            "inventory": basis["inventories"][arch],
        }
        validate_physical_proof(
            raw=github.proof_asset(record["proof_asset_id"]),
            sha256=record.get("proof_sha256", ""),
            expected=expected,
        )


def validate_physical_proof(raw: bytes, sha256: str, expected: dict) -> dict:
    """Validate retained observations of a REAL documented GPU evaluation.

    Exact JSON schema: expected identity fields (source_sha, architecture,
    platform (linux/native architecture), index_digest, manifest_digest,
    config_digest, run_id, run_attempt,
    runtime_fingerprint, guide_sha256, inventory), plus schema=1,
    test_kind=manual-pinned-docker|launcher, documented_command (executed guide
    command, <=4096 chars), guide_exact=true, clean_docker_gpu_host=true,
    evaluation_completed=true, exit_code=0, transcript_asset_id (positive fixed
    repository release asset ID), transcript_sha256. Digests use sha256:64hex.
    Inventory is complete CPU-only python/dpkg metadata. Transcript reference is
    a NONSECRET REDACTED summary: NEVER device codes, credentials/private corpus.
    Diagnostic matmul is not evaluation completion. Operator review of these
    observations is not unforgeable hardware provenance.

    Args:
        raw: Exact retained bytes, at most 256 KiB.
        sha256: Protected exact-byte digest.
        expected: Original observed identities and complete inventory.

    Returns:
        Validated decoded proof.
    """
    require(
        isinstance(raw, bytes) and 0 < len(raw) <= MAX_PROOF_BYTES,
        "physical proof size invalid",
    )
    require(
        re.fullmatch(r"sha256:[0-9a-f]{64}", sha256) and digest(raw) == sha256,
        "physical proof digest mismatch",
    )
    identity_keys = {
        "source_sha",
        "architecture",
        "platform",
        "index_digest",
        "manifest_digest",
        "config_digest",
        "run_id",
        "run_attempt",
        "runtime_fingerprint",
        "guide_sha256",
        "inventory",
    }
    require(set(expected) == identity_keys, "expected proof schema mismatch")
    proof = json.loads(raw)
    keys = {
        "schema",
        "test_kind",
        "documented_command",
        "guide_exact",
        "clean_docker_gpu_host",
        "evaluation_completed",
        "exit_code",
        "transcript_asset_id",
        "transcript_sha256",
    }
    require(
        isinstance(proof, dict) and set(proof) == set(expected) | keys,
        "physical proof schema mismatch",
    )
    require(
        all(proof[k] == v for k, v in expected.items()),
        "physical proof identity mismatch",
    )
    for key in (
        "index_digest",
        "manifest_digest",
        "config_digest",
        "runtime_fingerprint",
        "guide_sha256",
        "transcript_sha256",
    ):
        require(
            isinstance(proof[key], str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", proof[key]),
            "invalid physical proof content identity",
        )
    require(proof["architecture"] in ARCHES, "invalid native architecture")
    require(
        isinstance(proof["source_sha"], str)
        and re.fullmatch(r"[0-9a-f]{40}", proof["source_sha"]),
        "invalid source",
    )
    for key in ("run_id", "run_attempt"):
        require(type(proof[key]) is int and proof[key] > 0, "invalid image run")
    require(
        isinstance(proof["inventory"], dict)
        and set(proof["inventory"]) == {"python", "dpkg"}
        and all(isinstance(v, list) and v for v in proof["inventory"].values()),
        "complete CPU inventory required",
    )
    require(
        type(proof["schema"]) is int
        and proof["schema"] == 1
        and proof["test_kind"] in ("manual-pinned-docker", "launcher")
        and isinstance(proof["documented_command"], str)
        and 0 < len(proof["documented_command"]) <= 4096
        and proof["guide_exact"] is True
        and proof["clean_docker_gpu_host"] is True
        and proof["evaluation_completed"] is True
        and type(proof["exit_code"]) is int
        and proof["exit_code"] == 0
        and type(proof["transcript_asset_id"]) is int
        and proof["transcript_asset_id"] > 0,
        "successful documented physical evaluation required",
    )
    return proof


def verify_production(vercel: VercelAdapter, plan: dict, deployment: str) -> None:
    """Require READY alias identity and both exact public discovery records.

    Args:
        vercel: VercelAdapter adapter.
        plan: Staged plan.
        deployment: Expected deployment id.
    """
    require(vercel.production()["id"] == deployment, "production alias changed")
    entries = vercel.config()
    for key, value in release_values(plan).items():
        matches = [
            e for e in entries if e["key"] == key and "production" in e["target"]
        ]
        require(
            len(matches) == 1
            and matches[0]["target"] == ["production"]
            and matches[0]["type"] == "plain"
            and matches[0]["value"] == value,
            "production configuration differs from staged plan",
        )
    for arch in ARCHES:
        info = vercel.discovery(arch)
        expected = {
            k: plan[k]
            for k in (
                "protocol_version",
                "launcher_protocol_version",
                "image_repository",
                "image_digest",
                "image_reference",
                "worker_version",
                "euroeval_version",
            )
        }
        require(
            all(info.get(k) == v for k, v in expected.items())
            and info.get("architecture") == arch,
            "production broker differs from staged plan",
        )
    require(vercel.production()["id"] == deployment, "production alias raced discovery")


def release_values(plan: dict) -> dict[str, str]:
    """Return the only permitted production configuration mutations.

    Args:
        plan: Approved staged release.

    Returns:
        Public variable values.
    """
    return {
        "VOLUNTEER_WORKER_IMAGE_DIGEST": plan["image_digest"],
        "VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64": plan["image_digest"],
        "VOLUNTEER_WORKER_VERSION": plan["worker_version"],
        "EUROEVAL_VERSION": plan["euroeval_version"],
    }


def finish(
    github: GitHubAdapter, vercel: VercelAdapter, state_path: Path = STATE
) -> None:
    """Dispatch finalization only after a verified normal local deployment.

    Args:
        github: GitHubAdapter adapter.
        vercel: VercelAdapter adapter.
        state_path: Existing recovery journal.
    """
    if not state_path.exists():
        return
    state = json.loads(state_path.read_text())
    if state["status"] == "finalization-dispatched":
        return
    require(
        state["status"] == "prepared",
        "incomplete preparation; manual recovery required",
    )
    require(
        public_config(vercel.config()) == state["prepared_config"],
        "config changed since prepare",
    )
    require(state.get("deployment_url"), "local deployment receipt missing")
    deployment = vercel.resolve_deployment(state["deployment_url"])["id"]
    require(deployment != state["previous_deployment"], "no new deployment")
    validate_acceptance(
        github=github,
        records=state["plan"]["acceptance"],
        basis=state["plan"]["acceptance_basis"],
    )
    verify_production(vercel, state["plan"], deployment)
    github.dispatch(state["plan"], deployment)
    state["status"] = "finalization-dispatched"
    state_path.write_text(json.dumps(state))


def public_config(entries: list[dict]) -> dict:
    """Fingerprint metadata and plain values without retaining any secrets.

    Args:
        entries: Observed Vercel environment entries.

    Returns:
        Safe comparison snapshot.
    """
    return {
        e["id"]: {k: e.get(k) for k in ("key", "target", "type", "updatedAt")}
        | (
            {"value": e.get("value")}
            if e["type"] == "plain"
            and e["key"]
            in (
                "VOLUNTEER_WORKER_IMAGE_DIGEST",
                "VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64",
                "VOLUNTEER_WORKER_VERSION",
                "EUROEVAL_VERSION",
            )
            else {}
        )
        for e in entries
    }


def load_activation(
    *,
    ci_only: bool = False,
    policy_path: Path | None = None,
    marker_path: Path | None = None,
    project_path: Path = PROJECT,
) -> dict | None:
    """Load the single activation gate without adapters, secrets or network access.

    Explicit false overrides everything. CI never trusts a local marker. Local
    true-environment activation remains supported, but is project-validated too.
    Missing markers are inert; malformed or stale markers fail closed.

    Args:
        ci_only: Require explicit environment activation for stage/finalize.
        policy_path: Reviewed capability policy override.
        marker_path: Nonsecret local marker override.
        project_path: Vercel project linkage.

    Returns:
        Active policy, or None when operationally disabled.
    """
    enabled = os.environ.get("WORKER_RELEASE_ENABLED")
    ci = (
        ci_only
        or os.environ.get("CI") == "true"
        or os.environ.get("GITHUB_ACTIONS") == "true"
    )
    marker_path = marker_path if marker_path is not None else ACTIVATION
    if enabled == "false" or (ci and enabled != "true"):
        return None
    if enabled != "true" and (ci or not marker_path.exists()):
        return None
    policy = json.loads(
        (policy_path if policy_path is not None else POLICY).read_text()
    )
    if policy.get("enabled") is not True:
        return None
    if not ci:
        expected = _local_activation(policy=policy, project_path=project_path)
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            require(
                isinstance(marker, dict)
                and marker == expected
                and type(marker.get("schema")) is int,
                "local activation marker differs from release policy",
            )
    return policy


def prepare(
    github: GitHubAdapter,
    registry: RegistryAdapter,
    vercel: VercelAdapter,
    policy: dict,
    local: dict,
    state_path: Path = STATE,
) -> dict | None:
    """Prepare narrowly scoped production config before the normal local build.

    Args:
        github: GitHubAdapter adapter.
        registry: OCI adapter.
        vercel: Existing authenticated Vercel adapter.
        policy: Reviewed activation policy.
        local: Observed local source.
        state_path: Nonsecret recovery journal.

    Returns:
        Prepared plan or no pending release.
    """
    if not policy.get("enabled"):
        return None
    plan = github.pending()
    if not plan:
        return None
    latest, _ = registry.manifest("latest")
    if digest(latest) == plan["image_digest"]:
        return None
    if state_path.exists():
        old = json.loads(state_path.read_text())
        if old["status"] == "finalization-dispatched" and old["plan"][
            "image_digest"
        ] == digest(latest):
            archive = state_path.with_name(old["plan"]["tag"] + ".json")
            state_path.rename(archive)
    require(
        not state_path.exists(), "unresolved release journal; manual recovery required"
    )
    require(digest(latest) == plan["previous_digest"], "previous release changed")
    require(
        all(local[k] == plan[k] for k in local),
        "local versions/source/policy differ from staged plan",
    )
    require(github.head() == plan["source_sha"], "main advanced")
    validate_acceptance(
        github=github, records=plan["acceptance"], basis=plan["acceptance_basis"]
    )
    # The trusted staged plan carries the external reviewed decisions. Local
    # deployments must not need protected CI acceptance variables in their shell.
    before = vercel.production()["id"]
    entries = vercel.config()
    values = release_values(plan)
    selected = {}
    for key in values:
        matches = [
            e for e in entries if e["key"] == key and "production" in e["target"]
        ]
        require(
            len(matches) == 1
            and matches[0]["type"] == "plain"
            and matches[0]["target"] == ["production"],
            "missing exact Production plain release variable",
        )
        selected[key] = matches[0]
    require(
        all(
            selected[key]["value"] == plan["previous_digest"]
            for key in (
                "VOLUNTEER_WORKER_IMAGE_DIGEST",
                "VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64",
            )
        ),
        "production is not on the expected previous release",
    )
    state = {
        "plan": plan,
        "previous_deployment": before,
        "before_config": public_config(entries),
        "status": "preparing",
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state))
    for key, value in values.items():
        if selected[key]["value"] == value:
            continue
        require(
            github.head() == plan["source_sha"] and vercel.production()["id"] == before,
            "source or alias changed during prepare",
        )
        require(
            public_config(vercel.config()) == public_config(entries),
            "intervening config mutation",
        )
        vercel.update(selected[key], value)
        observed = vercel.config()
        expected = public_config(entries)
        actual = public_config(observed)
        identity = selected[key]["id"]
        expected[identity]["value"] = value
        expected[identity]["updatedAt"] = actual[identity]["updatedAt"]
        require(
            actual == expected,
            "unexpected config mutation during update; manual recovery required",
        )
        entries = observed
    state.update(status="prepared", prepared_config=public_config(entries))
    state_path.write_text(json.dumps(state))
    return plan


def source() -> dict:
    """Observe checked-out source versions, policy bytes and runtime fingerprint.

    Returns:
        Nonsecret source metadata.

    Raises:
        ValueError: When the worker version is absent.
    """
    version = str(
        Version(tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"])
    )
    command(
        [
            "uv",
            "run",
            "python",
            "src/scripts/generate_volunteer_scope_policy.py",
            "--version",
            version,
            "--check",
        ]
    )
    require(
        not command(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                "Dockerfile.worker",
                ".dockerignore",
                "VOLUNTEER_WORKER.md",
                "VOLUNTEER_WORKER_OPERATIONS.md",
                "src/scripts/run_volunteer_worker.py",
                "src/scripts/release_volunteer_worker.py",
                "src/scripts/deploy_vercel_frontend.py",
                "src/scripts/worker-release-policy.json",
                "src/scripts/worker_release_inventory.py",
                "src/scripts/prepare_vercel_worktree.py",
                "src/scripts/build_vercel_frontend.py",
                "src/scripts/verify_vercel_functions.py",
                ".github/workflows/worker-image.yaml",
                ".github/workflows/worker-release.yaml",
                ".github/workflows/ci.yaml",
                "Makefile",
                "pyproject.toml",
                "uv.lock",
                "src/euroeval",
                "src/euroeval_worker",
                *POLICIES,
            ]
        ).strip(),
        "release checkout has tracked modifications",
    )
    worker = re.search(
        r'__version__ = "([^"]+)"', Path("src/euroeval_worker/__init__.py").read_text()
    )
    if worker is None:
        raise ValueError("worker version missing")
    files = (
        command(
            [
                "git",
                "ls-files",
                "src/scripts/run_volunteer_worker.py",
                "Dockerfile.worker",
                ".dockerignore",
                "pyproject.toml",
                "uv.lock",
                "src/euroeval",
                "src/euroeval_worker",
            ]
        )
        .decode()
        .splitlines()
    )
    runtime = b"".join(
        p.encode() + b"\0" + Path(p).read_bytes()
        for p in files
        if not p.startswith("src/euroeval/dataset_configs/")
    )
    return {
        "source_sha": command(["git", "rev-parse", "HEAD"]).decode().strip(),
        "euroeval_version": version,
        "worker_version": worker[1],
        "policy_hash": digest(b"".join(Path(p).read_bytes() for p in POLICIES)),
        "runtime_fingerprint": digest(runtime),
        "guide_sha256": digest(Path("VOLUNTEER_WORKER.md").read_bytes()),
    }


def stage(
    github: GitHubAdapter, registry: RegistryAdapter, policy: dict, local: dict
) -> dict | None:
    """Stage an accepted current-main candidate without touching production/latest.

    Args:
        github: Injectable trusted GitHub adapter.
        registry: Injectable OCI adapter.
        policy: Reviewed acceptance policy.
        local: Observed source metadata.

    Returns:
        Published public plan or no eligible work.
    """
    if not policy.get("enabled"):
        return None
    sha = github.head()
    require(
        re.fullmatch(r"[0-9a-f]{40}", sha) and local["source_sha"] == sha,
        "stale source",
    )
    runs = {}
    for workflow in ("ci.yaml", "worker-image.yaml"):
        candidates = github.runs(workflow, sha)
        if not candidates:
            return None
        run = max(candidates, key=lambda r: (r["run_number"], r["run_attempt"]))
        if run["status"] != "completed":
            return None
        eligible(run, sha, workflow)
        runs[workflow] = run
    run = runs["worker-image.yaml"]
    inventories = {}
    indexes = {}
    artifacts = {}
    for arch in ARCHES:
        item = github.artifact(
            run, f"worker-release-{arch}-{run['id']}-{run['run_attempt']}"
        )
        require(
            item["source_sha"] == sha
            and item["run_id"] == run["id"]
            and item["run_attempt"] == run["run_attempt"]
            and item["architecture"] == arch,
            "artifact identity mismatch",
        )
        require(
            re.fullmatch(r"sha256:[0-9a-f]{64}", item["index_digest"]),
            "invalid candidate index",
        )
        require(
            item["inventory"]["python"] and item["inventory"]["dpkg"],
            "empty runtime inventory",
        )
        indexes[arch] = item["index_digest"]
        inventories[arch] = item["inventory"]
        artifacts[arch] = item["artifact_id"]
    descriptors, identities = registry.identities(indexes)
    candidate = {
        "source_sha": sha,
        "indexes": indexes,
        "identities": identities,
        "run_id": run["id"],
        "run_attempt": run["run_attempt"],
        "runtime_fingerprint": local["runtime_fingerprint"],
        "guide_sha256": local["guide_sha256"],
        "inventories": inventories,
    }
    baseline = policy.get("baseline", {})
    fast = (
        baseline.get("accepted_architectures") == list(ARCHES)
        and baseline.get("runtime_fingerprint") == local["runtime_fingerprint"]
        and baseline.get("guide_sha256") == local["guide_sha256"]
        and baseline.get("inventories") == inventories
    )
    basis = baseline if fast else candidate
    acceptance = (
        baseline.get("acceptance", {})
        if fast
        else policy.get("acceptance", {}).get(sha, {})
    )
    validate_acceptance(github=github, records=acceptance, basis=basis)
    raw = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": descriptors,
        },
        separators=(",", ":"),
    ).encode()
    tag = f"worker-release-{sha}-{run['id']}-{run['run_attempt']}"
    previous = github.pending()
    if previous and previous["tag"] == tag:
        return None
    latest, _ = registry.manifest("latest")
    require(
        not previous or previous["image_digest"] == digest(latest),
        "previous staged release must be finalized first",
    )
    require(github.head() == sha, "main advanced before staging")
    image_digest = registry.put(tag, raw)
    plan = {
        **local,
        "schema": 1,
        "tag": tag,
        "image_repository": IMAGE,
        "image_digest": image_digest,
        "image_reference": f"{IMAGE}@{image_digest}",
        "protocol_version": "volunteer-worker/v1",
        "launcher_protocol_version": "volunteer-launcher/v1",
        "indexes": indexes,
        "descriptors": descriptors,
        "identities": identities,
        "artifacts": artifacts,
        "run_id": run["id"],
        "run_attempt": run["run_attempt"],
        "ci_run_id": runs["ci.yaml"]["id"],
        "ci_run_attempt": runs["ci.yaml"]["run_attempt"],
        "previous_digest": digest(latest),
        "inventories": inventories,
        "acceptance": acceptance,
        "acceptance_basis": {k: basis[k] for k in candidate},
    }
    require(github.head() == sha, "main advanced before metadata publication")
    github.publish(plan)
    return plan


if __name__ == "__main__":
    raise SystemExit(main())
