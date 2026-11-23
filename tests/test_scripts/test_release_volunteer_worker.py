"""Public-interface tests with fake GitHub, OCI and Vercel observations only."""

import copy
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from scripts import deploy_vercel_frontend, worker_release_inventory
from scripts import release_volunteer_worker as release
from scripts.worker_release_inventory import collect

SHA = "a" * 40


@pytest.fixture
def activation_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a nonsecret linked checkout, without credentials or adapters.

    Args:
        tmp_path: Isolated checkout directory.
        monkeypatch: Environment and working-directory isolation.

    Returns:
        Local activation marker path.
    """
    policy = json.loads(release.POLICY.read_text())
    monkeypatch.chdir(tmp_path)
    for key in ("WORKER_RELEASE_ENABLED", "CI", "GITHUB_ACTIONS"):
        monkeypatch.delenv(key, raising=False)
    release.POLICY.parent.mkdir(parents=True)
    release.POLICY.write_text(json.dumps(policy))
    release.PROJECT.parent.mkdir()
    release.PROJECT.write_text(
        json.dumps({"projectId": policy["project_id"], "orgId": policy["team_id"]})
    )
    return release.ACTIVATION


@pytest.fixture
def source_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide committed source observations without running tools or using Git.

    Args:
        tmp_path: Isolated source directory.
        monkeypatch: Working-directory and command isolation.
    """
    files = {
        "pyproject.toml": '[project]\nversion = "2.0.0"\n',
        "src/euroeval_worker/__init__.py": '__version__ = "1.0.1"\n',
        "src/scripts/run_volunteer_worker.py": "# original launcher\n",
        "VOLUNTEER_WORKER.md": "Original guide\n",
        "src/euroeval/dataset_configs/example.py": "# original dataset\n",
        **dict.fromkeys(release.POLICIES, "{}\n"),
    }
    monkeypatch.chdir(tmp_path)
    for name, content in files.items():
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def observed(args: list[str], data: bytes | None = None) -> bytes:
        """Emulate committed Git paths, restricting listings to requested inputs.

        Args:
            args: Requested command arguments.
            data: Unused command input.

        Returns:
            Local source observations; policy generation is a no-op.
        """
        del data
        if args[:2] == ["git", "ls-files"]:
            return "\n".join(
                name
                for name in sorted(files)
                if any(name == p or name.startswith(p + "/") for p in args[2:])
            ).encode()
        if args[:2] == ["git", "rev-parse"]:
            return SHA.encode()
        assert args[:2] in (["git", "status"], ["uv", "run"])
        return b""

    monkeypatch.setattr(release, "command", observed)


@pytest.mark.parametrize("contents", ["not-json", "[]", "null", "{}"])
def test_activation_malformed(activation_checkout: Path, contents: str) -> None:
    """Malformed local state fails closed before adapter creation."""
    activation_checkout.write_text(contents)
    with pytest.raises(ValueError):
        release.load_activation()


@pytest.mark.parametrize("field", ["projectId", "orgId"])
def test_activation_rejects_relinked_project(
    activation_checkout: Path, field: str
) -> None:
    """Setup and subsequent hooks both refuse a different Vercel linkage."""
    assert release.main(["configure-local"]) == 0
    project = json.loads(release.PROJECT.read_text())
    project[field] = "wrong"
    release.PROJECT.write_text(json.dumps(project))
    assert release.main(["configure-local"]) == 1
    with pytest.raises(ValueError):
        release.load_activation()


@pytest.mark.parametrize("field", ["project_id", "team_id", "alias", "schema"])
def test_activation_rejects_wrong_marker(activation_checkout: Path, field: str) -> None:
    """Marker identity and schema cannot drift from reviewed project policy."""
    assert release.main(["configure-local"]) == 0
    marker = json.loads(activation_checkout.read_text())
    marker[field] = "wrong"
    activation_checkout.write_text(json.dumps(marker))
    with pytest.raises(ValueError):
        release.load_activation()


def test_artifact_and_empty_acceptance_refused() -> None:
    """Cross-source artifacts and unreviewed default acceptance fail closed."""
    github, registry, policy, local = inputs()
    github.bad_artifact = True
    with pytest.raises(ValueError):
        release.stage(github, registry, policy, local)
    github.bad_artifact = False
    policy["acceptance"] = {}
    with pytest.raises(ValueError):
        release.stage(github, registry, policy, local)
    assert not registry.writes


def inputs() -> tuple:
    """Create a reviewed candidate-specific physical acceptance fixture.

    Returns:
        Injectable adapters, policy and local source observations.
    """
    github, registry = FakeGitHub(), FakeRegistry()
    local = {
        "source_sha": SHA,
        "euroeval_version": "2.0.0",
        "worker_version": "1.0.1",
        "policy_hash": "policy",
        "runtime_fingerprint": release.digest(b"runtime"),
        "guide_sha256": release.digest(b"guide"),
    }
    policy: dict = {
        "enabled": True,
        "acceptance": {
            SHA: {
                a: {
                    "index_digest": "sha256:" + ("1" if a == "amd64" else "2") * 64,
                    "accepted": True,
                    "record_id": "physical-log",
                    "reviewer": "maintainer",
                    "proof_asset_id": 0,
                    "proof_sha256": "",
                }
                for a in release.ARCHES
            }
        },
    }
    for number, arch in enumerate(release.ARCHES, start=1):
        item = github.artifact({"id": 12}, arch)
        proof = {
            "schema": 1,
            "source_sha": SHA,
            "architecture": arch,
            "platform": {"os": "linux", "architecture": arch},
            "index_digest": item["index_digest"],
            "manifest_digest": item["index_digest"],
            "config_digest": "sha256:" + "4" * 64,
            "run_id": 12,
            "run_attempt": 2,
            "runtime_fingerprint": local["runtime_fingerprint"],
            "guide_sha256": local["guide_sha256"],
            "inventory": item["inventory"],
            "test_kind": "manual-pinned-docker",
            "documented_command": "synthetic unit fixture, NOT physical evidence",
            "guide_exact": True,
            "clean_docker_gpu_host": True,
            "evaluation_completed": True,
            "exit_code": 0,
            "transcript_asset_id": 100 + number,
            "transcript_sha256": release.digest(b"redacted synthetic summary"),
        }
        raw = json.dumps(proof).encode()
        github.proofs[number] = raw
        policy["acceptance"][SHA][arch].update(
            proof_asset_id=number, proof_sha256=release.digest(raw)
        )
    return github, registry, policy, local


class FakeGitHub:
    """In-memory GitHub observations and publication effects."""

    def __init__(self) -> None:
        """Start with two successful trusted workflows and no pending release."""
        self.plan = None
        self.sha = SHA
        self.run_overrides = {}
        self.bad_artifact = False
        self.proofs = {}

    def artifact(self, run: dict, name: str) -> dict:
        """Observe attempt-bound inventory metadata.

        Args:
            run: Selected run.
            name: Requested artifact name.

        Returns:
            Candidate artifact contents.
        """
        arch = "arm64" if "arm64" in name else "amd64"
        return {
            "source_sha": "b" * 40 if self.bad_artifact else SHA,
            "run_id": run["id"],
            "run_attempt": 2,
            "architecture": arch,
            "index_digest": "sha256:" + ("1" if arch == "amd64" else "2") * 64,
            "artifact_id": arch,
            "inventory": {"python": [["torch", "2"]], "dpkg": ["libc=1"]},
        }

    def head(self) -> str:
        """Observe main.

        Returns:
            Current main SHA.
        """
        return self.sha

    def pending(self) -> dict | None:
        """Observe pending plan.

        Returns:
            Current staged plan.
        """
        return self.plan

    def proof_asset(self, asset_id: int) -> bytes:
        """Return synthetic retained bytes, never physical hardware evidence.

        Args:
            asset_id: Fixture identity.

        Returns:
            Exact fixture bytes.
        """
        return self.proofs[asset_id]

    def publish(self, plan: dict) -> None:
        """Record public JSON only.

        Args:
            plan: Published plan.
        """
        self.plan = plan

    def runs(self, workflow: str, sha: str) -> list[dict]:
        """Observe successful workflow provenance.

        Args:
            workflow: Workflow filename.
            sha: Requested commit.

        Returns:
            Fake workflow observations.
        """
        return [
            dict(
                id=12,
                run_number=3,
                run_attempt=2,
                head_sha=sha,
                head_branch="main",
                event="push",
                head_repository={"full_name": release.REPOSITORY},
                repository={"full_name": release.REPOSITORY},
                path=f".github/workflows/{workflow}",
                conclusion="success",
                status="completed",
                pull_requests=[],
                **{},
            )
            | self.run_overrides
        ]


class FakeRegistry:
    """In-memory OCI indexes retaining arbitrary attestation descriptors."""

    def __init__(self) -> None:
        """Seed the prior immutable release."""
        self.raw = b'{"schemaVersion":2,"manifests":[]}'
        self.storage = {"latest": self.raw}
        self.writes = []
        self.descriptors = [
            {
                "digest": "sha256:" + "3" * 64,
                "platform": {"architecture": "unknown", "os": "unknown"},
                "annotations": {"vnd.docker.reference.type": "attestation-manifest"},
            }
        ]

    def identities(self, indexes: dict) -> tuple[list, dict]:
        """Return complete descriptors and config identities.

        Args:
            indexes: Original candidate index identities.

        Returns:
            Descriptor union and config mapping.
        """
        return copy.deepcopy(self.descriptors), {
            a: {"manifest": indexes[a], "config": "sha256:" + "4" * 64}
            for a in release.ARCHES
        }

    def manifest(self, reference: str) -> tuple[bytes, dict]:
        """Read a fake immutable manifest.

        Args:
            reference: Tag or digest.

        Returns:
            Bytes and decoded manifest.
        """
        raw = self.storage[reference]
        return raw, json.loads(raw)

    def put(self, tag: str, raw: bytes) -> str:
        """Record an index publication.

        Args:
            tag: Destination.
            raw: Exact original bytes.

        Returns:
            Content digest.
        """
        self.writes.append(tag)
        self.storage[tag] = raw
        self.storage[release.digest(raw)] = raw
        return release.digest(raw)


@pytest.mark.usefixtures("source_checkout")
@pytest.mark.parametrize(
    "path,changed_field",
    [
        ("VOLUNTEER_WORKER.md", "guide_sha256"),
        ("src/scripts/run_volunteer_worker.py", "runtime_fingerprint"),
        ("src/euroeval/dataset_configs/example.py", None),
        ("src/scripts/release_volunteer_worker.py", None),
        ("VOLUNTEER_WORKER_OPERATIONS.md", None),
    ],
)
def test_baseline_binds_observed_guide_and_launcher(
    path: str, changed_field: str | None
) -> None:
    """Release controls reuse unchanged runtime proof; guide/launcher changes do not.

    Args:
        path: Source input changed in the next candidate.
        changed_field: Metadata that must change, or None for a dataset-only edit.
    """
    github, registry, policy, local = inputs()
    local.update(release.source())
    for asset_id, raw in github.proofs.items():
        proof = json.loads(raw) | {
            key: local[key] for key in ("runtime_fingerprint", "guide_sha256")
        }
        raw = json.dumps(proof).encode()
        github.proofs[asset_id] = raw
        policy["acceptance"][SHA][proof["architecture"]]["proof_sha256"] = (
            release.digest(raw)
        )
    old = release.stage(github, registry, policy, local)
    assert old is not None
    policy["baseline"] = {
        **old["acceptance_basis"],
        "accepted_architectures": list(release.ARCHES),
        "acceptance": copy.deepcopy(old["acceptance"]),
    }
    policy["acceptance"] = {}
    github.plan = None
    registry.writes.clear()
    Path(path).write_text("# committed candidate change\n")
    candidate = release.source()
    for key in ("runtime_fingerprint", "guide_sha256"):
        assert (candidate[key] != local[key]) == (key == changed_field)

    if changed_field is None:
        new = release.stage(github, registry, policy, candidate)
        assert new is not None
        assert new["acceptance_basis"] == old["acceptance_basis"]
        return

    with pytest.raises(ValueError, match="both retained physical proofs required"):
        release.stage(github, registry, policy, candidate)
    assert not registry.writes
    assert github.plan is None

    # Fresh candidate-specific reviewed proofs remain valid for changed inputs.
    policy["acceptance"][SHA] = copy.deepcopy(old["acceptance"])
    for asset_id, raw in github.proofs.items():
        proof = json.loads(raw) | {
            key: candidate[key] for key in ("runtime_fingerprint", "guide_sha256")
        }
        raw = json.dumps(proof).encode()
        github.proofs[asset_id] = raw
        policy["acceptance"][SHA][proof["architecture"]]["proof_sha256"] = (
            release.digest(raw)
        )
    new = release.stage(github, registry, policy, candidate)
    assert new is not None
    assert new["acceptance_basis"][changed_field] == candidate[changed_field]
    assert new["acceptance_basis"] != old["acceptance_basis"]


def test_baseline_requires_actual_inventory_equality() -> None:
    """Mutable apt changes invalidate dataset-configuration-only reuse."""
    github, registry, policy, local = inputs()
    first = release.stage(github, registry, policy, local)
    assert first is not None
    accepted = policy.pop("acceptance")[SHA]
    github.plan = None
    registry.writes.clear()
    inventories = {
        a: github.artifact({"id": 12}, a)["inventory"] for a in release.ARCHES
    }
    policy["baseline"] = {
        **first["acceptance_basis"],
        "accepted_architectures": list(release.ARCHES),
        "acceptance": accepted,
        "runtime_fingerprint": local["runtime_fingerprint"],
        "inventories": inventories,
    }
    policy["baseline"]["inventories"]["arm64"]["dpkg"] = ["libc=changed"]
    with pytest.raises(ValueError):
        release.stage(github, registry, policy, local)
    policy["baseline"]["inventories"]["arm64"]["dpkg"] = ["libc=1"]
    assert release.stage(github, registry, policy, local)


def test_baseline_reuses_original_source_and_proof_references() -> None:
    """Unchanged runtime can reuse old source proofs, never relabel them as new."""
    github, registry, policy, local = inputs()
    old = release.stage(github, registry, policy, local)
    assert old is not None
    policy["baseline"] = {
        **old["acceptance_basis"],
        "accepted_architectures": list(release.ARCHES),
        "acceptance": old["acceptance"],
    }
    policy["acceptance"] = {}
    github.plan = None
    github.sha = "b" * 40
    local["source_sha"] = github.sha
    # Artifact adapter still reports old SHA, so override only observed source.
    original = github.artifact

    def artifact(run: dict, name: str) -> dict:
        """Observe the new build's unchanged runtime.

        Args:
            run: Image run.
            name: Inventory name.

        Returns:
            New source inventory metadata.
        """
        return original(run, name) | {"source_sha": github.sha}

    github.artifact = artifact
    new = release.stage(github, registry, policy, local)
    assert new is not None
    assert new["acceptance"] == old["acceptance"]
    assert new["acceptance_basis"]["source_sha"] == SHA
    assert new["source_sha"] != SHA


@pytest.mark.parametrize("operation", ["prepare", "predeploy", "finish"])
def test_ci_cannot_deploy_or_promote(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Active CI cannot invoke local production operations or construct adapters.

    Args:
        monkeypatch: Patch activation without credentials.
        operation: Local release operation attempted by CI.
    """
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(release, "load_activation", lambda **_: {"enabled": True})
    assert release.main([operation]) == 1


def test_click_and_legacy_main_disabled_and_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both public CLI seams stay inert when disabled and report usage failures."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WORKER_RELEASE_ENABLED", raising=False)
    runner = CliRunner()
    for phase in ("stage", "prepare", "predeploy", "finish"):
        assert runner.invoke(release.cli, [phase]).exit_code == 0
        assert release.main([phase]) == 0
    assert runner.invoke(release.cli, ["invalid"]).exit_code == 2
    assert release.main(["invalid"]) == 2
    assert release.main(["finalize"]) == 2
    assert release.main(["--help"]) == 0


def test_click_enabled_uses_adapter_seams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Click and legacy invocation reach the same injected stage implementation."""
    github, registry, policy, local = inputs()
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    monkeypatch.setattr(release, "POLICY", policy_path)
    monkeypatch.setattr(release, "GitHub", lambda: github)
    monkeypatch.setattr(release, "Registry", lambda: registry)
    monkeypatch.setattr(release, "source", lambda: local)
    monkeypatch.setenv("WORKER_RELEASE_ENABLED", "true")
    assert CliRunner().invoke(release.cli, ["stage"]).exit_code == 0
    assert github.plan is not None
    assert release.main(["stage"]) == 0
    assert len(registry.writes) == 1


def test_complete_two_phase_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Latest stays unchanged until local deployment and independent verification."""
    github, registry, policy, local = inputs()
    plan = release.stage(github, registry, policy, local)
    assert plan is not None
    assert registry.writes == [plan["tag"]]
    assert plan["descriptors"] == registry.descriptors
    vercel = FakeVercel(plan)
    journal = tmp_path / "state.json"
    local_policy = {**policy, "acceptance": {}, "baseline": {}}
    release.prepare(github, registry, vercel, local_policy, local, journal)
    assert "DO-NOT-PERSIST" not in journal.read_text()
    values = {e["key"]: e["value"] for e in vercel.entries}
    assert values["HF_CANARY_TOKEN"] == "DO-NOT-PERSIST"
    assert (
        values["VOLUNTEER_WORKER_IMAGE_DIGEST"]
        == values["VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64"]
        == plan["image_digest"]
    )
    with pytest.raises(ValueError):
        release.finish(github, registry, vercel, local, journal)
    assert "latest" not in registry.writes
    vercel.deployment = "new-ready"
    monkeypatch.setenv("WORKER_RELEASE_ENABLED", "true")
    receipt_policy = tmp_path / "receipt-policy.json"
    receipt_policy.write_text(json.dumps(policy))
    monkeypatch.setenv("CI", "true")
    deploy_vercel_frontend.record_release_deployment(
        output="Worker release deployment receipt: https://local-release.vercel.app",
        state_path=journal,
        policy_path=receipt_policy,
    )
    release.finish(github, registry, vercel, local, journal)
    assert registry.writes[-1] == "latest"
    assert json.loads(journal.read_text())["status"] == "completed"
    release.finish(github, registry, vercel, local, journal)
    assert registry.writes.count("latest") == 1


class FakeVercel:
    """In-memory production config and architecture-specific discovery."""

    def __init__(self, plan: dict) -> None:
        """Start with an old alias and unrelated encrypted secrets.

        Args:
            plan: Pending release.
        """
        self.plan = plan
        self.deployment = "old"
        self.alias_race = False
        self.bad_arch = False
        self.entries = [
            {
                "id": k,
                "key": k,
                "target": ["production"],
                "type": "plain",
                "value": plan.get("previous_digest", "old") if "DIGEST" in k else "old",
                "updatedAt": 1,
            }
            for k in release.release_values(plan)
        ]
        self.entries.append(
            {
                "id": "secret",
                "key": "HF_CANARY_TOKEN",
                "target": ["production"],
                "type": "encrypted",
                "value": "DO-NOT-PERSIST",
                "updatedAt": 1,
            }
        )

    def config(self) -> list[dict]:
        """Observe current configuration.

        Returns:
            Fake environment entries.
        """
        return copy.deepcopy(self.entries)

    def discovery(self, arch: str) -> dict:
        """Observe architecture-specific broker metadata, optionally racing alias.

        Args:
            arch: Architecture.

        Returns:
            Discovery fields.
        """
        if self.alias_race:
            self.deployment = "intervening"
        return self.plan | {"architecture": "amd64" if self.bad_arch else arch}

    def production(self) -> dict:
        """Observe alias/deployment identity.

        Returns:
            Current alias deployment.
        """
        return {"id": self.deployment}

    def resolve_deployment(self, hostname: str) -> dict:
        """Resolve the actual local deploy receipt independently of the alias.

        Args:
            hostname: Immutable deployment hostname.

        Returns:
            Local deployment identity.

        Raises:
            ValueError: When the receipt is not a valid deployment hostname.
        """
        if hostname != "local-release.vercel.app":
            raise ValueError("invalid deployment hostname")
        return {"id": "new-ready"}

    def update(self, entry: dict, value: str) -> None:
        """Apply one public variable mutation.

        Args:
            entry: Target metadata.
            value: Public value.
        """
        for e in self.entries:
            if e["id"] == entry["id"]:
                e["value"] = value
                e["updatedAt"] += 1


def test_configure_disable_local(activation_checkout: Path) -> None:
    """One-time setup arms local hooks only, and disable is idempotent."""
    assert release.load_activation() is None
    assert release.main(["configure-local"]) == 0
    marker = json.loads(activation_checkout.read_text())
    assert set(marker) == {"schema", "project_id", "team_id", "alias"}
    assert release.load_activation() is not None
    assert release.load_activation(ci_only=True) is None
    assert release.main(["disable-local"]) == 0
    assert release.main(["disable-local"]) == 0
    assert release.load_activation() is None


def test_disabled_hooks_never_construct_adapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default installation is inert and needs no credentials or pending release."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WORKER_RELEASE_ENABLED", raising=False)
    for name in ("stage", "prepare", "predeploy", "finish"):
        assert release.main([name]) == 0


@pytest.mark.parametrize("race", [False, True])
def test_finalizer_requires_both_architectures_and_stable_alias(race: bool) -> None:
    """Wrong ARM discovery or alias movement must never promote latest."""
    github, registry, policy, local = inputs()
    plan = release.stage(github, registry, policy, local)
    assert plan is not None
    vercel = FakeVercel(plan)
    vercel.deployment = "new-ready"
    for entry in vercel.entries:
        if entry["key"] in release.release_values(plan):
            entry["value"] = release.release_values(plan)[str(entry["key"])]
    vercel.alias_race = race
    vercel.bad_arch = not race
    with pytest.raises(ValueError):
        release.finalize(github, registry, vercel, plan["tag"], "new-ready")
    assert "latest" not in registry.writes


def test_github_artifact_observation_checks_archive_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real artifact adapter rejects mismatched archive bytes and ignores URLs."""
    payload = b'{"source_sha":"candidate"}'
    item = {
        "id": 31,
        "name": "attempt-2",
        "expired": False,
        "workflow_run": {"head_sha": SHA},
        "digest": release.digest(payload),
        "archive_download_url": "https://untrusted.example/steal",
    }
    requests = []

    def api(self: release.GitHub, path: str, data: dict | None = None) -> dict:
        """Return observed artifact metadata.

        Args:
            self: Adapter instance.
            path: Requested repository endpoint.
            data: Optional request body.

        Returns:
            Fake GitHub artifact listing.
        """
        return {"artifacts": [item]}

    def execute(args: list[str], data: bytes | None = None) -> bytes:
        """Return corrupted download bytes through the public transport seam.

        Args:
            args: Requested tool operation.
            data: Optional stdin.

        Returns:
            Deliberately corrupted archive bytes.
        """
        requests.append(args)
        return b"corrupted"

    monkeypatch.setattr(release.GitHub, "api", api)
    monkeypatch.setattr(release, "command", execute)
    with pytest.raises(ValueError, match="archive digest mismatch"):
        release.GitHub().artifact({"id": 12, "head_sha": SHA}, "attempt-2")
    assert requests == [
        ["gh", "api", "repos/EuroEval/EuroEval/actions/artifacts/31/zip"]
    ]


def test_incomplete_workflow_waits_without_effects() -> None:
    """Either completion order waits for the other workflow without publishing."""
    github, registry, policy, local = inputs()
    github.run_overrides = {"status": "in_progress", "conclusion": None}
    assert release.stage(github, registry, policy, local) is None
    assert not registry.writes


def test_inventory_uses_only_cpu_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime inventory never requests GPUs, imports inference or uses a network."""
    calls = []

    def execute(args: list[str], data: bytes | None = None) -> bytes:
        """Capture public command requests.

        Args:
            args: Tool arguments.
            data: Optional stdin.

        Returns:
            CPU metadata fixture.
        """
        calls.append((args, data))
        return b'{"python":[["torch","2"]],"dpkg":["libc=1"]}'

    monkeypatch.setattr(worker_release_inventory, "command", execute)
    item = collect(
        architecture="arm64",
        index_digest="sha256:" + "f" * 64,
        source_sha=SHA,
        run_id=12,
        run_attempt=2,
    )
    assert item["run_attempt"] == 2
    args = calls[0][0]
    assert "--network=none" in args and "--read-only" in args
    assert "--gpus" not in args and "import torch" not in args[-1]


@pytest.mark.parametrize("write_succeeded", [False, True])
def test_local_finish_recovers_ambiguous_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_succeeded: bool
) -> None:
    """Retry revalidates production and never blindly repeats a successful push.

    Args:
        tmp_path: Temporary journal directory.
        monkeypatch: Patch registry publication failure.
        write_succeeded: Whether the remote write occurred before losing its response.
    """
    github, registry, policy, local = inputs()
    plan = release.stage(github, registry, policy, local)
    assert plan is not None
    vercel = FakeVercel(plan)
    journal = tmp_path / "state.json"
    release.prepare(github, registry, vercel, policy, local, journal)
    state = json.loads(journal.read_text())
    state["deployment_url"] = "local-release.vercel.app"
    journal.write_text(json.dumps(state))
    vercel.deployment = "new-ready"
    original = registry.put

    def fail_publication(tag: str, raw: bytes) -> str:
        """Lose the publication response, optionally after the remote write.

        Args:
            tag: Registry destination.
            raw: Published bytes.

        Raises:
            OSError: When the publication response is lost.
        """
        if write_succeeded:
            original(tag=tag, raw=raw)
        raise OSError("publication response lost")

    monkeypatch.setattr(registry, "put", fail_publication)
    with pytest.raises(OSError):
        release.finish(github, registry, vercel, local, journal)
    assert json.loads(journal.read_text())["status"] == "prepared"
    monkeypatch.setattr(registry, "put", original)
    release.finish(github, registry, vercel, local, journal)
    assert registry.writes.count("latest") == 1
    assert json.loads(journal.read_text())["status"] == "completed"


@pytest.mark.parametrize(
    "mutation",
    ["receipt", "source", "attempt", "proof", "config", "arm", "alias", "latest"],
)
def test_local_finish_refuses_changed_evidence(tmp_path: Path, mutation: str) -> None:
    """Local promotion requires the actual receipt and unchanged release evidence.

    Args:
        tmp_path: Temporary journal directory.
        mutation: Required observation changed after deployment.
    """
    github, registry, policy, local = inputs()
    plan = release.stage(github, registry, policy, local)
    assert plan is not None
    vercel = FakeVercel(plan)
    journal = tmp_path / "state.json"
    release.prepare(github, registry, vercel, policy, local, journal)
    state = json.loads(journal.read_text())
    state["deployment_url"] = "local-release.vercel.app"
    vercel.deployment = "new-ready"
    if mutation == "receipt":
        del state["deployment_url"]
    elif mutation == "source":
        local = local | {"source_sha": "b" * 40}
    elif mutation == "attempt":
        github.run_overrides = {"run_attempt": 3}
    elif mutation == "proof":
        github.proofs[1] = b"changed"
    elif mutation == "config":
        vercel.entries[0]["value"] = "changed"
    elif mutation == "arm":
        vercel.bad_arch = True
    elif mutation == "alias":
        vercel.alias_race = True
    else:
        registry.storage["latest"] = b'{"schemaVersion":2,"manifests":[{}]}'
    journal.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        release.finish(github, registry, vercel, local, journal)
    assert "latest" not in registry.writes
    assert json.loads(journal.read_text())["status"] == "prepared"


@pytest.mark.parametrize("flag", ["CI", "GITHUB_ACTIONS", "WORKER_RELEASE_ENABLED"])
def test_local_marker_cannot_override_external_disable(
    activation_checkout: Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    """CI requires explicit true and explicit false overrides a valid marker."""
    assert release.main(["configure-local"]) == 0
    monkeypatch.setenv(flag, "false" if flag == "WORKER_RELEASE_ENABLED" else "true")
    assert release.load_activation() is None
    assert release.load_activation(ci_only=True) is None


def test_no_benchmark_data_transport() -> None:
    """Release tooling and workflow contain no benchmark archive mechanism."""
    workflow = Path(".github/workflows/worker-release.yaml").read_text()
    assert "vercel build" not in workflow and "make frontend" not in workflow
    assert "upload-artifact" not in workflow
    assert (
        "csv" not in Path("src/scripts/release_volunteer_worker.py").read_text().lower()
    )


def test_no_marker_hooks_do_not_construct_adapters(
    activation_checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installed configured capability alone has no operational authority."""

    def forbidden() -> None:
        """Reject adapter construction in disabled hooks."""
        pytest.fail("disabled hook constructed an adapter")

    monkeypatch.setattr(release, "GitHub", forbidden)
    monkeypatch.setattr(release, "Registry", forbidden)
    monkeypatch.setattr(release, "Vercel", forbidden)
    for operation in ("stage", "prepare", "predeploy", "finish"):
        assert release.main([operation]) == 0
    assert release.main(["configure-local"]) == 0
    monkeypatch.setenv("WORKER_RELEASE_ENABLED", "false")
    for operation in ("stage", "prepare", "predeploy", "finish"):
        assert release.main([operation]) == 0
    release.STATE.write_text("malformed ignored journal")
    deploy_vercel_frontend.record_release_deployment(output="")
    assert release.STATE.read_text() == "malformed ignored journal"


def test_no_pending_release_is_a_noop(tmp_path: Path) -> None:
    """No approved pending plan must not block a normal frontend deploy."""
    github, registry, policy, local = inputs()
    vercel = FakeVercel(local | {"image_digest": "old"})
    assert (
        release.prepare(
            github, registry, vercel, policy, local, tmp_path / "state.json"
        )
        is None
    )
    release.finish(github, registry, vercel, local, tmp_path / "state.json")
    assert "latest" not in registry.writes


def test_original_registry_indexes_keep_every_attestation() -> None:
    """The real descriptor-union interface preserves multiple attestation records."""

    class MemoryRegistry(release.Registry):
        """Override only the public transport seam."""

        def manifest(self, reference: str) -> tuple[bytes, dict]:
            """Observe two platform/config pairs and distinct attestations.

            Args:
                reference: Source index or child manifest identity.

            Returns:
                Fake raw bytes and decoded manifest.
            """
            if reference in release.ARCHES:
                return b"", {
                    "manifests": [
                        {
                            "digest": reference + "-child",
                            "platform": {"architecture": reference, "os": "linux"},
                        },
                        {
                            "digest": reference + "-sbom",
                            "platform": {"architecture": "unknown", "os": "unknown"},
                        },
                        {
                            "digest": reference + "-provenance",
                            "platform": {"architecture": "unknown", "os": "unknown"},
                        },
                    ]
                }
            return b"", {"config": {"digest": reference + "-config"}}

    descriptors, identities = MemoryRegistry().identities(
        {a: a for a in release.ARCHES}
    )
    assert len(descriptors) == 6
    assert identities["arm64"]["config"] == "arm64-child-config"


def test_plain_frontend_receipt_path(
    activation_checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The normal deploy helper main records receipts with no per-release export."""
    assert release.main(["configure-local"]) == 0
    release.STATE.write_text(json.dumps({"status": "prepared"}))
    monkeypatch.setattr(
        deploy_vercel_frontend,
        "_run_deploy",
        lambda: (
            "Worker release deployment receipt: https://actual.vercel.app",
            0,
            True,
            True,
        ),
    )
    assert deploy_vercel_frontend.main([]) == 0
    assert (
        json.loads(release.STATE.read_text())["deployment_url"] == "actual.vercel.app"
    )


def test_plain_misconfigured_secret_is_not_retained() -> None:
    """Even an incorrectly plain unrelated token is excluded from recovery values."""
    entry = {
        "id": "secret",
        "key": "HF_CANARY_TOKEN",
        "target": ["production"],
        "type": "plain",
        "value": "never-retain-this-token",
        "updatedAt": 1,
    }
    snapshot = release.public_config([entry])
    assert "never-retain-this-token" not in json.dumps(snapshot)
    assert "value" not in snapshot["secret"]


def test_predeploy_intervening_alias_stops_before_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real CLI predeploy hook refuses a changed alias before deployment."""
    github, registry, policy, local = inputs()
    plan = release.stage(github, registry, policy, local)
    assert plan is not None
    vercel = FakeVercel(plan)
    journal = tmp_path / "state.json"
    release.prepare(github, registry, vercel, policy, local, journal)
    vercel.deployment = "intervening"
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    monkeypatch.setattr(release, "POLICY", policy_path)
    monkeypatch.setattr(release, "STATE", journal)
    monkeypatch.setattr(release, "GitHub", lambda: github)
    monkeypatch.setattr(release, "Registry", lambda: registry)
    monkeypatch.setattr(release, "Vercel", lambda _: vercel)
    monkeypatch.setenv("WORKER_RELEASE_ENABLED", "true")
    monkeypatch.delenv("CI", raising=False)
    assert release.main(["predeploy"]) == 1
    assert "latest" not in registry.writes


def test_prepare_rejects_shared_or_secret_release_variable(tmp_path: Path) -> None:
    """All mutations are preflighted before changing any configuration."""
    github, registry, policy, local = inputs()
    plan = release.stage(github, registry, policy, local)
    assert plan is not None
    vercel = FakeVercel(plan)
    vercel.entries[1]["target"] = ["production", "preview"]
    before = copy.deepcopy(vercel.entries)
    with pytest.raises(ValueError):
        release.prepare(
            github, registry, vercel, policy, local, tmp_path / "state.json"
        )
    assert vercel.entries == before


def test_proof_asset_adapter_uses_fixed_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proof retrieval never follows a supplied download URL or accepts URL IDs."""
    raw = b'{"synthetic":true}'
    paths = []
    commands = []

    def api(self: release.GitHub, path: str, data: dict | None = None) -> dict:
        """Observe a fixed-repository asset.

        Args:
            self: GitHub adapter.
            path: Repository-relative request.
            data: Optional request data.

        Returns:
            Asset metadata with an ignored hostile URL.
        """
        paths.append(path)
        return {
            "id": 42,
            "size": len(raw),
            "browser_download_url": "https://evil.invalid",
        }

    def execute(args: list[str], data: bytes | None = None) -> bytes:
        """Capture fixed-repository byte retrieval.

        Args:
            args: Public tool arguments.
            data: Optional stdin.

        Returns:
            Synthetic proof bytes.
        """
        commands.append(args)
        return raw

    monkeypatch.setattr(release.GitHub, "api", api)
    monkeypatch.setattr(release, "command", execute)
    assert release.GitHub().proof_asset(42) == raw
    assert paths == ["releases/assets/42"]
    assert commands[0][-1] == "repos/EuroEval/EuroEval/releases/assets/42"
    with pytest.raises(ValueError):
        release.GitHub().proof_asset(0)
    assert len(commands) == 1


def test_proof_rechecked_before_config_write(tmp_path: Path) -> None:
    """A retained proof altered after staging fails before any production write."""
    github, registry, policy, local = inputs()
    plan = release.stage(github, registry, policy, local)
    assert plan is not None
    vercel = FakeVercel(plan)
    before = vercel.config()
    github.proofs[1] += b" "
    with pytest.raises(ValueError):
        release.prepare(
            github, registry, vercel, policy, local, tmp_path / "state.json"
        )
    assert vercel.config() == before
    assert not (tmp_path / "state.json").exists()


@pytest.mark.parametrize("mutation", ["missing", "tampered", "oversized", "schema"])
def test_retained_proof_bytes_fail_closed(mutation: str) -> None:
    """Missing, altered, oversized and unexpected proof schemas cannot publish."""
    github, registry, policy, local = inputs()
    record = policy["acceptance"][SHA]["amd64"]
    if mutation == "missing":
        del record["proof_asset_id"]
    elif mutation == "tampered":
        github.proofs[1] += b" "
    else:
        proof = json.loads(github.proofs[1])
        proof["unexpected"] = "x" * (
            release.MAX_PROOF_BYTES if mutation == "oversized" else 1
        )
        github.proofs[1] = json.dumps(proof).encode()
        record["proof_sha256"] = release.digest(github.proofs[1])
    with pytest.raises(ValueError):
        release.stage(github, registry, policy, local)
    assert not registry.writes


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_sha", "b" * 40),
        ("architecture", "arm64"),
        ("platform", {"os": "linux", "architecture": "arm64"}),
        ("index_digest", "sha256:" + "9" * 64),
        ("manifest_digest", "sha256:" + "9" * 64),
        ("config_digest", "sha256:" + "9" * 64),
        ("run_id", 99),
        ("run_attempt", 1),
        ("runtime_fingerprint", "sha256:" + "9" * 64),
        ("guide_sha256", "sha256:" + "9" * 64),
        ("evaluation_completed", False),
        ("exit_code", 1),
        ("guide_exact", False),
        ("clean_docker_gpu_host", False),
        ("test_kind", "diagnostic-matmul"),
        ("inventory", {"python": [], "dpkg": []}),
    ],
)
def test_retained_proof_identity_and_completion_rejected(
    field: str, value: object
) -> None:
    """Even correctly rehashed proofs cannot contradict reviewed build identities."""
    github, registry, policy, local = inputs()
    proof = json.loads(github.proofs[1])
    proof[field] = value
    github.proofs[1] = json.dumps(proof).encode()
    policy["acceptance"][SHA]["amd64"]["proof_sha256"] = release.digest(
        github.proofs[1]
    )
    with pytest.raises(ValueError):
        release.stage(github, registry, policy, local)
    assert not registry.writes
    assert github.plan is None


def test_source_checks_release_control_dirty_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dirty release-control inputs stop staging without treating CSVs as inputs."""

    def observed(args: list[str], data: bytes | None = None) -> bytes:
        """Return only local command observations and inspect dirty-check scope.

        Args:
            args: Requested command arguments.
            data: Unused command input.

        Returns:
            A simulated dirty checkout response.
        """
        del data
        if args[:2] == ["git", "status"]:
            for path in (
                "src/scripts/run_volunteer_worker.py",
                "src/scripts/release_volunteer_worker.py",
                "src/scripts/deploy_vercel_frontend.py",
                "src/scripts/worker-release-policy.json",
                ".github/workflows/worker-image.yaml",
                ".github/workflows/worker-release.yaml",
                "VOLUNTEER_WORKER_OPERATIONS.md",
            ):
                assert path in args
            assert not any("csv" in path or path == ".vercel" for path in args)
            return b" M src/scripts/deploy_vercel_frontend.py"
        return b""

    monkeypatch.setattr(release, "command", observed)
    with pytest.raises(ValueError, match="tracked modifications"):
        release.source()


@pytest.mark.parametrize(
    "override",
    [
        {"event": "pull_request"},
        {"head_repository": {"full_name": "fork/repo"}},
        {"head_branch": "other"},
        {"conclusion": "failure"},
        {"run_attempt": 3},
        {"path": "https://untrusted/workflow"},
    ],
)
def test_untrusted_workflow_refused(override: dict) -> None:
    """Source and attempt provenance cannot be replaced by supplied booleans."""
    github, registry, policy, local = inputs()
    github.run_overrides = override
    with pytest.raises(ValueError):
        release.stage(github, registry, policy, local)
    assert not registry.writes


@pytest.mark.parametrize("project", ["expected", "other"])
def test_vercel_cli_observes_actual_ready_project(
    monkeypatch: pytest.MonkeyPatch, project: str
) -> None:
    """Authenticated CLI observations must bind the production alias to its project.

    Args:
        monkeypatch: Replace only the external authenticated CLI transport.
        project: Observed deployment project.
    """

    def response(args: list[str], data: bytes | None = None) -> bytes:
        """Return the actual Vercel endpoint response contracts.

        Args:
            args: CLI request.
            data: Optional request body.

        Returns:
            JSON response bytes.
        """
        if args[2].startswith("/v4/aliases/"):
            return json.dumps({"deployment": {"id": "dpl-test"}}).encode()
        return json.dumps(
            {
                "id": "dpl-test",
                "projectId": project,
                "readyState": "READY",
                "target": "production",
            }
        ).encode()

    monkeypatch.setattr(release, "command", response)
    adapter = release.Vercel(
        {"alias": "example.com", "team_id": "team", "project_id": "expected"}
    )
    if project == "other":
        with pytest.raises(ValueError):
            adapter.production()
    else:
        assert adapter.production()["id"] == "dpl-test"
