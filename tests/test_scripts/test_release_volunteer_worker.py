"""Public-interface tests with fake GitHub, OCI and Vercel observations only."""

import copy
import json
from pathlib import Path

import pytest

from scripts import deploy_vercel_frontend, worker_release_inventory
from scripts import release_volunteer_worker as release
from scripts.worker_release_inventory import collect

SHA = "a" * 40


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
        "runtime_fingerprint": "runtime",
    }
    policy = {
        "enabled": True,
        "acceptance": {
            SHA: {
                a: {
                    "index_digest": "sha256:" + ("1" if a == "amd64" else "2") * 64,
                    "accepted": True,
                    "record_id": "physical-log",
                    "reviewer": "maintainer",
                }
                for a in release.ARCHES
            }
        },
    }
    return github, registry, policy, local


class FakeGitHub:
    """In-memory GitHub observations and publication effects."""

    def __init__(self) -> None:
        """Start with two successful trusted workflows and no pending release."""
        self.plan = None
        self.sha = SHA
        self.run_overrides = {}
        self.bad_artifact = False
        self.dispatched = []

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

    def dispatch(self, plan: dict, deployment: str) -> None:
        """Record a finalization request.

        Args:
            plan: Published plan.
            deployment: Verified deployment id.
        """
        self.dispatched.append((plan["tag"], deployment))

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


def test_baseline_requires_actual_inventory_equality() -> None:
    """Mutable apt changes invalidate dataset-configuration-only reuse."""
    github, registry, policy, local = inputs()
    accepted = policy.pop("acceptance")[SHA]
    inventories = {
        a: github.artifact({"id": 12}, a)["inventory"] for a in release.ARCHES
    }
    policy["baseline"] = {
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
    release.prepare(github, registry, vercel, policy, local, journal)
    assert "DO-NOT-PERSIST" not in journal.read_text()
    values = {e["key"]: e["value"] for e in vercel.entries}
    assert values["HF_CANARY_TOKEN"] == "DO-NOT-PERSIST"
    assert (
        values["VOLUNTEER_WORKER_IMAGE_DIGEST"]
        == values["VOLUNTEER_WORKER_IMAGE_DIGEST_ARM64"]
        == plan["image_digest"]
    )
    with pytest.raises(ValueError):
        release.finish(github, vercel, journal)
    assert not github.dispatched
    vercel.deployment = "new-ready"
    monkeypatch.setenv("WORKER_RELEASE_ENABLED", "true")
    receipt_policy = tmp_path / "receipt-policy.json"
    receipt_policy.write_text(json.dumps({"enabled": True}))
    deploy_vercel_frontend.record_release_deployment(
        output="Worker release deployment receipt: https://local-release.vercel.app",
        state_path=journal,
        policy_path=receipt_policy,
    )
    release.finish(github, vercel, journal)
    assert github.dispatched
    release.finalize(github, registry, vercel, plan["tag"], "new-ready")
    assert registry.writes[-1] == "latest"
    with pytest.raises(ValueError):
        release.finalize(github, registry, vercel, plan["tag"], "new-ready")


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
            hostname: Immutable deployment URL.

        Returns:
            Local deployment identity.
        """
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


def test_disabled_hooks_never_construct_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default installation is inert and needs no credentials or pending release."""
    monkeypatch.delenv("WORKER_RELEASE_ENABLED", raising=False)
    for name in ("stage", "prepare", "predeploy", "finish", "finalize"):
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


def test_no_benchmark_data_transport() -> None:
    """Release tooling and workflow contain no benchmark archive mechanism."""
    workflow = Path(".github/workflows/worker-release.yaml").read_text()
    assert "vercel build" not in workflow and "make frontend" not in workflow
    assert "upload-artifact" not in workflow
    assert (
        "csv" not in Path("src/scripts/release_volunteer_worker.py").read_text().lower()
    )


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
    release.finish(github, vercel, tmp_path / "state.json")
    assert not github.dispatched


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
    """The real CLI predeploy hook refuses a changed alias and never dispatches."""
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
    assert release.main(["predeploy"]) == 1
    assert not github.dispatched
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
