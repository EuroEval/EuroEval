"""Tests for safe volunteer worker operations."""

import json
import subprocess
import typing as t
from pathlib import Path

import pytest

import src.scripts.volunteer_worker_operations as operations


def test_apply_github_creates_only_missing_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GitHub setup is idempotent and does not recreate existing labels."""
    commands: list[list[str]] = []

    def fake_command(command: list[str], **kwargs: object) -> operations.CommandResult:
        commands.append(command)
        if command[:3] == ["gh", "api", f"repos/{operations.REPOSITORY}/labels"]:
            return operations.CommandResult(
                0, json.dumps([{"name": operations.LABELS[0]}])
            )
        return operations.CommandResult(0)

    monkeypatch.setattr(operations, "run_command", fake_command)
    diagnostics = operations.apply_github()

    assert all(not diagnostic.failed for diagnostic in diagnostics)
    created = [
        command for command in commands if command[:3] == ["gh", "label", "create"]
    ]
    assert len(created) == 2
    assert operations.LABELS[0] not in {" ".join(command) for command in created}


def test_apply_hf_creates_eu_only_after_absence_is_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HF apply creates only an explicitly absent bucket."""
    commands: list[list[str]] = []
    info_calls = 0

    def fake_command(command: list[str], **kwargs: object) -> operations.CommandResult:
        nonlocal info_calls
        commands.append(command)
        if command[3:5] == ["buckets", "info"]:
            info_calls += 1
            if info_calls == 1:
                return operations.CommandResult(1, stderr="404 not found")
            return operations.CommandResult(0, '{"private":true}')
        return operations.CommandResult(0)

    monkeypatch.setattr(operations, "run_command", fake_command)
    diagnostics = operations.apply_hf(
        environment={"HF_TOKEN": "token", "HF_STAGING_BUCKET": "bucket"}
    )
    assert not any(item.failed for item in diagnostics)
    create = next(command for command in commands if "create" in command)
    assert create[-2:] == ["--region", "eu"]


def test_apply_requires_explicit_component_and_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default apply path cannot mutate anything."""
    called = False

    def fail_command(command: list[str], **kwargs: object) -> operations.CommandResult:
        nonlocal called
        called = True
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(operations, "run_command", fail_command)
    assert operations.main(["apply"]) == 2
    assert operations.main(["apply", "--github"]) == 2
    assert not called


def test_apply_vercel_adds_atomically_and_validates_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Vercel apply never removes variables and sends values only on stdin."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".vercel").mkdir()
    (tmp_path / ".vercel/project.json").write_text(
        '{"projectId":"project-id","orgId":"team-id","projectName":"euroeval"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(operations, "source_versions", lambda: ("1.0", "1.0"))
    values = {name: f"value-{name}" for name in operations.REQUIRED_ENVIRONMENT}
    values.update(
        {
            "VERCEL_PROJECT_ID": "project-id",
            "VERCEL_ORG_ID": "team-id",
            "VERCEL_PROJECT_NAME": "euroeval",
        }
    )
    commands: list[list[str]] = []

    def fake_command(command: list[str], **kwargs: object) -> operations.CommandResult:
        commands.append(command)
        if command[:3] == ["vercel", "project", "inspect"]:
            return operations.CommandResult(0, '{"id":"project-id","name":"euroeval"}')
        if command[:3] == ["vercel", "env", "ls"]:
            metadata = [
                {
                    "key": name,
                    "target": ["production"],
                    "type": "plain"
                    if name in operations.PUBLIC_CONFIG
                    else "sensitive",
                }
                for name in operations.REQUIRED_ENVIRONMENT
            ]
            return operations.CommandResult(0, json.dumps(metadata))
        assert "env" in command and "add" in command
        assert kwargs["input_text"] == values[command[3]] + "\n"
        assert "--force" in command and "--yes" in command and "--type" in command
        return operations.CommandResult(0)

    monkeypatch.setattr(operations, "run_command", fake_command)
    diagnostics = operations.apply_vercel(environment=values)
    assert not any(item.failed for item in diagnostics)
    assert not any(command[2] == "rm" for command in commands if len(command) > 2)
    assert all(
        value not in command for command in commands for value in values.values()
    )


def test_docker_inspection_preserves_safe_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anonymous Docker inspection keeps PATH but cannot use host credentials."""
    digest = "sha256:" + "a" * 64
    captured: dict[str, str] = {}
    for name in operations.BASIC_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", "/custom/bin")
    monkeypatch.setenv("XDG_DATA_HOME", "/tmp/data")
    monkeypatch.setenv("GH_TOKEN", "host-secret")
    monkeypatch.setenv("DOCKER_CONFIG", "/host/config")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = args[0]
        assert isinstance(command, list)
        child_environment = kwargs["env"]
        assert isinstance(child_environment, dict)
        if command[0] == "docker":
            captured.update(child_environment)
            config = Path(child_environment["DOCKER_CONFIG"])
            assert config.is_dir()
            assert not list(config.iterdir())
            output = json.dumps(
                {
                    "digest": digest,
                    "manifests": [
                        {
                            "descriptor": {"digest": "sha256:" + "b" * 64},
                            "platform": {"os": "linux", "architecture": "amd64"},
                        }
                    ],
                }
            )
            return subprocess.CompletedProcess(command, 0, output, "")
        return subprocess.CompletedProcess(command, 0, "public", "")

    monkeypatch.setattr(operations.subprocess, "run", fake_run)
    diagnostics = operations.check_ghcr(
        environment={"VOLUNTEER_WORKER_IMAGE_DIGEST": digest}
    )

    assert not any(item.failed for item in diagnostics)
    assert captured["PATH"] == "/custom/bin"
    assert captured["XDG_DATA_HOME"] == "/tmp/data"
    assert "GH_TOKEN" not in captured
    assert captured["DOCKER_CONFIG"] != "/host/config"


def test_generator_check_exit_and_no_write_modes(tmp_path: Path) -> None:
    """Generator status propagates through a subprocess and checks do not write."""
    output = tmp_path / "scope-policy.json"
    command = [
        "uv",
        "run",
        "python",
        "src/scripts/generate_volunteer_scope_policy.py",
        "--check",
        "--output",
        str(output),
    ]
    missing = subprocess.run(command, capture_output=True, text=True, check=False)
    assert missing.returncode != 0
    output.write_text("stale", encoding="utf-8")
    before = output.read_bytes()
    stale = subprocess.run(command, capture_output=True, text=True, check=False)
    assert stale.returncode != 0
    assert output.read_bytes() == before
    dry_run = subprocess.run(
        [item for item in command if item != "--check"] + ["--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry_run.returncode == 0
    assert output.read_bytes() == before


def test_ghcr_manifest_check_is_anonymous_and_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GHCR checks use an empty Docker config and never pull or log out."""
    digest = "sha256:" + "a" * 64
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def fake_command(command: list[str], **kwargs: object) -> operations.CommandResult:
        commands.append(command)
        if command[0] == "gh":
            return operations.CommandResult(0, "public")
        command_environment = kwargs.get("environment")
        assert isinstance(command_environment, dict)
        environments.append(t.cast(dict[str, str], command_environment))
        return operations.CommandResult(
            0,
            json.dumps(
                {
                    "digest": digest,
                    "manifests": [
                        {
                            "descriptor": {"digest": "sha256:" + "b" * 64},
                            "platform": {"os": "linux", "architecture": "amd64"},
                        },
                        {
                            "descriptor": {"digest": "sha256:" + "c" * 64},
                            "platform": {"os": "unknown", "architecture": "unknown"},
                        },
                    ],
                }
            ),
        )

    monkeypatch.setattr(operations, "run_command", fake_command)
    diagnostics = operations.check_ghcr(
        environment={"VOLUNTEER_WORKER_IMAGE_DIGEST": digest, "GH_TOKEN": "secret"}
    )
    assert not any(item.failed for item in diagnostics)
    assert not any(command[1] in {"pull", "logout"} for command in commands)
    assert environments == [{"DOCKER_CONFIG": environments[0]["DOCKER_CONFIG"]}]


def test_hf_uses_auth_whoami_and_only_creates_absent_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing HF buckets are inspected, not recreated or region-claimed."""
    commands: list[list[str]] = []

    def fake_command(command: list[str], **kwargs: object) -> operations.CommandResult:
        commands.append(command)
        if command[3:5] == ["buckets", "info"]:
            return operations.CommandResult(0, '{"private":true}')
        if command[3:6] == ["hf", "auth", "whoami"]:
            return operations.CommandResult(0)
        return operations.CommandResult(0, '{"private":true}')

    monkeypatch.setattr(operations, "run_command", fake_command)
    diagnostics = operations.check_hf(
        environment={"HF_TOKEN": "token", "HF_STAGING_BUCKET": "bucket"}
    )
    assert commands[0] == ["uv", "run", "hf", "auth", "whoami"]
    assert any("manual/unverifiable" in item.message for item in diagnostics)
    assert not any("create" in command for command in commands)


def test_manifest_accepts_attested_index_with_independent_proofs() -> None:
    """An index digest and its amd64 child are separate attestations."""
    digest = "sha256:" + "a" * 64
    output = {
        "digest": digest,
        "manifests": [
            {
                "descriptor": {"digest": "sha256:" + "b" * 64},
                "platform": {"os": "linux", "architecture": "amd64"},
            },
            {
                "descriptor": {"digest": "sha256:" + "c" * 64},
                "platform": {"os": "unknown", "architecture": "unknown"},
            },
        ],
    }

    assert operations._manifest_matches(json.dumps(output), digest)


def test_manifest_rejects_digest_mismatch_with_amd64_child() -> None:
    """An amd64 child cannot make a mismatched index digest valid."""
    digest = "sha256:" + "a" * 64
    output = {
        "digest": "sha256:" + "d" * 64,
        "manifests": [
            {
                "descriptor": {"digest": "sha256:" + "b" * 64},
                "platform": {"os": "linux", "architecture": "amd64"},
            }
        ],
    }

    assert not operations._manifest_matches(json.dumps(output), digest)


def test_manifest_rejects_index_without_amd64_child() -> None:
    """A matching index digest is insufficient without an amd64 child."""
    digest = "sha256:" + "a" * 64
    output = {
        "digest": digest,
        "manifests": [
            {
                "descriptor": {"digest": "sha256:" + "b" * 64},
                "platform": {"os": "linux", "architecture": "arm64"},
            }
        ],
    }

    assert not operations._manifest_matches(json.dumps(output), digest)


def test_plan_does_not_run_commands_or_print_secret_values(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Plan is local-only and only names environment variables."""
    secret = "do-not-print-this"
    called = False

    def fail_command(command: list[str], **kwargs: object) -> operations.CommandResult:
        nonlocal called
        called = True
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(operations, "run_command", fail_command)
    operations.print_plan(environment={"GITHUB_TOKEN": secret})

    output = capsys.readouterr().out
    assert not called
    assert "GITHUB_TOKEN (present)" in output
    assert secret not in output
    assert "EuroEval/EuroEval" in output


def test_redis_requires_exact_pong_and_never_prints_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Malformed responses fail without leaking the URL."""
    url = "https://secret.example.invalid/rest"
    monkeypatch.setattr(
        operations,
        "request_http",
        lambda *args, **kwargs: operations.HttpResult(200, b'{"result":"NOPE"}'),
    )
    diagnostics = operations.check_redis(
        environment={
            "UPSTASH_REDIS_REST_URL": url,
            "UPSTASH_REDIS_REST_TOKEN": "secret",
        }
    )
    operations.print_diagnostics(diagnostics)
    assert diagnostics[0].failed
    assert url not in capsys.readouterr().out


def test_run_command_isolates_tool_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrelated broker secrets never reach child processes."""
    captured: dict[str, str] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.update(kwargs["env"])
        command = args[0]
        assert isinstance(command, list)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(operations.subprocess, "run", fake_run)
    environment_names = operations.BASIC_ENVIRONMENT | set(
        operations.TOOL_AUTH_ENVIRONMENT["gh"]
    )
    for name in environment_names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", "/bin")
    monkeypatch.setenv("HOME", "/tmp")
    monkeypatch.setenv("XDG_DATA_HOME", "/tmp/data")
    operations.run_command(
        ["gh", "auth", "status"],
        environment={
            "PATH": "/bin",
            "HOME": "/tmp",
            "XDG_DATA_HOME": "/tmp/data",
            "GH_TOKEN": "gh-secret",
            "UPSTASH_REDIS_REST_TOKEN": "redis-secret",
            "WORKER_COORDINATOR_SECRET": "coordinator-secret",
        },
    )

    assert captured == {
        "PATH": "/bin",
        "HOME": "/tmp",
        "XDG_DATA_HOME": "/tmp/data",
        "GH_TOKEN": "gh-secret",
    }


def test_smoke_accepts_protected_401_or_deployed_missing_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protected smoke probes do not depend on local secret presence."""

    def fake_request(
        url: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> operations.HttpResult:
        if method == "GET":
            return operations.HttpResult(405, b'{"error":"Method not allowed"}')
        if method == "OPTIONS":
            return operations.HttpResult(204, b"")
        if url.endswith("/claim"):
            return operations.HttpResult(401, b'{"error":"authentication failed"}')
        return operations.HttpResult(503, b'{"error":"missing configuration"}')

    monkeypatch.setattr(operations, "request_http", fake_request)
    diagnostics = operations.smoke(
        base_url="https://euroeval.com", routes=["claim"], environment={}
    )

    assert not any(diagnostic.failed for diagnostic in diagnostics)


@pytest.mark.parametrize(
    "environment_name", ["VERCEL_PROJECT_ID", "VERCEL_ORG_ID", "VERCEL_PROJECT_NAME"]
)
def test_vercel_identity_rejects_environment_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, environment_name: str
) -> None:
    """Configured Vercel identity values must match the local link."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".vercel").mkdir()
    (tmp_path / ".vercel/project.json").write_text(
        '{"projectId":"project-id","orgId":"team-id","projectName":"euroeval"}',
        encoding="utf-8",
    )
    called = False

    def fail_command(command: list[str], **kwargs: object) -> operations.CommandResult:
        nonlocal called
        called = True
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(operations, "run_command", fail_command)
    diagnostics, verified = operations._check_vercel_project(
        environment={environment_name: "different"}
    )

    assert not verified
    assert diagnostics[0].failed
    assert environment_name in diagnostics[0].message
    assert not called


def test_vercel_identity_requires_project_name_in_local_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A project link without its name cannot authorise Vercel changes."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".vercel").mkdir()
    (tmp_path / ".vercel/project.json").write_text(
        '{"projectId":"project-id","orgId":"team-id"}', encoding="utf-8"
    )

    diagnostics, verified = operations._check_vercel_project(environment={})

    assert not verified
    assert diagnostics[0].failed
    assert "projectName" in diagnostics[0].message
