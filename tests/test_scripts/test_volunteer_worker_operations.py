"""Tests for safe volunteer worker operations."""

import json

import pytest

import src.scripts.volunteer_worker_operations as operations


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


def test_smoke_uses_only_safe_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke probes never send credentials or a mutating valid request."""
    requests: list[tuple[str, str, bytes | None]] = []

    def fake_request(
        url: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> operations.HttpResult:
        requests.append((url, method, body))
        if method == "GET":
            return operations.HttpResult(405, b'{"error":"Method not allowed"}')
        if method == "OPTIONS":
            return operations.HttpResult(204, b"")
        return operations.HttpResult(401, b"{}")

    monkeypatch.setattr(operations, "request_http", fake_request)
    diagnostics = operations.smoke(
        base_url="https://euroeval.com",
        routes=["claim"],
        environment={
            "WORKER_COORDINATOR_SECRET": "configured",
            "VOLUNTEER_PROMOTION_SECRET": "configured",
        },
    )

    assert not any(diagnostic.failed for diagnostic in diagnostics)
    assert all(
        body
        in (None, b"{}", b'{"protocol_version":"volunteer-worker/v1","issue_number":1}')
        for _, _, body in requests
    )
    assert not any(
        method == "POST"
        and body
        not in (b"{}", b'{"protocol_version":"volunteer-worker/v1","issue_number":1}')
        for _, method, body in requests
    )
