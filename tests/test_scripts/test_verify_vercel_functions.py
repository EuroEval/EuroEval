"""Tests for Vercel function output verification."""

import json
from pathlib import Path

import pytest

import src.scripts.verify_vercel_functions as module


def test_concrete_node_runtime_passes(tmp_path: Path) -> None:
    """Vercel's concrete Node runtime is accepted."""
    runtimes = dict(module.EXPECTED_RUNTIMES)
    runtimes["api/worker/finalise"] = "nodejs24.x"
    runtimes["api/worker/result"] = "nodejs24.x"
    _write_output(tmp_path, runtimes)

    module.verify_vercel_functions(functions_directory=tmp_path)


def _write_output(root: Path, runtimes: dict[str, str] | None = None) -> None:
    """Write a complete Build Output API functions tree."""
    runtimes = runtimes or module.EXPECTED_RUNTIMES
    for route, runtime in runtimes.items():
        config = root / f"{route}.func" / ".vc-config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps({"runtime": runtime}), encoding="utf-8")


def test_exact_allow_list_passes(tmp_path: Path) -> None:
    """The expected route and runtime set is accepted."""
    _write_output(tmp_path)

    module.verify_vercel_functions(functions_directory=tmp_path)


@pytest.mark.parametrize(
    "route", ["api/worker/_lib", "api/worker/scope-policy.generated"]
)
def test_helper_or_generated_route_fails(tmp_path: Path, route: str) -> None:
    """Private helpers and generated policy files cannot become functions."""
    _write_output(tmp_path)
    config = tmp_path / f"{route}.func" / ".vc-config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('{"runtime": "edge"}', encoding="utf-8")

    with pytest.raises(module.VerificationError, match="extra routes"):
        module.verify_vercel_functions(functions_directory=tmp_path)


@pytest.mark.parametrize("config_contents", ["not json", '{"runtime": 42}'])
def test_malformed_config_fails(tmp_path: Path, config_contents: str) -> None:
    """Malformed function metadata aborts verification."""
    _write_output(tmp_path)
    config = tmp_path / "api/worker/result.func/.vc-config.json"
    config.write_text(config_contents, encoding="utf-8")

    with pytest.raises(module.VerificationError, match="Malformed function config"):
        module.verify_vercel_functions(functions_directory=tmp_path)


def test_missing_config_fails(tmp_path: Path) -> None:
    """A function directory without metadata is rejected."""
    _write_output(tmp_path)
    (tmp_path / "api/worker/result.func/.vc-config.json").unlink()

    with pytest.raises(module.VerificationError, match="Missing function config"):
        module.verify_vercel_functions(functions_directory=tmp_path)


def test_missing_route_fails(tmp_path: Path) -> None:
    """A missing expected function aborts verification."""
    runtimes = {
        route: runtime
        for route, runtime in module.EXPECTED_RUNTIMES.items()
        if route != "api/worker/result"
    }
    _write_output(tmp_path, runtimes)

    with pytest.raises(module.VerificationError, match="missing routes"):
        module.verify_vercel_functions(functions_directory=tmp_path)


def test_wrong_runtime_fails(tmp_path: Path) -> None:
    """A route emitted for the wrong runtime aborts verification."""
    runtimes = dict(module.EXPECTED_RUNTIMES)
    runtimes["api/worker/result"] = "edge"
    _write_output(tmp_path, runtimes)

    with pytest.raises(module.VerificationError, match="runtime mismatch"):
        module.verify_vercel_functions(functions_directory=tmp_path)
