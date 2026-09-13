"""Tests for Vercel function output verification."""

import json
from pathlib import Path

import pytest

import src.scripts.verify_vercel_functions as module


def test_commonjs_node_package_metadata_fails(tmp_path: Path) -> None:
    """CommonJS package metadata cannot support the emitted ESM bundle."""
    _write_output(tmp_path)
    package = tmp_path / "api/worker/result.func/package.json"
    package.write_text('{"type": "commonjs"}', encoding="utf-8")

    with pytest.raises(
        module.VerificationError, match="Non-module Node package metadata"
    ):
        module.verify_vercel_functions(functions_directory=tmp_path)


def _write_output(root: Path, runtimes: dict[str, str] | None = None) -> None:
    """Write a complete Build Output API functions tree."""
    runtimes = runtimes or module.EXPECTED_RUNTIMES
    for route, runtime in runtimes.items():
        function_directory = root / f"{route}.func"
        config = function_directory / ".vc-config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps({"runtime": runtime}), encoding="utf-8")
        if runtime.startswith("nodejs"):
            entrypoint = function_directory.joinpath(*route.split("/")).with_suffix(
                ".js"
            )
            entrypoint.parent.mkdir(parents=True, exist_ok=True)
            entrypoint.write_text("export {}\n", encoding="utf-8")
            (function_directory / "package.json").write_text(
                json.dumps({"type": "module"}), encoding="utf-8"
            )


def test_concrete_node_runtime_passes(tmp_path: Path) -> None:
    """Vercel's concrete Node runtime is accepted."""
    runtimes = dict(module.EXPECTED_RUNTIMES)
    runtimes["api/worker/finalise"] = "nodejs24.x"
    runtimes["api/worker/result"] = "nodejs24.x"
    _write_output(tmp_path, runtimes)

    module.verify_vercel_functions(functions_directory=tmp_path)


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


@pytest.mark.parametrize("package_contents", ["not json", '{"type": 42}', "[]"])
def test_malformed_node_package_metadata_fails(
    tmp_path: Path, package_contents: str
) -> None:
    """Malformed Node package metadata aborts verification."""
    _write_output(tmp_path)
    package = tmp_path / "api/worker/result.func/package.json"
    package.write_text(package_contents, encoding="utf-8")

    with pytest.raises(
        module.VerificationError, match="Malformed Node package metadata"
    ):
        module.verify_vercel_functions(functions_directory=tmp_path)


def test_missing_config_fails(tmp_path: Path) -> None:
    """A function directory without metadata is rejected."""
    _write_output(tmp_path)
    (tmp_path / "api/worker/result.func/.vc-config.json").unlink()

    with pytest.raises(module.VerificationError, match="Missing function config"):
        module.verify_vercel_functions(functions_directory=tmp_path)


def test_missing_node_package_metadata_fails(tmp_path: Path) -> None:
    """A Node bundle without package metadata is rejected."""
    _write_output(tmp_path)
    (tmp_path / "api/worker/result.func/package.json").unlink()

    with pytest.raises(module.VerificationError, match="Missing Node package metadata"):
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


def test_source_package_metadata_does_not_satisfy_bundle(tmp_path: Path) -> None:
    """Source-tree metadata cannot mask missing bundle metadata."""
    functions_directory = tmp_path / "functions"
    (tmp_path / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    _write_output(functions_directory)
    (functions_directory / "api/worker/result.func/package.json").unlink()

    with pytest.raises(module.VerificationError, match="Missing Node package metadata"):
        module.verify_vercel_functions(functions_directory=functions_directory)


def test_wrong_runtime_fails(tmp_path: Path) -> None:
    """A route emitted for the wrong runtime aborts verification."""
    runtimes = dict(module.EXPECTED_RUNTIMES)
    runtimes["api/worker/result"] = "edge"
    _write_output(tmp_path, runtimes)

    with pytest.raises(module.VerificationError, match="runtime mismatch"):
        module.verify_vercel_functions(functions_directory=tmp_path)
