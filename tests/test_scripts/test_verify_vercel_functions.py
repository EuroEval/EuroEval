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
            entrypoint.write_text(
                "export async function fetch(_request) { "
                'return new Response("ok"); }\n',
                encoding="utf-8",
            )
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


def test_node_entrypoint_accepts_callable_fetch_alias(tmp_path: Path) -> None:
    """A callable aliased to fetch is a valid module export."""
    _write_output(tmp_path)
    _write_node_entrypoint(
        tmp_path,
        "api/worker/result",
        'function handler() { return new Response("ok"); }\n'
        "export { handler as fetch };\n",
    )

    module.verify_vercel_functions(functions_directory=tmp_path)


def _write_node_entrypoint(root: Path, route: str, source: str) -> None:
    """Replace one generated Node entrypoint with source for a focused test."""
    entrypoint = ((root / f"{route}.func").joinpath(*route.split("/"))).with_suffix(
        ".js"
    )
    entrypoint.write_text(source, encoding="utf-8")


def test_node_entrypoint_accepts_callable_fetch_function(tmp_path: Path) -> None:
    """A valid callable named fetch passes without invoking the handler."""
    _write_output(tmp_path)
    _write_node_entrypoint(
        tmp_path,
        "api/worker/result",
        'export function fetch() { throw new Error("handler invoked"); }\n',
    )

    module.verify_vercel_functions(functions_directory=tmp_path)


@pytest.mark.parametrize("route", ["api/worker/finalise", "api/worker/result"])
def test_node_entrypoint_defaults_are_rejected(tmp_path: Path, route: str) -> None:
    """Node endpoints must not default-export handler functions."""
    _write_output(tmp_path)
    entrypoint = ((tmp_path / f"{route}.func").joinpath(*route.split("/"))).with_suffix(
        ".js"
    )
    entrypoint.write_text(
        "export default function handler() { return new Response() }\n",
        encoding="utf-8",
    )

    with pytest.raises(module.VerificationError, match="must not export default"):
        module.verify_vercel_functions(functions_directory=tmp_path)


def test_node_entrypoint_does_not_inherit_parent_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Import validation must not expose parent-process secrets to bundles."""
    _write_output(tmp_path)
    monkeypatch.setenv("EUROEVAL_TEST_SECRET", "not-for-the-bundle")
    _write_node_entrypoint(
        tmp_path,
        "api/worker/result",
        'if (process.env.EUROEVAL_TEST_SECRET) throw new Error("secret leaked");\n'
        'export function fetch() { return new Response("ok"); }\n',
    )

    module.verify_vercel_functions(functions_directory=tmp_path)


@pytest.mark.parametrize("route", ["api/worker/finalise", "api/worker/result"])
def test_node_entrypoint_exports_require_named_fetch(
    tmp_path: Path, route: str
) -> None:
    """Node endpoints must export a callable named fetch entrypoint."""
    _write_output(tmp_path)
    entrypoint = ((tmp_path / f"{route}.func").joinpath(*route.split("/"))).with_suffix(
        ".js"
    )
    entrypoint.write_text("export {}\n", encoding="utf-8")

    with pytest.raises(module.VerificationError, match="must export a callable fetch"):
        module.verify_vercel_functions(functions_directory=tmp_path)


def test_node_entrypoint_ignores_fake_exports_in_comments_and_strings(
    tmp_path: Path,
) -> None:
    """Comments and strings must not affect the semantic export check."""
    _write_output(tmp_path)
    _write_node_entrypoint(
        tmp_path,
        "api/worker/result",
        '// export default handler\nconst text = "export function fetch() {}";\n'
        'export function fetch() { return new Response("ok"); }\n',
    )

    module.verify_vercel_functions(functions_directory=tmp_path)


def test_node_entrypoint_import_timeout_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module that never completes import must not hang verification."""
    _write_output(tmp_path)
    _write_node_entrypoint(
        tmp_path,
        "api/worker/result",
        "await new Promise((resolve) => setTimeout(resolve, 60000));\n",
    )
    monkeypatch.setattr(module, "_NODE_IMPORT_TIMEOUT_SECONDS", 0.1)

    with pytest.raises(module.VerificationError, match="timed out"):
        module.verify_vercel_functions(functions_directory=tmp_path)


def test_node_entrypoint_rejects_default_alongside_fetch(tmp_path: Path) -> None:
    """A default export is invalid even when fetch is also callable."""
    _write_output(tmp_path)
    _write_node_entrypoint(
        tmp_path,
        "api/worker/result",
        'export function fetch() { return new Response("ok"); }\n'
        "export default fetch;\n",
    )

    with pytest.raises(module.VerificationError, match="must not export default"):
        module.verify_vercel_functions(functions_directory=tmp_path)


def test_node_entrypoint_rejects_non_callable_fetch(tmp_path: Path) -> None:
    """A named fetch export must contain a function value."""
    _write_output(tmp_path)
    _write_node_entrypoint(tmp_path, "api/worker/result", "export const fetch = 42;\n")

    with pytest.raises(module.VerificationError, match="callable fetch"):
        module.verify_vercel_functions(functions_directory=tmp_path)


@pytest.mark.parametrize(
    "source",
    [
        "export function fetch( {\n",
        'import "./missing-dependency.js";\n'
        'export function fetch() { return new Response("ok"); }\n',
    ],
)
def test_node_entrypoint_rejects_syntax_or_import_failure(
    tmp_path: Path, source: str
) -> None:
    """Syntax and dependency failures must fail import validation."""
    _write_output(tmp_path)
    _write_node_entrypoint(tmp_path, "api/worker/result", source)

    with pytest.raises(
        module.VerificationError, match="Could not import Node entrypoint"
    ):
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
