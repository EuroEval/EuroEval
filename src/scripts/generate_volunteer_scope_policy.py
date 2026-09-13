"""Generate the versioned, exact-language volunteer scope policy."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import tempfile
import typing as t
from pathlib import Path

from packaging.version import Version

from euroeval.dataset_configs import get_all_dataset_configs
from euroeval.enums import ModelType
from leaderboards.evaluation_common import official_dataset_language_pairs

MODEL_TYPES = ("encoder", "generative")
DEFAULT_OUTPUT = Path("api/worker/scope-policy.json")
DEFAULT_TS_OUTPUT = Path("api/worker/_lib/scope-policy.generated.ts")


def main(argv: list[str] | None = None) -> int:
    """Generate, check, or preview the generated policy.

    Returns:
        Zero when the requested operation succeeds, otherwise one.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", default=None, help="EuroEval version (default: installed package)"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ts-output",
        type=Path,
        default=None,
        help=(
            "TypeScript mirror output (defaults to "
            f"{DEFAULT_TS_OUTPUT}; custom JSON outputs use a sibling mirror)"
        ),
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--check", action="store_true", help="Check without writing the policy"
    )
    modes.add_argument(
        "--dry-run", action="store_true", help="Report changes without writing"
    )
    args = parser.parse_args(argv)
    version = args.version or importlib.metadata.version("euroeval")
    policy = build_policy(version, official_pairs())
    ts_output = args.ts_output or (
        DEFAULT_TS_OUTPUT
        if args.output == DEFAULT_OUTPUT
        else args.output.with_name("scope-policy.generated.ts")
    )
    outputs = [(args.output, encode_policy(policy))]
    if ts_output is not None:
        outputs.append((ts_output, encode_typescript_policy(policy)))
    states = [
        (output, output.read_bytes() if output.is_file() else None, encoded)
        for output, encoded in outputs
    ]
    changed = any(current != encoded for _, current, encoded in states)

    if args.check:
        for output, current, encoded in states:
            if current is None:
                print(f"Missing generated policy: {output}")
            elif current != encoded:
                print(f"Stale generated policy: {output}")
        if changed:
            return 1
        print("Generated scope policies are current.")
        return 0
    if args.dry_run:
        state = "would change" if changed else "would not change"
        for output, _, _ in states:
            print(f"{output}: {state}")
        return 0
    write_policies(outputs)
    for output, _, _ in states:
        print(f"Wrote generated policy: {output}")
    return 0


def build_policy(
    euroeval_version: str,
    pairs: set[tuple[str, str]],
    model_types: tuple[str, ...] = MODEL_TYPES,
) -> dict[str, object]:
    """Build a policy from exact worker identity defaults and dataset contracts.

    Returns:
        A JSON-serialisable policy document.
    """
    euroeval_version = str(Version(euroeval_version))
    configs = _configs_by_name()
    entries: list[dict[str, object]] = []
    for model_type in model_types:
        by_language: dict[str, list[str]] = {}
        task_groups_by_language: dict[str, set[str]] = {}
        for dataset, language in sorted(pairs):
            config = configs[dataset]
            languages = getattr(config, "languages")
            if language not in {
                item.code for item in languages
            } or not _allowed_for_model_type(config, model_type):
                continue
            # Benchmarker defaults are validation_split=False and few_shot=True.
            suffix = json.dumps([dataset, False, True], separators=(",", ":"))
            by_language.setdefault(language, []).append(suffix)
            task_groups_by_language.setdefault(language, set()).add(
                config.task.task_group.value
            )
        entries.extend(
            {
                "euroeval_version": euroeval_version,
                "model_type": model_type,
                "language": language,
                "language_group": language,
                "identity_suffixes": suffixes,
                "count": len(suffixes),
                "task_groups": sorted(task_groups_by_language[language]),
                "warnings": [],
            }
            for language, suffixes in sorted(by_language.items())
        )
    return {
        "policy_version": f"volunteer-scope/{euroeval_version}",
        "policies": entries,
    }


def _allowed_for_model_type(config: object, model_type: str) -> bool:
    """Return whether one broad model type may run a dataset."""
    allowed = getattr(config, "allowed_model_types")
    if model_type == "encoder":
        return ModelType.ENCODER in allowed
    if model_type == "generative":
        return ModelType.GENERATIVE in allowed
    return False


def _configs_by_name() -> dict[str, object]:
    """Load dataset configs for model-type-aware scope filtering.

    Returns:
        Dataset configurations keyed by their public name.
    """
    return t.cast(
        dict[str, object],
        get_all_dataset_configs(
            custom_datasets_file=Path(""),
            dataset_ids=[],
            api_key=None,
            cache_dir=Path(".cache"),
            trust_remote_code=False,
            run_with_cli=False,
        ),
    )


def encode_policy(policy: dict[str, object]) -> bytes:
    """Return the byte-stable representation of a policy."""
    return (json.dumps(policy, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def encode_typescript_policy(policy: dict[str, object]) -> bytes:
    """Return a typed, static TypeScript representation of ``policy``."""
    document = json.dumps(policy, ensure_ascii=False, indent=2)
    source = f"""/* Generated by src/scripts/generate_volunteer_scope_policy.py. */
export type ScopePolicyEntry = {{
  euroeval_version: string;
  model_type: "encoder" | "generative";
  language: string;
  language_group: string;
  identity_suffixes: string[];
  count?: number;
  task_groups: string[];
  warnings?: string[];
}};

export type ScopePolicy = {{
  policy_version: string;
  policies: ScopePolicyEntry[];
}};

const generatedScopePolicy = {document} as const satisfies ScopePolicy;

export default generatedScopePolicy;
"""
    return source.encode("utf-8")


def official_pairs() -> set[tuple[str, str]]:
    """Load the same official dataset/language pairs as the queue.

    Returns:
        Official dataset and language pairs.
    """
    return official_dataset_language_pairs()


def write_policies(outputs: list[tuple[Path, bytes]]) -> None:
    """Atomically stage and replace all generated policy outputs."""
    temporary_paths: list[Path] = []
    try:
        for output, encoded in outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(encoded)
                temporary.flush()
                temporary_paths.append(Path(temporary.name))
        for (output, _), temporary in zip(outputs, temporary_paths, strict=True):
            temporary.replace(output)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
