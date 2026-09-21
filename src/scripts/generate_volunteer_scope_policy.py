"""Generate the versioned, exact-language volunteer scope policy."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import tempfile
from pathlib import Path

from packaging.version import Version

from euroeval.data_models import DatasetConfig
from euroeval.dataset_configs import get_all_dataset_configs
from euroeval.enums import GenerativeType, ModelType, ShotMode
from euroeval.shot_modes import effective_shot_mode, result_identity_values
from leaderboards.evaluation_common import official_dataset_language_pairs

MODEL_TYPES = ("encoder", "generative")
DEFAULT_OUTPUT = Path("api/worker/scope-policy.json")
DEFAULT_TS_OUTPUT = Path("api/worker/_lib/scope-policy.generated.ts")


def main(argv: list[str] | None = None) -> int:
    """Generate, check, or preview the generated policy.

    Args:
        argv (optional): Command-line arguments, excluding the program name.
        Defaults to None, which reads from ``sys.argv``.

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
    policy = build_policy(euroeval_version=version, pairs=official_pairs())
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
    """Build a policy from the worker's effective shot-mode plans.

    Args:
        euroeval_version:
            EuroEval version encoded in the policy and result identities.
        pairs:
            Official dataset/language pairs available to volunteers.
        model_types (optional):
            Model capabilities for which to emit entries. Defaults to both worker
            capabilities.

    Returns:
        A JSON-serialisable policy document containing exact alternatives for each
        model type and language.
    """
    version = str(Version(euroeval_version))
    configs = _configs_by_name()
    entries: list[dict[str, object]] = []
    for model_type in model_types:
        datasets_by_language: dict[str, list[tuple[str, DatasetConfig]]] = {}
        task_groups_by_language: dict[str, set[str]] = {}
        for dataset, language in sorted(pairs):
            config = configs[dataset]
            languages = getattr(config, "languages")
            if language not in {item.code for item in languages} or not (
                _allowed_for_model_type(config=config, model_type=model_type)
            ):
                continue
            datasets_by_language.setdefault(language, []).append((dataset, config))
            task_groups_by_language.setdefault(language, set()).add(
                config.task.task_group.value
            )
        for language, datasets in sorted(datasets_by_language.items()):
            alternatives = _identity_alternatives(
                datasets=datasets, model_type=model_type
            )
            if not alternatives:
                continue
            entries.append(
                {
                    "euroeval_version": version,
                    "model_type": model_type,
                    "language": language,
                    "language_group": language,
                    "allowed_identity_suffix_sets": alternatives,
                    "task_groups": sorted(task_groups_by_language[language]),
                    "warnings": [],
                }
            )
    return {"policy_version": f"volunteer-scope/{version}", "policies": entries}


def _allowed_for_model_type(config: object, model_type: str) -> bool:
    """Return whether one broad model type may run a dataset.

    Args:
        config:
            Dataset configuration to inspect.
        model_type:
            Broad worker model capability.

    Returns:
        Whether the dataset permits the model capability.
    """
    allowed = getattr(config, "allowed_model_types")
    if model_type == "encoder":
        return ModelType.ENCODER in allowed
    if model_type == "generative":
        return ModelType.GENERATIVE in allowed
    return False


def _configs_by_name() -> dict[str, DatasetConfig]:
    """Load dataset configs for model-type-aware scope filtering.

    Returns:
        Dataset configurations keyed by their public name.
    """
    return get_all_dataset_configs(
        custom_datasets_file=Path(""),
        dataset_ids=[],
        api_key=None,
        cache_dir=Path(".cache"),
        trust_remote_code=False,
        run_with_cli=False,
    )


def _identity_alternatives(
    datasets: list[tuple[str, DatasetConfig]], model_type: str
) -> list[list[str]]:
    """Build complete identity alternatives for one model-type/language pair.

    Args:
        datasets:
            Dataset configurations for one language, in deterministic order.
        model_type:
            Broad worker model capability.

    Returns:
        Complete, de-duplicated identity suffix alternatives in runtime order.
    """
    plans = (
        ((None, (ShotMode.FEW_SHOT,)),)
        if model_type == "encoder"
        else (
            (GenerativeType.BASE, (ShotMode.FEW_SHOT,)),
            (GenerativeType.INSTRUCTION_TUNED, (ShotMode.ZERO_SHOT, ShotMode.FEW_SHOT)),
        )
    )
    alternatives: list[list[str]] = []
    seen_alternatives: set[tuple[str, ...]] = set()
    for generative_type, requested_modes in plans:
        suffixes: list[str] = []
        for requested_mode in requested_modes:
            for dataset, config in datasets:
                if generative_type is not None and generative_type not in getattr(
                    config, "allowed_generative_types"
                ):
                    continue
                mode = effective_shot_mode(
                    shot_mode=requested_mode, dataset_config=config
                )
                if mode is None:
                    continue
                few_shot, validation_split = result_identity_values(
                    shot_mode=mode, dataset_config=config, evaluate_test_split=False
                )
                suffix = json.dumps(
                    [dataset, validation_split, few_shot], separators=(",", ":")
                )
                if suffix not in suffixes:
                    suffixes.append(suffix)
        key = tuple(suffixes)
        if suffixes and key not in seen_alternatives:
            seen_alternatives.add(key)
            alternatives.append(suffixes)
    return alternatives


def encode_policy(policy: dict[str, object]) -> bytes:
    """Return the byte-stable representation of a policy.

    Args:
        policy:
            JSON-serialisable policy document.

    Returns:
        UTF-8 encoded, deterministically formatted policy bytes.
    """
    return (json.dumps(policy, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def encode_typescript_policy(policy: dict[str, object]) -> bytes:
    """Return a typed, static TypeScript representation of ``policy``.

    Args:
        policy:
            JSON-serialisable policy document.

    Returns:
        UTF-8 encoded TypeScript source containing the policy.
    """
    document = json.dumps(policy, ensure_ascii=False, indent=2)
    source = f"""/* Generated by src/scripts/generate_volunteer_scope_policy.py. */
export type ScopePolicyEntry = {{
  euroeval_version: string;
  model_type: "encoder" | "generative";
  language: string;
  language_group: string;
  allowed_identity_suffix_sets: string[][];
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
    """Atomically stage and replace all generated policy outputs.

    Args:
        outputs:
            Destination paths and their encoded policy contents.
    """
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
