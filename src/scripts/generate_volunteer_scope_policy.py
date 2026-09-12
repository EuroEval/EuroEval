"""Generate the versioned, exact-language volunteer scope policy."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import typing as t
from pathlib import Path

from packaging.version import Version

from euroeval.dataset_configs import get_all_dataset_configs
from euroeval.enums import ModelType
from leaderboards.evaluation_common import official_dataset_language_pairs

MODEL_TYPES = ("encoder", "generative")


def main(argv: list[str] | None = None) -> int:
    """Generate, check, or preview the generated policy.

    Returns:
        Zero when the requested operation succeeds, otherwise one.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", default=None, help="EuroEval version (default: installed package)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("api/worker/scope-policy.json")
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
    encoded = encode_policy(policy)
    current = args.output.read_bytes() if args.output.is_file() else None
    changed = current != encoded

    if args.check:
        if current is None:
            print(f"Missing generated policy: {args.output}")
            return 1
        if changed:
            print(f"Stale generated policy: {args.output}")
            return 1
        print(f"Generated policy is current: {args.output}")
        return 0
    if args.dry_run:
        state = "would change" if changed else "would not change"
        print(f"{args.output}: {state}")
        return 0
    write_policy(output=args.output, encoded=encoded)
    print(f"Wrote generated policy: {args.output}")
    return 0


def encode_policy(policy: dict[str, object]) -> bytes:
    """Return the byte-stable representation of a policy."""
    return (json.dumps(policy, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_policy(output: Path, encoded: bytes) -> None:
    """Atomically write encoded policy bytes to ``output``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output)


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


def official_pairs() -> set[tuple[str, str]]:
    """Load the same official dataset/language pairs as the queue.

    Returns:
        Official dataset and language pairs.
    """
    return official_dataset_language_pairs()


if __name__ == "__main__":
    main()
