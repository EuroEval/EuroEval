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

PROFILES = (
    "bert",
    "roberta",
    "eurobert",
    "llama",
    "mistral",
    "qwen",
    "gemma",
    "phi",
    "falcon",
    "gpt2",
)
ENCODER_PROFILES = frozenset({"bert", "roberta", "eurobert"})


def official_pairs() -> set[tuple[str, str]]:
    """Load the same official dataset/language pairs as the queue.

    Returns:
        Official dataset and language pairs.
    """
    return official_dataset_language_pairs()


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


def _allowed_for_profile(config: object, profile: str) -> bool:
    """Return whether one architecture profile may run a dataset."""
    allowed = getattr(config, "allowed_model_types")
    if profile in ENCODER_PROFILES:
        return ModelType.ENCODER in allowed
    return ModelType.GENERATIVE in allowed


def build_policy(
    euroeval_version: str,
    pairs: set[tuple[str, str]],
    profiles: tuple[str, ...] = PROFILES,
) -> dict[str, object]:
    """Build a policy from exact worker identity defaults and dataset contracts.

    Returns:
        A JSON-serialisable policy document.
    """
    euroeval_version = str(Version(euroeval_version))
    configs = _configs_by_name()
    entries: list[dict[str, object]] = []
    for profile in profiles:
        by_language: dict[str, list[str]] = {}
        for dataset, language in sorted(pairs):
            config = configs[dataset]
            languages = getattr(config, "languages")
            if language not in {
                item.code for item in languages
            } or not _allowed_for_profile(config, profile):
                continue
            # Benchmarker defaults are validation_split=False and few_shot=True.
            suffix = json.dumps([dataset, False, True], separators=(",", ":"))
            by_language.setdefault(language, []).append(suffix)
        entries.extend(
            {
                "euroeval_version": euroeval_version,
                "model_profile": profile,
                "language": language,
                "language_group": language,
                "identity_suffixes": suffixes,
                "count": len(suffixes),
                "warnings": [],
            }
            for language, suffixes in sorted(by_language.items())
        )
    return {
        "policy_version": f"volunteer-scope/{euroeval_version}",
        "policies": entries,
    }


def main() -> None:
    """Write the generated policy atomically."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", default=None, help="EuroEval version (default: installed package)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("api/worker/scope-policy.json")
    )
    args = parser.parse_args()
    version = args.version or importlib.metadata.version("euroeval")
    policy = build_policy(version, official_pairs())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    encoded = json.dumps(policy, ensure_ascii=False, indent=2)
    temporary.write_text(encoded + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
