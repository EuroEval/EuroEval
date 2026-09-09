"""Generate the versioned, exact-language volunteer scope policy."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

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


def official_pairs() -> set[tuple[str, str]]:
    """Load the same official dataset/language pairs as the queue.

    Returns:
        The official dataset/language pairs.
    """
    return official_dataset_language_pairs()


def build_policy(
    euroeval_version: str,
    pairs: set[tuple[str, str]],
    profiles: tuple[str, ...] = PROFILES,
) -> dict[str, object]:
    """Build a policy whose keys are ISO language codes, never groups.

    Returns:
        A JSON-serialisable policy document.
    """
    by_language: dict[str, list[str]] = {}
    for dataset, language in sorted(pairs):
        # Queue runs use the validation split and the default zero-shot setting.
        suffix = json.dumps([dataset, True, False], separators=(",", ":"))
        by_language.setdefault(language, []).append(suffix)
    entries = [
        {
            "euroeval_version": euroeval_version,
            "model_profile": profile,
            "language": language,
            "language_group": language,
            "identity_suffixes": suffixes,
            "count": len(suffixes),
            "warnings": [],
        }
        for profile in profiles
        for language, suffixes in sorted(by_language.items())
    ]
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
