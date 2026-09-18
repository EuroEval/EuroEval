"""Generate a private local contamination-canary corpus."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from transformers import AutoTokenizer

from euroeval.private_canary import (
    MODEL_ID,
    MODEL_REVISION,
    generate_canary_corpus,
    repository_root,
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments and generate the private canary corpus.

    Raises:
        ValueError: If an output directory is inside the repository.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--augmented-dir", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("~/.cache/huggingface/euroeval-canary")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    repository = repository_root()
    paths = (
        args.corpus_jsonl,
        args.key,
        args.augmented_dir,
        args.private_dir,
        args.cache_dir,
    )
    if any(path.expanduser().resolve().is_relative_to(repository) for path in paths):
        raise ValueError(
            "canary inputs, cache and outputs must be outside the repository"
        )
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.chmod(0o700)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache_dir)
    )
    generate_canary_corpus(
        corpus_jsonl=args.corpus_jsonl.expanduser().resolve(),
        key_path=args.key.expanduser().resolve(),
        augmented_dir=args.augmented_dir.expanduser().resolve(),
        private_dir=args.private_dir.expanduser().resolve(),
        tokenizer=tokenizer,
    )
    LOGGER.info("private canary corpus generated")


if __name__ == "__main__":
    main()
