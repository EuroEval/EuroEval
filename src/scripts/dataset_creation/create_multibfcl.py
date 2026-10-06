"""Prepare test-only multilingual BFCL datasets for EuroEval.

Publication is opt-in: run with ``--publish`` after reviewing the generated data.
"""

import argparse
import json
import logging

from datasets import DatasetDict, load_dataset

logger = logging.getLogger(__name__)

SOURCE_REPO = "syvai/multi-bfcl"
MAX_TEST_SAMPLES = 2048
# One source subset per non-English EuroEval language configuration. The source uses
# pt-pt for Portuguese, while EuroEval identifies the benchmark with pt.
LANGUAGE_SOURCES = {
    "sq": "sq",
    "be": "be",
    "bs": "bs",
    "bg": "bg",
    "ca": "ca",
    "hr": "hr",
    "cs": "cs",
    "da": "da",
    "nl": "nl",
    "et": "et",
    "fo": "fo",
    "fi": "fi",
    "fr": "fr",
    "de": "de",
    "el": "el",
    "hu": "hu",
    "is": "is",
    "it": "it",
    "lv": "lv",
    "lt": "lt",
    "lb": "lb",
    "no": "no",
    "pl": "pl",
    "pt": "pt-pt",
    "ro": "ro",
    "sr": "sr",
    "sk": "sk",
    "sl": "sl",
    "es": "es",
    "sv": "sv",
    "uk": "uk",
}


def main() -> None:
    """Prepare each language, publishing only when explicitly requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language", choices=LANGUAGE_SOURCES, help="One language only"
    )
    parser.add_argument("--publish", action="store_true", help="Upload to the Hub")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    codes = [args.language] if args.language else LANGUAGE_SOURCES
    for code in codes:
        dataset, repo_id = build_dataset(language_code=code)
        logger.info("Prepared %s (%d test rows)", repo_id, len(dataset["test"]))
        if args.publish:
            dataset.push_to_hub(repo_id=repo_id, private=True)


def build_dataset(language_code: str) -> tuple[DatasetDict, str]:
    """Build a deterministic test-only dataset from a source language subset.

    Args:
        language_code:
            The EuroEval two-letter language code.

    Returns:
        The dataset and its destination Hub repository ID. Truncated datasets
        receive the ``-mini`` suffix.

    Raises:
        ValueError:
            If the language code is not supported.
    """
    if language_code not in LANGUAGE_SOURCES:
        raise ValueError(f"Unsupported MultiBFCL language: {language_code}")
    source = load_dataset(
        path=SOURCE_REPO, name=LANGUAGE_SOURCES[language_code], split="test"
    )
    # A fixed shuffle avoids systematically excluding the final BFCL categories.
    truncated = len(source) > MAX_TEST_SAMPLES
    test = source.shuffle(seed=42).select(range(min(len(source), MAX_TEST_SAMPLES)))
    test = test.map(_convert_row, remove_columns=test.column_names)
    suffix = "-mini" if truncated else ""
    repo_id = f"EuroEval/multi-bfcl-{language_code}{suffix}"
    return DatasetDict({"test": test}), repo_id


def _convert_row(row: dict[str, str]) -> dict[str, str]:
    """Convert a source record into tool-calling metric input and reference.

    Args:
        row:
            A source record with JSON-encoded question, function and ground truth.

    Returns:
        A record containing prompt text, tool schema and reference calls.
    """
    question = json.loads(row["question"])[0][0]["content"]
    functions = json.loads(row["function"])
    ground_truth = json.loads(row["ground_truth"])
    function_str = json.dumps(functions, ensure_ascii=False)
    return {
        "text": f"Functions:\n{function_str}\nQuestion: {question}",
        "function": function_str,
        "target_text": json.dumps(ground_truth, ensure_ascii=False),
    }


if __name__ == "__main__":
    main()
