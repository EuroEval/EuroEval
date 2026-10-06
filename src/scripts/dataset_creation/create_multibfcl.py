"""Prepare and publish multilingual BFCL validation and test datasets for EuroEval."""

import json
import logging

from datasets import DatasetDict, load_dataset

logger = logging.getLogger(__name__)

SOURCE_REPO = "syvai/multi-bfcl"
MAX_TEST_SAMPLES = 2048
MAX_VAL_SAMPLES = 256
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
    """Prepare and publish datasets for every supported language."""
    logging.basicConfig(level=logging.INFO)
    for code in LANGUAGE_SOURCES:
        dataset, repo_id = build_dataset(language_code=code)
        logger.info(
            "Prepared %s (%d validation, %d test rows)",
            repo_id,
            len(dataset["val"]),
            len(dataset["test"]),
        )
        dataset.push_to_hub(repo_id=repo_id, private=True)


def build_dataset(language_code: str) -> tuple[DatasetDict, str]:
    """Build disjoint validation and test splits from a source test subset.

    Args:
        language_code:
            The EuroEval two-letter language code.

    Returns:
        The dataset and its ``-mini`` destination Hub repository ID. The first
        2,048 shuffled records retain the published test split; the next 256
        become validation records.

    Raises:
        ValueError:
            If the language code is unsupported or fewer than 2,304 source
            records are available.
    """
    if language_code not in LANGUAGE_SOURCES:
        raise ValueError(f"Unsupported MultiBFCL language: {language_code}")
    source = load_dataset(
        path=SOURCE_REPO, name=LANGUAGE_SOURCES[language_code], split="test"
    )
    required = MAX_TEST_SAMPLES + MAX_VAL_SAMPLES
    if len(source) < required:
        raise ValueError(
            f"MultiBFCL {language_code} requires {required} distinct source rows; "
            f"found {len(source)}"
        )
    # Retain the original shuffled test prefix; draw validation only after it.
    shuffled = source.shuffle(seed=42)
    test = shuffled.select(range(MAX_TEST_SAMPLES))
    val = shuffled.select(range(MAX_TEST_SAMPLES, required))
    test = test.map(_convert_row, remove_columns=test.column_names)
    val = val.map(_convert_row, remove_columns=val.column_names)
    repo_id = f"EuroEval/multi-bfcl-{language_code}-mini"
    return DatasetDict({"val": val, "test": test}), repo_id


def _convert_row(row: dict[str, str]) -> dict[str, str]:
    """Convert a source record into tool-calling metric input and reference.

    Args:
        row:
            A source record with JSON-encoded question, function and ground truth.

    Returns:
        A record containing prompt text, tool schema and reference calls.
    """
    messages = json.loads(row["question"])[0]
    if len(messages) == 1 and messages[0]["role"] == "user":
        question = f"Question: {messages[0]['content']}"
    else:
        role_labels = {"system": "System", "user": "Question", "assistant": "Assistant"}
        question = "\n".join(
            f"{role_labels.get(message['role'], message['role'].title())}: "
            f"{message['content']}"
            for message in messages
        )
    functions = json.loads(row["function"])
    ground_truth = json.loads(row["ground_truth"])
    function_str = json.dumps(functions, ensure_ascii=False)
    return {
        "text": f"Functions:\n{function_str}\n{question}",
        "function": function_str,
        "target_text": json.dumps(ground_truth, ensure_ascii=False),
    }


if __name__ == "__main__":
    main()
