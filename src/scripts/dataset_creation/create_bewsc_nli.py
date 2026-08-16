# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
# ]
# ///

"""Create the BeWSC-NLI dataset and upload it to the Hugging Face Hub."""

import logging

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)

SOURCE_DATASET = "maaxap/BelarusianGLUE"
SOURCE_CONFIG = "bewsc_as_wnli"
TARGET_DATASET = "EuroEval/bewsc-nli"


def main() -> None:
    """Create the BeWSC-NLI dataset and upload it to the Hugging Face Hub."""
    raw_dataset = load_dataset(path=SOURCE_DATASET, name=SOURCE_CONFIG)
    dataset = process_dataset(raw_dataset=raw_dataset)

    logger.info("Uploading %s", TARGET_DATASET)
    dataset.push_to_hub(TARGET_DATASET, private=True)
    logger.info("Uploaded %s", TARGET_DATASET)


def process_dataset(raw_dataset: DatasetDict) -> DatasetDict:
    """Format BeWSC-NLI while preserving the source split membership and order.

    Args:
        raw_dataset:
            The source dataset with ``train``, ``validation`` and ``test`` splits.

    Returns:
        A EuroEval dataset with ``train``, ``val`` and ``test`` splits. Each row has a
        Belarusian premise/hypothesis input in ``text`` and a string NLI label.
    """
    return DatasetDict(
        {
            "train": _process_split(dataset=raw_dataset["train"]),
            "val": _process_split(dataset=raw_dataset["validation"]),
            "test": _process_split(dataset=raw_dataset["test"]),
        }
    )


def _process_split(dataset: Dataset) -> Dataset:
    """Format one source split without changing its order or membership.

    Args:
        dataset:
            A source split containing ``sentence1``, ``sentence2`` and ``label``.

    Returns:
        The formatted split with only ``text`` and ``label`` columns.
    """
    return Dataset.from_dict(
        {
            "text": [
                _format_text(premise=row["sentence1"], hypothesis=row["sentence2"])
                for row in dataset
            ],
            "label": [_map_label(label=row["label"]) for row in dataset],
        }
    )


def _format_text(premise: str, hypothesis: str) -> str:
    """Format a premise and hypothesis as a clear Belarusian input.

    Args:
        premise:
            The source premise.
        hypothesis:
            The source hypothesis.

    Returns:
        A labelled premise/hypothesis pair.
    """
    return f"Перадумова: {premise}\nГіпотэза: {hypothesis}"


def _map_label(label: int) -> str:
    """Map a BeWSC-NLI label to EuroEval's binary NLI labels.

    Args:
        label:
            The source label, where 1 denotes entailment and 0 denotes
            non-entailment.

    Returns:
        The corresponding EuroEval label.

    Raises:
        ValueError:
            If the source label is not 0 or 1.
    """
    if label == 1:
        return "entailment"
    if label == 0:
        return "non_entailment"
    raise ValueError(f"Unexpected BeWSC-NLI label: {label}")


if __name__ == "__main__":
    main()
