# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
# ]
# ///

"""Create the BeRTE-WD dataset and upload it to the Hugging Face Hub."""

import logging

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)

SOURCE_DATASET = "maaxap/BelarusianGLUE"
SOURCE_CONFIG = "bertewd"
TARGET_DATASET = "EuroEval/berte-wd-mini"
MAX_TRAIN = 1024
MAX_VAL = 256
MAX_TEST = 360


def main() -> None:
    """Create the BeRTE-WD dataset and upload it to the Hugging Face Hub."""
    raw_dataset = load_dataset(SOURCE_DATASET, SOURCE_CONFIG)
    dataset = process_dataset(raw_dataset=raw_dataset)

    logger.info("Uploading %s", TARGET_DATASET)
    dataset.push_to_hub(TARGET_DATASET, private=True)
    logger.info("Uploaded %s", TARGET_DATASET)


def process_dataset(raw_dataset: DatasetDict) -> DatasetDict:
    """Format BeRTE-WD and cap each original split independently.

    Args:
        raw_dataset:
            The source dataset with ``train``, ``validation`` and ``test`` splits.

    Returns:
        A EuroEval dataset with ``train``, ``val`` and ``test`` splits. Each row has a
        Belarusian premise/hypothesis input in ``text`` and a string NLI label.

    """
    return DatasetDict(
        {
            "train": _process_split(dataset=raw_dataset["train"], max_size=MAX_TRAIN),
            "val": _process_split(dataset=raw_dataset["validation"], max_size=MAX_VAL),
            "test": _process_split(dataset=raw_dataset["test"], max_size=MAX_TEST),
        }
    )


def _process_split(dataset: Dataset, max_size: int) -> Dataset:
    """Format and cap one source split without changing its order or membership.

    Returns:
        The formatted and capped split.
    """
    dataset = dataset.select(range(min(len(dataset), max_size)))
    return Dataset.from_dict(
        {
            "text": [
                _format_text(premise=row["text"], hypothesis=row["hypothesis"])
                for row in dataset
            ],
            "label": [_map_label(label=row["label"]) for row in dataset],
        }
    )


def _format_text(premise: str, hypothesis: str) -> str:
    """Format a premise and hypothesis as a clear Belarusian input.

    Returns:
        A labelled premise/hypothesis pair.
    """
    return f"Перадумова: {premise}\nГіпотэза: {hypothesis}"


def _map_label(label: int) -> str:
    """Map a BeRTE-WD label to EuroEval's binary NLI labels.

    Returns:
        The corresponding EuroEval label.

    Raises:
        ValueError:
            If the source label is not 0 or 1.
    """
    if label == 1:
        return "entailment"
    if label == 0:
        return "contradiction"
    raise ValueError(f"Unexpected BeRTE-WD label: {label}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
