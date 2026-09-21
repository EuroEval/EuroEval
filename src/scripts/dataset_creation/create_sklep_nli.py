# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets>=3.5.0",
#     "huggingface-hub>=0.24.0",
# ]
# ///

"""Create the SKLEP three-way Slovak NLI mini dataset."""

from datasets import Dataset, DatasetDict, load_dataset

try:
    from dataset_utils import MAX_TEST, MAX_TRAIN, MAX_VAL, cap_split, upload_private
except ModuleNotFoundError:
    from src.scripts.dataset_creation.dataset_utils import (
        MAX_TEST,
        MAX_TRAIN,
        MAX_VAL,
        cap_split,
        upload_private,
    )

SOURCE_DATASET = "slovak-nlp/sklep"
TARGET_DATASET = "EuroEval/sklep-nli-mini"
LABELS = ("entailment", "neutral", "contradiction")


def main() -> None:
    """Download, process, and privately upload the SKLEP NLI dataset."""
    source = load_dataset(path=SOURCE_DATASET, name="nli")
    assert isinstance(source, DatasetDict)
    dataset = process_dataset(raw_dataset=source)
    upload_private(dataset=dataset, repository_id=TARGET_DATASET)


def process_dataset(raw_dataset: DatasetDict) -> DatasetDict:
    """Convert SKLEP NLI pairs to EuroEval's text and label schema.

    Returns:
        The processed dataset with independently capped splits.
    """
    return DatasetDict(
        {
            "train": _process_split(
                cap_split(dataset=raw_dataset["train"], maximum=MAX_TRAIN)
            ),
            "val": _process_split(
                cap_split(dataset=raw_dataset["validation"], maximum=MAX_VAL)
            ),
            "test": _process_split(
                cap_split(dataset=raw_dataset["test"], maximum=MAX_TEST)
            ),
        }
    )


def _process_split(source: Dataset) -> Dataset:
    """Convert one NLI source split.

    Returns:
        The processed split.
    """
    rows = [
        {
            "text": (
                f"Prvé tvrdenie: {row['premise']}\nDruhé tvrdenie: {row['hypothesis']}"
            ),
            "label": LABELS[int(row["label"])],
        }
        for row in source
    ]
    return Dataset.from_list(rows)


if __name__ == "__main__":
    main()
