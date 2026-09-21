# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets>=3.5.0",
#     "huggingface-hub>=0.24.0",
# ]
# ///

"""Create the SKLEP binary Slovak RTE mini dataset."""

from datasets import Dataset, DatasetDict, load_dataset

try:
    from dataset_utils import make_dataset_dict, upload_private
except ModuleNotFoundError:
    from src.scripts.dataset_creation.dataset_utils import (
        make_dataset_dict,
        upload_private,
    )

SOURCE_DATASET = "slovak-nlp/sklep"
TARGET_DATASET = "EuroEval/sklep-rte-mini"
LABELS = ("entailment", "not entailment")


def main() -> None:
    """Download, process, and privately upload the SKLEP RTE dataset."""
    source = load_dataset(path=SOURCE_DATASET, name="rte")
    assert isinstance(source, DatasetDict)
    dataset = process_dataset(raw_dataset=source)
    upload_private(dataset=dataset, repository_id=TARGET_DATASET)


def process_dataset(raw_dataset: DatasetDict) -> DatasetDict:
    """Convert SKLEP RTE pairs to EuroEval's binary NLI schema.

    Returns:
        The processed dataset with independently capped splits.
    """
    return make_dataset_dict(
        train=_process_split(raw_dataset["train"]),
        validation=_process_split(raw_dataset["validation"]),
        test=_process_split(raw_dataset["test"]),
    )


def _process_split(source: Dataset) -> Dataset:
    """Convert one RTE source split.

    Returns:
        The processed split.
    """
    rows = [
        {
            "text": f"Prvé tvrdenie: {row['text1']}\nDruhé tvrdenie: {row['text2']}",
            "label": LABELS[int(row["label"])],
        }
        for row in source
    ]
    return Dataset.from_list(rows)


if __name__ == "__main__":
    main()
