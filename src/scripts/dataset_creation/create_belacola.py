# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
# ]
# ///

"""Create the BelaCoLA dataset and upload it to the Hugging Face Hub."""

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi

SOURCE_REPOSITORY = "maaxap/BelarusianGLUE"
SOURCE_CONFIG = "belacola_in_domain"
TARGET_REPOSITORY = "EuroEval/belacola-mini"

SPLIT_LIMITS = {"train": 1024, "validation": 256, "test": 300}
LABEL_MAPPING = {0: "incorrect", 1: "correct"}


def main() -> None:
    """Create the BelaCoLA dataset and upload it to the Hugging Face Hub."""
    source_dataset = load_dataset(
        path=SOURCE_REPOSITORY, name=SOURCE_CONFIG, token=True
    )
    assert isinstance(source_dataset, DatasetDict)

    dataset = process_dataset(source_dataset=source_dataset)

    HfApi().delete_repo(repo_id=TARGET_REPOSITORY, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(repo_id=TARGET_REPOSITORY, private=True)


def process_dataset(source_dataset: DatasetDict) -> DatasetDict:
    """Convert and cap the source BelaCoLA splits.

    The source dataset is already shuffled and has separate in-domain train,
    validation and test splits. Selecting the first rows in each split keeps the
    conversion deterministic while preserving those split boundaries.

    Args:
        source_dataset:
            The in-domain BelaCoLA dataset loaded from the Hugging Face Hub.

    Returns:
        A dataset with EuroEval's ``text`` and ``label`` columns and train, val
        and test splits.
    """
    return DatasetDict(
        {
            output_split: _process_split(
                dataset=source_dataset[source_split], limit=SPLIT_LIMITS[source_split]
            )
            for output_split, source_split in {
                "train": "train",
                "val": "validation",
                "test": "test",
            }.items()
        }
    )


def _process_split(dataset: Dataset, limit: int) -> Dataset:
    """Select, rename and label one BelaCoLA split.

    Returns:
        The processed split with ``text`` and ``label`` columns.
    """
    dataset = dataset.select(range(min(dataset.num_rows, limit)))
    dataset = dataset.select_columns(["sentence", "label"])
    dataset = dataset.rename_column("sentence", "text")
    return dataset.map(lambda sample: {"label": LABEL_MAPPING[sample["label"]]})


if __name__ == "__main__":
    main()
