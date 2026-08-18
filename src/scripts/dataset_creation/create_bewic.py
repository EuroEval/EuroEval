# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
# ]
# ///

"""Create the BeWiC dataset and upload it to the Hugging Face Hub.

BeWiC is the Belarusian Word-in-Context task from the BelarusianGLUE benchmark. Given
one word in two sentences, the task is to determine whether it has the same sense in
both contexts.
"""

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi

SOURCE_REPOSITORY = "maaxap/BelarusianGLUE"
SOURCE_CONFIG = "bewic"
TARGET_REPOSITORY = "EuroEval/bewic-mini"

SPLIT_LIMITS = {"train": 1024, "validation": 256, "test": 400}
LABEL_MAPPING = {0: "different_sense", 1: "same_sense"}


def main() -> None:
    """Create the BeWiC dataset and upload it to the Hugging Face Hub."""
    source_dataset = load_dataset(
        path=SOURCE_REPOSITORY, name=SOURCE_CONFIG, token=True
    )
    assert isinstance(source_dataset, DatasetDict)

    dataset = process_dataset(source_dataset=source_dataset)

    HfApi().delete_repo(repo_id=TARGET_REPOSITORY, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(repo_id=TARGET_REPOSITORY, private=True)


def process_dataset(source_dataset: DatasetDict) -> DatasetDict:
    """Convert and cap the source BeWiC splits.

    The source dataset has separate train, validation and test splits. Selecting rows
    independently in each split preserves those boundaries and makes the conversion
    deterministic.

    Args:
        source_dataset:
            The BeWiC dataset loaded from the Hugging Face Hub.

    Returns:
        A dataset with EuroEval's ``text`` and ``label`` columns and train, val and test
        splits.
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
    """Select and format one BeWiC split.

    Args:
        dataset:
            A source BeWiC split.
        limit:
            The maximum number of rows to retain.

    Returns:
        The processed split with ``text`` and ``label`` columns.
    """
    dataset = dataset.select(range(min(dataset.num_rows, limit)))
    dataset = dataset.select_columns(["word", "sentence1", "sentence2", "label"])

    def format_sample(sample: dict[str, str | int]) -> dict[str, str]:
        return {
            "text": (
                f"Слова: {str(sample['word']).strip()}\n"
                f"Кантэкст 1: {str(sample['sentence1']).strip()}\n"
                f"Кантэкст 2: {str(sample['sentence2']).strip()}"
            ),
            "label": LABEL_MAPPING[int(sample["label"])],
        }

    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
    return dataset.select_columns(["text", "label"])


if __name__ == "__main__":
    main()
