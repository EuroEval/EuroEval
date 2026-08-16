"""Tests for the BelaCoLA dataset creation script."""

from datasets import Dataset, DatasetDict

from src.scripts.dataset_creation.create_belacola import process_dataset


def test_process_dataset_preserves_splits_and_caps_independently() -> None:
    """Each source split is capped independently and mapped to EuroEval columns."""
    source_dataset = DatasetDict(
        {
            "train": _make_split(size=1026, prefix="train"),
            "validation": _make_split(size=258, prefix="validation"),
            "test": _make_split(size=301, prefix="test"),
        }
    )

    result = process_dataset(source_dataset=source_dataset)

    assert list(result) == ["train", "val", "test"]
    assert [result[split].num_rows for split in result] == [1024, 256, 300]
    assert result["train"].column_names == ["text", "label"]
    assert result["train"]["text"][-1] == "train 1023"
    assert result["val"]["text"][0] == "validation 0"
    assert result["test"]["text"][-1] == "test 299"
    assert result["train"]["label"][:2] == ["incorrect", "correct"]


def _make_split(size: int, prefix: str) -> Dataset:
    """Create a small source-shaped split for testing.

    Returns:
        A dataset with the same columns as the BelaCoLA source data.
    """
    return Dataset.from_dict(
        {
            "idx": list(range(size)),
            "label": [index % 2 for index in range(size)],
            "sentence": [f"{prefix} {index}" for index in range(size)],
            "source": ["rucola"] * size,
            "detailed_source": ["test"] * size,
        }
    )
