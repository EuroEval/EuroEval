"""Tests for the BeWiC dataset creation script."""

from datasets import Dataset, DatasetDict

from src.scripts.dataset_creation.create_bewic import process_dataset


def test_process_dataset_constructs_text_with_both_contexts() -> None:
    """The processed text identifies the target word and includes both contexts."""
    source_dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "word": ["слова"],
                    "sentence1": ["Першы кантэкст."],
                    "sentence2": ["Другі кантэкст."],
                    "label": [1],
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "word": ["слова"],
                    "sentence1": ["Першы кантэкст."],
                    "sentence2": ["Другі кантэкст."],
                    "label": [1],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "word": ["слова"],
                    "sentence1": ["Першы кантэкст."],
                    "sentence2": ["Другі кантэкст."],
                    "label": [1],
                }
            ),
        }
    )

    result = process_dataset(source_dataset=source_dataset)

    assert result["train"]["text"][0] == (
        "Слова: слова\nКантэкст 1: Першы кантэкст.\nКантэкст 2: Другі кантэкст."
    )


def test_process_dataset_maps_source_labels() -> None:
    """Source labels 0 and 1 become different- and same-sense labels."""
    source_dataset = DatasetDict(
        {
            "train": _make_split(size=2, prefix="train"),
            "validation": _make_split(size=2, prefix="validation"),
            "test": _make_split(size=2, prefix="test"),
        }
    )

    result = process_dataset(source_dataset=source_dataset)

    assert result["train"]["label"] == ["different_sense", "same_sense"]


def _make_split(size: int, prefix: str) -> Dataset:
    """Create a small source-shaped split for testing.

    Returns:
        A dataset with the same columns as the BeWiC source data.
    """
    return Dataset.from_dict(
        {
            "word": [f"word {index}" for index in range(size)],
            "sentence1": [f"{prefix} context one {index}" for index in range(size)],
            "sentence2": [f"{prefix} context two {index}" for index in range(size)],
            "start1": [0] * size,
            "end1": [1] * size,
            "start2": [0] * size,
            "end2": [1] * size,
            "label": [index % 2 for index in range(size)],
            "idx": list(range(size)),
        }
    )


def test_process_dataset_preserves_splits_and_caps_independently() -> None:
    """Each source split is capped independently and mapped to EuroEval columns."""
    source_dataset = DatasetDict(
        {
            "train": _make_split(size=1026, prefix="train"),
            "validation": _make_split(size=258, prefix="validation"),
            "test": _make_split(size=401, prefix="test"),
        }
    )

    result = process_dataset(source_dataset=source_dataset)

    assert list(result) == ["train", "val", "test"]
    assert [result[split].num_rows for split in result] == [1024, 256, 400]
    assert result["train"].column_names == ["text", "label"]
    assert result["train"]["text"][-1].startswith("Слова: word 1023")
    assert result["val"]["text"][0].startswith("Слова: word 0")
    assert result["test"]["text"][-1].startswith("Слова: word 399")
