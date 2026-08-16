"""Tests for the BeRTE-WD dataset creation script."""

import pytest
from datasets import Dataset, DatasetDict

from src.scripts.dataset_creation.create_berte_wd import _map_label, process_dataset


@pytest.mark.parametrize(
    ("source_label", "expected_label"), [(0, "contradiction"), (1, "entailment")]
)
def test_map_label(source_label: int, expected_label: str) -> None:
    """Source labels map to the two EuroEval NLI labels."""
    assert _map_label(label=source_label) == expected_label


def test_map_label_rejects_unknown_labels() -> None:
    """Unexpected source labels fail rather than silently corrupting data."""
    with pytest.raises(ValueError, match="Unexpected BeRTE-WD label"):
        _map_label(label=2)


def test_process_dataset_caps_splits_independently() -> None:
    """Processing preserves split membership while applying each cap separately."""
    source = DatasetDict(
        {
            "train": _source_split(1080),
            "validation": _source_split(360),
            "test": _source_split(360),
        }
    )

    result = process_dataset(raw_dataset=source)

    assert {split: len(dataset) for split, dataset in result.items()} == {
        "train": 1024,
        "val": 256,
        "test": 360,
    }
    assert result["train"]["text"][-1].startswith("Перадумова: перадумова 1023")
    assert result["val"]["text"][-1].startswith("Перадумова: перадумова 255")
    assert result["test"]["text"][-1].startswith("Перадумова: перадумова 359")


def _source_split(size: int) -> Dataset:
    """Create a source-shaped split with identifiable rows.

    Returns:
        A source-shaped dataset split.
    """
    return Dataset.from_dict(
        {
            "text": [f"перадумова {index}" for index in range(size)],
            "hypothesis": [f"гіпотэза {index}" for index in range(size)],
            "label": [index % 2 for index in range(size)],
        }
    )


def test_process_dataset_formats_pairs_and_maps_labels() -> None:
    """Each output row combines both statements and uses binary NLI labels."""
    source = DatasetDict(
        {
            "train": _source_split(2),
            "validation": _source_split(1),
            "test": _source_split(1),
        }
    )

    result = process_dataset(raw_dataset=source)

    assert result["train"][0] == {
        "text": "Перадумова: перадумова 0\nГіпотэза: гіпотэза 0",
        "label": "contradiction",
    }
    assert result["train"][1]["label"] == "entailment"
