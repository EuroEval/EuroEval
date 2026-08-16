"""Tests for the BeWSC-NLI dataset creation script."""

import pytest
from datasets import Dataset, DatasetDict

from src.euroeval.dataset_configs.belarusian import BEWSC_NLI_CONFIG
from src.scripts.dataset_creation.create_bewsc_nli import (
    _format_text,
    _map_label,
    process_dataset,
)


def test_config_uses_binary_nli_prompts() -> None:
    """The dataset config exposes only entailment and non-entailment choices."""
    assert BEWSC_NLI_CONFIG.name == "bewsc-nli"
    assert BEWSC_NLI_CONFIG.source == "EuroEval/bewsc-nli"
    assert BEWSC_NLI_CONFIG.labels == ["entailment", "non_entailment"]
    assert set(BEWSC_NLI_CONFIG.prompt_label_mapping.values()) == {
        "праўда",
        "не вынікае",
    }
    assert "нейтраль" not in BEWSC_NLI_CONFIG.prompt_prefix
    assert "супярэч" not in BEWSC_NLI_CONFIG.prompt_prefix


def test_format_text() -> None:
    """A source sentence pair is formatted as a Belarusian premise and hypothesis."""
    assert _format_text(premise="Першая частка", hypothesis="Другая частка") == (
        "Перадумова: Першая частка\nГіпотэза: Другая частка"
    )


@pytest.mark.parametrize(
    ("source_label", "expected_label"), [(0, "non_entailment"), (1, "entailment")]
)
def test_map_label(source_label: int, expected_label: str) -> None:
    """Source labels map to the two EuroEval NLI labels."""
    assert _map_label(label=source_label) == expected_label


def test_map_label_rejects_unknown_labels() -> None:
    """Unexpected source labels fail rather than silently corrupting data."""
    with pytest.raises(ValueError, match="Unexpected BeWSC-NLI label"):
        _map_label(label=2)


def test_process_dataset_preserves_splits_and_output_columns() -> None:
    """Processing preserves source split order and emits only EuroEval columns."""
    source = DatasetDict(
        {
            "train": _source_split(570),
            "validation": _source_split(200),
            "test": _source_split(200),
        }
    )

    result = process_dataset(raw_dataset=source)

    assert {split: len(dataset) for split, dataset in result.items()} == {
        "train": 570,
        "val": 200,
        "test": 200,
    }
    assert result["train"].column_names == ["text", "label"]
    assert result["val"].column_names == ["text", "label"]
    assert result["test"].column_names == ["text", "label"]
    assert result["train"][0]["text"].endswith("гіпотэза 0")
    assert result["train"][-1]["text"].endswith("гіпотэза 569")
    assert result["val"][-1]["text"].endswith("гіпотэза 199")
    assert result["test"][-1]["text"].endswith("гіпотэза 199")


def _source_split(size: int) -> Dataset:
    """Create a source-shaped split with identifiable rows.

    Args:
        size:
            Number of rows in the split.

    Returns:
        A source-shaped dataset split.
    """
    return Dataset.from_dict(
        {
            "index": [str(index) for index in range(size)],
            "sentence1": [f"перадумова {index}" for index in range(size)],
            "sentence2": [f"гіпотэза {index}" for index in range(size)],
            "label": [index % 2 for index in range(size)],
        }
    )
