"""Tests for dataset preprocessing utilities."""

from datasets import Dataset, DatasetDict

from euroeval.dataset_preprocessors import preprocess_wmt24pp_en_da


def test_preprocess_wmt24pp_en_da_handles_all_splits() -> None:
    """Test that WMT24++ preprocessing renames columns across splits."""
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "source": ["a", "b"],
                    "target": ["x", "y"],
                    "is_bad_source": [False, True],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "source": ["c"],
                    "target": ["z"],
                }
            ),
        }
    )

    processed = preprocess_wmt24pp_en_da(dataset)

    assert "text" in processed["train"].column_names
    assert "target_text" in processed["train"].column_names
    assert "source" not in processed["train"].column_names
    assert "target" not in processed["train"].column_names
    assert len(processed["train"]) == 1

    assert "text" in processed["test"].column_names
    assert "target_text" in processed["test"].column_names
    assert "source" not in processed["test"].column_names
    assert "target" not in processed["test"].column_names
    assert len(processed["test"]) == 1
