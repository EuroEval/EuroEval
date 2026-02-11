"""Dataset preprocessing utilities."""

import typing as t

if t.TYPE_CHECKING:
    from datasets import DatasetDict


def preprocess_wmt24pp_en_da(dataset: "DatasetDict") -> "DatasetDict":
    """Prepare the WMT24++ en-da subset for EuroEval."""
    for split_name, split in dataset.items():
        if "is_bad_source" in split.column_names:
            split = split.filter(lambda x: not x["is_bad_source"])
        if "source" in split.column_names and "target" in split.column_names:
            split = split.rename_columns({"source": "text", "target": "target_text"})
        dataset[split_name] = split
    return dataset
