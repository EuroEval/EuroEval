"""Dataset preprocessing utilities."""

import typing as t

if t.TYPE_CHECKING:
    from datasets import DatasetDict


def preprocess_wmt24pp_en_da(dataset: "DatasetDict") -> "DatasetDict":
    """Prepare the WMT24++ en-da subset for EuroEval."""
    if "train" not in dataset:
        return dataset

    train = dataset["train"].filter(lambda x: not x["is_bad_source"])
    train = train.rename_columns({"source": "text", "target": "target_text"})
    dataset["train"] = train
    return dataset
