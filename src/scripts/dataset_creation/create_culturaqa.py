# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
# ]
# ///

"""Create the CulturaQA dataset and upload it to the HF Hub."""

import pandas as pd
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from huggingface_hub.hf_api import HfApi

MAX_TRAIN_SIZE = 1_024
EXPECTED_SOURCE_SPLIT_SIZES = dict(train=2_000, val=200, test=500)


def main() -> None:
    """Create the CulturaQA dataset and upload it to the HF Hub."""
    dataset_id = "IMISLab/CulturaQA"

    # Load the dataset with 'all' config
    raw_dataset = load_dataset(dataset_id, "all", token=True)
    assert isinstance(raw_dataset, DatasetDict)

    # Verify original split sizes
    split_sizes = {split: len(raw_dataset[split]) for split in raw_dataset}
    assert split_sizes == EXPECTED_SOURCE_SPLIT_SIZES, (
        f"Expected split sizes {EXPECTED_SOURCE_SPLIT_SIZES}, got {split_sizes}"
    )

    # Transform each split: question -> text, answer -> target_text
    train_df = raw_dataset["train"].to_pandas()
    val_df = raw_dataset["val"].to_pandas()
    test_df = raw_dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Select and rename columns to EuroEval format
    train_df = train_df[["id", "question", "answer", "category"]].rename(
        columns={"question": "text", "answer": "target_text"}
    )
    val_df = val_df[["id", "question", "answer", "category"]].rename(
        columns={"question": "text", "answer": "target_text"}
    )
    test_df = test_df[["id", "question", "answer", "category"]].rename(
        columns={"question": "text", "answer": "target_text"}
    )

    train_df = train_df.head(MAX_TRAIN_SIZE).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_df, split=Split.TRAIN)
    val_dataset = Dataset.from_pandas(val_df, split=Split.VALIDATION)
    test_dataset = Dataset.from_pandas(test_df, split=Split.TEST)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    )

    # Create dataset ID
    euroeval_dataset_id = "EuroEval/culturaqa-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(euroeval_dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(euroeval_dataset_id, private=True)


if __name__ == "__main__":
    main()
