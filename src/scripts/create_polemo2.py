# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the PolEmo2-mini sentiment dataset and upload it to the HF Hub."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the PolEmo2-mini sentiment dataset and upload it to the HF Hub."""
    # Define the base download URL
    repo_id = "clarin-pl/polemo2-official"

    # Download the dataset
    dataset = load_dataset(path=repo_id)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to dataframes
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Process each split
    train_df = process(train_df)
    val_df = process(val_df)
    test_df = process(test_df)

    # Create validation split (use original validation split)
    val_size = 256
    val_df = val_df.sample(n=min(val_size, len(val_df)), random_state=4242)

    # Create test split - start with original test split, add from train if needed
    test_size = 2048
    if len(test_df) >= test_size:
        test_df = test_df.sample(n=test_size, random_state=4242)
    else:
        # Need to add samples from train split to reach test_size
        needed_samples = test_size - len(test_df)
        additional_test_df = train_df.sample(n=needed_samples, random_state=4242)
        test_df = pd.concat([test_df, additional_test_df], ignore_index=True)
        # Remove the samples we used for test from train
        train_df = train_df.drop(additional_test_df.index)

    # Create train split (remaining samples from train)
    train_size = 1024
    train_df = train_df.sample(n=min(train_size, len(train_df)), random_state=4242)

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/polemo2-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Process the PolEmo2 dataset.

    Args:
        df: The PolEmo2 dataset to process.

    Returns:
        The processed PolEmo2 dataset.
    """
    # Rename columns to match expected format
    df.rename(columns={"text": "text", "target": "label"}, inplace=True)

    # Map PolEmo2 labels to standard sentiment labels
    # PolEmo2 uses:
    #   0 (neutral),
    #   1 (negative),
    #   2 (positive),
    #   3 (ambiguous)
    # We map to: positive, neutral, negative and exclude ambiguous samples
    label_mapping = {2: "positive", 0: "neutral", 1: "negative"}
    df["label"] = df["label"].map(label_mapping)

    # Remove rows with unmapped labels (excludes ambiguous samples)
    df = df.dropna(subset=["label"])

    # Keep only text and label columns
    df = df[["text", "label"]]

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Filter by text length
    df["text_len"] = df["text"].str.len()
    df = df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )
    df = df.drop(columns=["text_len"])

    return df


if __name__ == "__main__":
    main()
