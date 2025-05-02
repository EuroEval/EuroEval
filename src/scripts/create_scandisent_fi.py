# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Finnish part of the ScandiSent dataset and upload it to the HF Hub."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the Finnish part of the ScandiSent dataset and upload it to the HF Hub."""
    # Define the base download URL
    repo_id = "timpal0l/scandisent"

    # Download the dataset
    dataset = load_dataset(path=repo_id, token=True, split="train")
    assert isinstance(dataset, Dataset)

    # Convert the dataset to a dataframe
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Keep only Finnish samples
    df = df[df["language"] == "fi"]

    # Drop all columns except for `text` and `label_text`
    columns_to_drop = [col for col in df.columns if col not in ["text", "label"]]
    df.drop(columns=columns_to_drop, inplace=True)

    # Change labels from integer to string
    df["label"] = df["label"].map({0: "negative", 1: "positive"})

    # Create validation split
    val_size = 256
    val_df = df.sample(n=val_size, random_state=4242)
    df = df.drop(val_df.index)

    # Create test split
    test_size = 2048
    test_df = df.sample(n=test_size, random_state=4242)
    df = df.drop(test_df.index)

    # Create train split
    train_size = 1024
    train_df = df.sample(n=train_size, random_state=4242)

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Only work with samples where the document is not very large or small
    # We do it after we have made the splits to ensure that the dataset is minimally
    # affected.
    new_train_df = train_df.copy()
    new_train_df["text_len"] = new_train_df.text.str.len()
    new_train_df = new_train_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )
    new_val_df = val_df.copy()
    new_val_df["text_len"] = new_val_df.text.str.len()
    new_val_df = new_val_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )
    new_test_df = test_df.copy()
    new_test_df["text_len"] = new_test_df.text.str.len()
    new_test_df = new_test_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/scandisent-fi-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
