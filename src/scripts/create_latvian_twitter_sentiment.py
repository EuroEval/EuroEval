# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Latvian Twitter Sentiment dataset and upload it to the HF Hub."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the Latvian Twitter Sentiment dataset and upload it to the HF Hub."""
    # Define the base download URL
    repo_id = "matiss/Latvian-Twitter-Eater-Corpus-Sentiment"

    # Download the dataset
    dataset = load_dataset(path=repo_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to dataframes
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Drop unnecessary columns, keeping only 'text' and 'label'
    columns_to_keep = ["text", "label"]
    columns_to_drop = [col for col in train_df.columns if col not in columns_to_keep]
    train_df.drop(columns=columns_to_drop, inplace=True)
    test_df.drop(columns=columns_to_drop, inplace=True)

    # Map integer labels to text labels
    # According to the dataset: 0=neutral, 1=positive, 2=negative
    label_mapping = {0: "neutral", 1: "positive", 2: "negative"}
    train_df["label"] = train_df["label"].map(label_mapping)
    test_df["label"] = test_df["label"].map(label_mapping)

    def filter_by_length(df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe by text length.

        Args:
            df (pd.DataFrame): The dataframe to filter.

        Returns:
            pd.DataFrame: The filtered dataframe.
        """
        new_df = df.copy()
        new_df["text_len"] = new_df.text.str.len()
        return new_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
            "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
        )

    train_df = filter_by_length(train_df)
    test_df = filter_by_length(test_df)

    # Create test split
    # Take samples from train split to each 2048 test samples
    test_size = 2048
    additional_needed = test_size - len(test_df)
    additional_test_samples = train_df.sample(n=additional_needed, random_state=4242)
    test_df = pd.concat([test_df, additional_test_samples], ignore_index=True)
    train_df = train_df.drop(index=additional_test_samples.index.tolist())

    # Create validation split from training data
    val_size = 256
    val_df = train_df.sample(n=val_size, random_state=4242)
    train_df = train_df.drop(index=val_df.index.tolist())

    # Create train split
    train_size = 1024
    train_df = train_df.sample(n=train_size, random_state=4242)

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Create DatasetDict
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/latvian-twitter-sentiment-mini"

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
