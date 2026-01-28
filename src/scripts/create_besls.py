# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Belarusian Sentiment dataset and upload it to the HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi

from .constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa


def main() -> None:
    """Create the Belarusian Sentiment dataset and upload it to the HF Hub."""
    # Define the base download URL
    repo_id = "maaxap/BelarusianGLUE"

    # Download the dataset
    dataset = load_dataset(path=repo_id, name="besls", token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to dataframes
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    train_df = process(df=train_df)
    val_df = process(df=val_df)
    test_df = process(df=test_df)

    final_train_df, final_val_df, final_test_df = make_splits(
        df_train=train_df, df_val=val_df, df_test=test_df
    )

    # Create DatasetDict
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(final_train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(final_val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(final_test_df, split=Split.TEST),
        }
    )

    # Create dataset ID
    dataset_id = "EuroEval/besls"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe.

    Args:
        df: The dataframe to process.

    Returns:
        The processed dataframe.
    """
    # Rename sentence to text
    df.rename(columns={"sentence": "text"}, inplace=True)

    # Keep only relevant columns
    columns_to_keep = ["text", "label"]
    df = df[columns_to_keep]

    # Map labels
    label_mapping = {0: "negative", 1: "positive"}
    df["label"] = df["label"].map(lambda x: label_mapping[int(x)])

    # Filter by text length
    df = _filter_by_length(df)
    return df


def make_splits(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make splits for the dataset.

    Args:
        df_train: The training dataframe.
        df_val: The validation dataframe.
        df_test: The test dataframe.

    Returns:
        The final training, validation, and test dataframes.
    """
    train_size = 256
    val_size = 128

    # Sample train and val split and keep remaining samples for test
    final_train_df = df_train.sample(n=train_size, random_state=4242)
    remaining_train_df = df_train[~df_train.index.isin(final_train_df.index)]

    final_val_df = df_val.sample(n=val_size, random_state=4242)
    remaining_val_df = df_val[~df_val.index.isin(final_val_df.index)]

    final_test_df = pd.concat(
        [remaining_train_df, remaining_val_df, df_test], ignore_index=True
    )

    # Reset indices
    final_train_df = final_train_df.reset_index(drop=True)
    final_val_df = final_val_df.reset_index(drop=True)
    final_test_df = final_test_df.reset_index(drop=True)

    assert isinstance(final_train_df, pd.DataFrame)
    assert isinstance(final_val_df, pd.DataFrame)
    assert isinstance(final_test_df, pd.DataFrame)

    return final_train_df, final_val_df, final_test_df


def _filter_by_length(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe by text length.

    Args:
        df:
            The dataframe to filter.

    Returns:
        The filtered dataframe.
    """
    new_df = df.copy()
    new_df["text_len"] = new_df.text.str.len()
    return new_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )


if __name__ == "__main__":
    main()
