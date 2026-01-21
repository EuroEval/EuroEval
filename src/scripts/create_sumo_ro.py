# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the SumO-Ro summarisation dataset."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi

from .constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE


def main() -> None:
    """Create the SumO-Ro summarisation dataset and upload to HF Hub."""
    dataset_id = "Gabrielaaaaaa/SumO-Ro"

    # Load the full dataset
    dataset = load_dataset(dataset_id)
    assert isinstance(dataset, DatasetDict)

    # Convert to pandas for processing
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    train_df = process_df(df=train_df)
    val_df = process_df(df=val_df)
    test_df = process_df(df=test_df)

    train_df_final, val_df_final, test_df_final = make_splits(
        train_df=train_df, val_df=val_df, test_df=test_df
    )

    # Collect datasets in a dataset dictionary
    mini_dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df_final, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df_final, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df_final, split=Split.TEST),
        }
    )

    # Create dataset ID
    mini_dataset_id = "EuroEval/sumo-ro-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(mini_dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    mini_dataset.push_to_hub(mini_dataset_id, private=True)


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe to have the text and target_text columns.

    Args:
        df: The dataframe to process.

    Returns:
        The processed dataframe.
    """
    # Only keep rows where column "input_text" starts with <SUMMARY>
    df = df.loc[df["input_text"].str.startswith("<SUMMARY>")]

    # Make text column which is the input_text column without the <SUMMARY> prefix
    df["text"] = df["input_text"].str.replace("<SUMMARY>", "", regex=False)

    # Strip leading and trailing whitespace
    df["text"] = df["text"].str.strip()

    # Only keep rows where the text is not very large or small
    lengths = df.text.str.len()
    lower_bound = MIN_NUM_CHARS_IN_ARTICLE
    upper_bound = MAX_NUM_CHARS_IN_ARTICLE
    df = df.loc[lengths.between(lower_bound, upper_bound)]

    # Only keep the text and target_text columns
    keep_columns = ["text", "target_text"]
    df = df.loc[keep_columns]
    return df


def make_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make splits for the dataset.

    Args:
        train_df: The training dataframe.
        val_df: The validation dataframe.
        test_df: The test dataframe.

    Returns:
        The final training, validation, and test dataframes.
    """
    # Split sizes
    train_size = 1024
    val_size = 256
    test_size = 2048

    # Create train split
    train_df_final = train_df.sample(n=train_size, random_state=4242)

    # Create validation split
    val_df_final = val_df.sample(n=val_size, random_state=4242)
    remaining_val_df = val_df[~val_df.index.isin(val_df_final.index)]

    # Create test split
    n_missing_test_samples = test_size - len(test_df)
    additional_test_samples = remaining_val_df.sample(
        n=n_missing_test_samples, random_state=4242
    )
    test_df_final = pd.concat([test_df, additional_test_samples], ignore_index=True)

    # Reset indices
    train_df_final = train_df_final.reset_index(drop=True)
    val_df_final = val_df_final.reset_index(drop=True)
    test_df_final = test_df_final.reset_index(drop=True)

    assert isinstance(train_df_final, pd.DataFrame)
    assert isinstance(val_df_final, pd.DataFrame)
    assert isinstance(test_df_final, pd.DataFrame)

    return train_df_final, val_df_final, test_df_final


if __name__ == "__main__":
    main()
