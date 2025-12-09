# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the DACSA summarization datasets and upload to HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi

from .constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE


def main() -> None:
    """Create the DACSA summarization datasets and upload to HF Hub."""
    # Load the DACSA dataset
    dacsa_id = "ELiRF/dacsa"
    languages = {"ca": "catalan", "es": "spanish"}

    for lang, subset_name in languages.items():
        dataset = load_dataset(dacsa_id, subset_name)
        assert isinstance(dataset, DatasetDict)

        train_df = dataset["train"].to_pandas()
        val_df = dataset["validation"].to_pandas()
        test_df = dataset["test.ni"].to_pandas()
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

        mini_dataset_id = f"EuroEval/dacsa-{lang}-mini"
        HfApi().delete_repo(mini_dataset_id, repo_type="dataset", missing_ok=True)
        mini_dataset.push_to_hub(mini_dataset_id, private=True)


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe.

    Args:
        df: A dataframe to process.

    Returns:
        Processed dataframe.
    """
    # Renames
    df = df.rename(columns={"article": "text", "summary": "target_text"})

    # Keep only samples where the text is not very large or small
    lengths = df.text.str.len()
    lower_bound = MIN_NUM_CHARS_IN_ARTICLE
    upper_bound = MAX_NUM_CHARS_IN_ARTICLE
    df = df.loc[lengths.between(lower_bound, upper_bound)]

    # Keep only the necessary columns
    keep_columns = ["text", "target_text"]
    df = df.loc[keep_columns]
    return df


def make_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make the splits.

    Args:
        train_df: A dataframe to make the train split from.
        val_df: A dataframe to make the validation split from.
        test_df: A dataframe to make the test split from.

    Returns:
        The train, validation, and test splits.
    """
    # Create train split
    train_size = 1024
    train_df_final = train_df.sample(n=train_size, random_state=4242)

    # Create validation split
    val_size = 256
    val_df_final = val_df.sample(n=val_size, random_state=4242)

    # Create test split
    test_size = 2048
    test_df_final = test_df.sample(n=test_size, random_state=4242)

    # Reset the index
    train_df_final = train_df_final.reset_index(drop=True)
    val_df_final = val_df_final.reset_index(drop=True)
    test_df_final = test_df_final.reset_index(drop=True)

    return train_df_final, val_df_final, test_df_final


if __name__ == "__main__":
    main()
