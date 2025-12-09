# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Catalan GuiaCat sentiment dataset and upload to HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the Catalan GuiaCat sentiment dataset and upload to HF Hub."""
    # Load source dataset
    repo_id = "projecte-aina/GuiaCat"
    dataset = load_dataset(repo_id)
    assert isinstance(dataset, DatasetDict)
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Process data
    train_df = process(df=train_df)
    val_df = process(df=val_df)
    test_df = process(df=test_df)

    # Make splits
    train_df_final, val_df_final, test_df_final = make_splits(
        train_df=train_df, val_df=val_df, test_df=test_df
    )

    # Create dataset dictionary
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df_final),
            "val": Dataset.from_pandas(val_df_final),
            "test": Dataset.from_pandas(test_df_final),
        }
    )

    # Push to HF Hub
    dataset_id = "EuroEval/guia-cat-mini"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset_dict.push_to_hub(dataset_id, private=True)


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe.

    Args:
        df: The dataframe to process.

    Returns:
        The processed dataframe.
    """
    # Map labels
    label_mapping = {
        "molt bo": "positive",
        "bo": "positive",
        "regular": "neutral",
        "dolent": "negative",
        "molt dolent": "negative",
    }
    df["label"] = df["label"].map(lambda x: label_mapping[x])

    # Keep only text and label columns
    keep_columns = ["text", "label"]
    df = df.loc[keep_columns]
    return df


def make_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make dataset splits.

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

    # Train split
    train_df_final = train_df.sample(n=train_size, random_state=42)
    train_df_remaining = train_df[~train_df.index.isin(train_df_final.index)]

    # Validation split
    val_df_final = val_df.sample(n=val_size, random_state=42)

    # Test split
    n_missing_test_samples = test_size - len(test_df)
    test_df_additional = train_df_remaining.sample(
        n=n_missing_test_samples, random_state=42
    )
    test_df_final = pd.concat([test_df, test_df_additional], ignore_index=True)

    # Reset the index
    train_df_final = train_df_final.reset_index(drop=True)
    val_df_final = val_df_final.reset_index(drop=True)
    test_df_final = test_df_final.reset_index(drop=True)

    assert isinstance(train_df_final, pd.DataFrame)
    assert isinstance(val_df_final, pd.DataFrame)
    assert isinstance(test_df_final, pd.DataFrame)

    return train_df_final, val_df_final, test_df_final


if __name__ == "__main__":
    main()
