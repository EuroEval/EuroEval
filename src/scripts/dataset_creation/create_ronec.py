# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the RoNEC NER dataset and upload it to the HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi

NER_CONVERSION_DICT = {
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}


def main() -> None:
    """Create the RoNEC NER dataset and upload it to the HF Hub."""
    # Define dataset ID
    repo_id = "community-datasets/ronec"

    # Download the dataset
    dataset = load_dataset(path=repo_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    train_df, val_df, test_df = convert_to_dataframe(dataset=dataset)

    # Process the dataframes
    train_df = process(df=train_df)
    val_df = process(df=val_df)
    test_df = process(df=test_df)

    # Make splits
    train_df_final, val_df_final, test_df_final = make_splits(
        train_df=train_df, val_df=val_df, test_df=test_df
    )

    # Create dataset dictionary
    mini_dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df_final),
            "val": Dataset.from_pandas(val_df_final),
            "test": Dataset.from_pandas(test_df_final),
        }
    )

    # Create dataset ID
    dataset_id = "EuroEval/ronec-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    mini_dataset.push_to_hub(dataset_id, private=True)


def convert_to_dataframe(
    dataset: DatasetDict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert the dataset to a dataframe.

    Args:
        dataset: The dataset to convert.

    Returns:
        The training, validation, and test dataframes.
    """
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    return train_df, val_df, test_df


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe.

    Args:
        df:
            The dataframe to process.

    Returns:
        The processed dataframe.
    """
    # Sanity check that the number of tokens and named entity tags are equal
    assert all(
        len(ner_tags) == len(tokens)
        for ner_tags, tokens in zip(df["ner_tags"], df["tokens"])
    ), "The length of `ner_tags` and `tokens` are not equal in each row."

    # Rename `ner_tags` to `labels`
    df.rename(columns={"ner_tags": "labels"}, inplace=True)

    # Convert the NER tags from IDs to strings
    df["labels"] = df["labels"].map(
        lambda ner_tags: [NER_CONVERSION_DICT.get(ner_tag, "O") for ner_tag in ner_tags]
    )

    # Keep only tokens and labels columns
    keep_columns = ["tokens", "labels"]
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
    # Sizes
    train_size = 1024
    val_size = 256
    test_size = 2048

    # Train
    train_df_final = train_df.sample(n=train_size, random_state=4242)

    # Val
    val_df_final = val_df.sample(n=val_size, random_state=4242)
    remaining_val_df = val_df[~val_df.index.isin(val_df_final.index)]

    # Test
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
