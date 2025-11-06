# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Slovenian NER dataset from SSJ500k and upload it to the HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the Slovenian NER dataset from SSJ500k and upload it to the HF Hub."""
    # Define dataset ID
    repo_id = "cjvt/ssj500k"

    # Download the dataset
    dataset = load_dataset(
        repo_id, "named_entity_recognition", split="train", trust_remote_code=True
    )
    df = dataset.to_pandas()

    # Process the dataframe
    df = process(df=df)

    # Define split sizes
    val_size = 256
    test_size = 2048
    train_size = 1024

    # Create validation split
    val_df = df.sample(n=val_size, random_state=42)

    # Remove validation samples from the full dataset
    temp_df = df.drop(val_df.index)

    # Create test split
    test_df = temp_df.sample(n=test_size, random_state=42)

    # Create train split
    train_df = temp_df.drop(test_df.index).sample(n=train_size, random_state=42)

    # Reset indices for each split
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset_dict = DatasetDict(
        train=Dataset.from_pandas(train_df),
        validation=Dataset.from_pandas(val_df),
        test=Dataset.from_pandas(test_df),
    )

    # Set dataset ID for the Hugging Face Hub
    dataset_id = "EuroEval/ssj500k-ner-mini"

    # Delete existing repo if needed
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push to Hugging Face Hub
    dataset_dict.push_to_hub(dataset_id, private=True)


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Process dataframe to ensure correct format.

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    df.rename(columns={"words": "tokens", "ne_tags": "labels"}, inplace=True)

    keep_columns = ["tokens", "labels"]
    df = df[keep_columns]

    assert all(
        len(tokens) == len(labels) for tokens, labels in zip(df["tokens"], df["labels"])
    ), "The length of `tokens` and `labels` are not equal in each row."

    return df


if __name__ == "__main__":
    main()
