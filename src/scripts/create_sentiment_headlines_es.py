# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the SentimentHeadlines-es dataset and upload to HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the Spanish sentiment headlines dataset and upload to HF Hub."""
    # Define the base download URL
    repo_id = "pysentimiento/spanish-targeted-sentiment-headlines"

    # Download the dataset
    dataset = load_dataset(path=repo_id)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    train_df = dataset["train"].to_pandas()
    val_df = dataset["dev"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Concatenate the splits
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Rename titulo -> text
    df.rename(columns={"titulo": "text"}, inplace=True)

    # Create the label column
    df["label"] = df["label"].map({0: "negative", 1: "neutral", 2: "positive"})

    # Keep only columns text and label
    df = df[["text", "label"]]

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Create validation split
    val_size = 256
    val_df = df.sample(n=val_size, random_state=4242)

    # Create test split
    test_size = 1024
    filtered_df = df[~df.index.isin(val_df.index)]
    test_df = filtered_df.sample(n=test_size, random_state=4242)

    # Create train split
    train_df = filtered_df[~filtered_df.index.isin(test_df.index)]
    assert len(train_df) > 800, "The training set should have at least 800 samples."

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
    dataset_id = "EuroEval/sentiment-headlines-es"

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
