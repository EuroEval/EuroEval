# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the ilpost summarisation dataset."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE
from datasets import Dataset, DatasetDict, Split
from datasets.load import load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the ilpost summarisation dataset and upload to HF Hub."""
    dataset_id = "ARTeLab/ilpost"

    # Load the dataset
    dataset = load_dataset(dataset_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    train_df.rename(columns={"source": "text", "target": "target_text"}, inplace=True)
    val_df.rename(columns={"source": "text", "target": "target_text"}, inplace=True)
    test_df.rename(columns={"source": "text", "target": "target_text"}, inplace=True)

    train_df.dropna(subset=["text", "target_text"], inplace=True)
    val_df.dropna(subset=["text", "target_text"], inplace=True)
    test_df.dropna(subset=["text", "target_text"], inplace=True)

    # Only work with samples where the text is not very large or small
    train_lengths = train_df.text.str.len()
    val_lengths = val_df.text.str.len()
    test_lengths = test_df.text.str.len()
    train_df = train_df[
        train_lengths.between(MIN_NUM_CHARS_IN_ARTICLE, MAX_NUM_CHARS_IN_ARTICLE)
    ]
    val_df = val_df[
        val_lengths.between(MIN_NUM_CHARS_IN_ARTICLE, MAX_NUM_CHARS_IN_ARTICLE)
    ]
    test_df = test_df[
        test_lengths.between(MIN_NUM_CHARS_IN_ARTICLE, MAX_NUM_CHARS_IN_ARTICLE)
    ]

    # Create validation split
    val_size = 256
    val_df = val_df.sample(n=val_size, random_state=4242)

    # Create train split
    train_size = 1024
    train_df = train_df.sample(n=train_size, random_state=4242)

    # Create test split
    test_size = 2048
    test_df = test_df.sample(n=test_size, random_state=4242)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/ilpost-sum"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api: HfApi = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
