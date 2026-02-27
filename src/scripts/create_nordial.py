# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the NorDial dialect classification dataset and upload it to the HF Hub."""

import logging

import pandas as pd
import requests
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi

from .constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_nordial")

BASE_URL = (
    "https://raw.githubusercontent.com/jerbarnes/nordial/refs/heads/main/"
    "tweet_level/data/{split}.json"
)


def main() -> None:
    """Create the NorDial dataset and upload it to the HF Hub."""
    train_df = build_dataframe(split="train")
    val_df = build_dataframe(split="dev")
    test_df = build_dataframe(split="test")

    # Only work with samples where the document is not very large or small
    train_df = filter_by_length(df=train_df)
    val_df = filter_by_length(df=val_df)
    test_df = filter_by_length(df=test_df)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    dataset_id = "EuroEval/nordial"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def build_dataframe(split: str) -> pd.DataFrame:
    """Build a dataframe from a NorDial JSON split.

    Args:
        split:
            The split name, one of "train", "dev", or "test".

    Returns:
        A dataframe with 'text' and 'label' columns.
    """
    url = BASE_URL.format(split=split)
    response = requests.get(url=url, timeout=30)
    response.raise_for_status()
    records = response.json()
    df = pd.DataFrame.from_records(records)
    df = df.rename(columns=dict(category="label"))
    df = df[["text", "label"]].reset_index(drop=True)
    return df


def filter_by_length(df: pd.DataFrame) -> pd.DataFrame:
    """Filter samples by text length.

    Args:
        df:
            The dataframe to filter.

    Returns:
        The filtered dataframe.
    """
    df = df.copy()
    df["text_len"] = df.text.str.len()
    df = df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )
    return df.drop(columns=["text_len"]).reset_index(drop=True)


if __name__ == "__main__":
    main()
