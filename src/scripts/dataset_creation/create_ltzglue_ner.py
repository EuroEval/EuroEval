# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the ltzGLUE NER dataset and upload to HF Hub."""

import logging

import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_URL = "https://media.githubusercontent.com/media/plumaj/ltzGLUE/main/data/ner"

# Capping limits for mini datasets
MAX_TRAIN = 1024
MAX_VAL = 256
MAX_TEST = 2048


def main() -> None:
    """Create the ltzGLUE-NER dataset and upload to HF Hub."""
    logger.info("Downloading ltzGLUE-NER data from GitHub...")
    train_data = _download_split("train")
    val_data = _download_split("dev")
    test_data = _download_split("test")

    logger.info(
        f"Downloaded: {len(train_data)} train, {len(val_data)} val, "
        f"{len(test_data)} test"
    )

    train_df = _load_split(train_data)
    val_df = _load_split(val_data)
    test_df = _load_split(test_data)

    # Cap each split independently, preserving source split boundaries
    train_df = _cap_split(train_df, MAX_TRAIN)
    val_df = _cap_split(val_df, MAX_VAL)
    test_df = _cap_split(test_df, MAX_TEST)

    logger.info(
        f"Capped splits: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test"
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df[["tokens", "labels"]]),
            "val": Dataset.from_pandas(val_df[["tokens", "labels"]]),
            "test": Dataset.from_pandas(test_df[["tokens", "labels"]]),
        }
    )

    dataset_id = "EuroEval/ltzglue-ner-mini"
    logger.info(f"Uploading to {dataset_id}...")

    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)
    logger.info(f"✓ Uploaded {dataset_id}")


def _download_split(split: str) -> list[dict]:
    """Download a single split from GitHub.

    Args:
        split:
            Split name (train, dev, test).

    Returns:
        List of records from the JSON file.
    """
    url = f"{BASE_URL}/{split}.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def _cap_split(df: pd.DataFrame, max_size: int) -> pd.DataFrame:
    """Cap a split to max_size rows using deterministic sampling.

    Args:
        df:
            DataFrame to cap.
        max_size:
            Maximum number of rows to keep.

    Returns:
        Capped DataFrame. Rows are selected deterministically from the start
        of the split to ensure reproducibility and preserve split membership.
    """
    if len(df) <= max_size:
        return df
    return df.iloc[:max_size].reset_index(drop=True)


def _load_split(data: list[dict]) -> pd.DataFrame:
    """Load NER data.

    Args:
        data:
            List of records from the JSON file.

    Returns:
        DataFrame with tokens and labels columns.
    """
    return pd.DataFrame(
        [{"tokens": item["tokens"], "labels": item["ner_tags"]} for item in data]
    )


if __name__ == "__main__":
    main()
