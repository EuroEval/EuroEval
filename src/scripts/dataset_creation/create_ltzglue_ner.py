# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "scikit-learn==1.6.1",
# ]
# ///

"""Create the ltzGLUE NER dataset and upload to HF Hub."""

import logging

import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_URL = "https://media.githubusercontent.com/media/plumaj/ltzGLUE/main/data/ner"


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

    final_train, final_val, final_test = _create_splits(train_df, val_df, test_df)

    logger.info(
        f"Created splits: {len(final_train)} train, {len(final_val)} val, "
        f"{len(final_test)} test"
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(final_train[["tokens", "labels"]]),
            "val": Dataset.from_pandas(final_val[["tokens", "labels"]]),
            "test": Dataset.from_pandas(final_test[["tokens", "labels"]]),
        }
    )

    dataset_id = "EuroEval/ltzglue-ner"
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


def _create_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create standardized EuroEval splits (1024/256/2048).

    Args:
        train_df:
            Training data.
        val_df:
            Validation data.
        test_df:
            Test data.

    Returns:
        Tuple of (train, val, test) DataFrames with capped sizes.
    """
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)

    n_train = min(1024, int(len(all_data) * 0.5))
    n_val = min(256, int(len(all_data) * 0.15))

    train_data, temp = train_test_split(all_data, train_size=n_train, random_state=42)
    n_test = min(2048, len(temp) - n_val)
    val_data, test_data = train_test_split(
        temp, test_size=n_test, random_state=42
    )

    for df in [train_data, val_data, test_data]:
        df.reset_index(drop=True, inplace=True)

    return train_data, val_data, test_data


if __name__ == "__main__":
    main()
