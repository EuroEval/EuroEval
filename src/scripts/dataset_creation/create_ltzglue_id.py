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

"""Create the ltzGLUE ID dataset and upload to HF Hub."""

import logging

import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_URL = "https://media.githubusercontent.com/media/plumaj/ltzGLUE/main/data/id"


def main() -> None:
    """Create the ltzGLUE-ID dataset.

    Note that ltzGLUE ID only provides lb.valid.json and lb.test.json.
    No training data is available.
    """
    logger.info("Downloading ltzGLUE-ID data from GitHub...")
    val_data = _download_split("lb.valid")
    test_data = _download_split("lb.test")

    logger.info(
        f"Downloaded: {len(val_data)} val, {len(test_data)} test (no training data)"
    )

    combined = pd.concat(
        [_load_split(val_data), _load_split(test_data)], ignore_index=True
    )

    final_train, final_val, final_test = _make_splits(combined)

    logger.info(
        f"Created splits: {len(final_train)} train, {len(final_val)} val, "
        f"{len(final_test)} test"
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(final_train[["text", "label"]]),
            "val": Dataset.from_pandas(final_val[["text", "label"]]),
            "test": Dataset.from_pandas(final_test[["text", "label"]]),
        }
    )

    dataset_id = "EuroEval/ltzglue-id"
    logger.info(f"Uploading to {dataset_id}...")

    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)
    logger.info(f"✓ Uploaded {dataset_id}")


def _download_split(split: str) -> list[dict]:
    """Download a single split from GitHub.

    Args:
        split:
            Split name (lb.valid, lb.test).

    Returns:
        List of records from the JSON file.
    """
    url = f"{BASE_URL}/{split}.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def _load_split(data: list[dict]) -> pd.DataFrame:
    """Load intent detection data.

    Args:
        data:
            List of records from the JSON file.

    Returns:
        DataFrame with text and label columns.
    """
    return pd.DataFrame(
        [{"text": item["text"], "label": str(item["label"])} for item in data]
    )


def _make_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits without stratification.

    Args:
        df:
            Combined dataset.

    Returns:
        Tuple of (train, val, test) DataFrames.

    Note:
        Does not use stratification as ID has classes with very few samples.
    """
    n = len(df)
    n_train = min(1024, int(n * 0.5))
    n_val = min(256, int(n * 0.15))

    train, temp = train_test_split(df, train_size=n_train, random_state=42)
    val, test = train_test_split(temp, train_size=n_val / len(temp), random_state=42)

    for d in [train, val, test]:
        d.reset_index(drop=True, inplace=True)
    return train, val, test


if __name__ == "__main__":
    main()
