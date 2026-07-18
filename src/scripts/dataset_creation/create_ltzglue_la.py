# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the ltzGLUE LA dataset (binary and multi-class) and upload to HF Hub."""

import logging

import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_URL_BIN = (
    "https://media.githubusercontent.com/media/plumaj/ltzGLUE/main/data/la/binary"
)
BASE_URL_MULTI = (
    "https://media.githubusercontent.com/media/plumaj/ltzGLUE/main/data/la/multi"
)

# Capping limits for mini datasets
MAX_TRAIN = 1024
MAX_VAL = 256
MAX_TEST = 2048


def main() -> None:
    """Create both LA binary and multi-class datasets."""
    logger.info("Downloading ltzGLUE-LA binary data from GitHub...")
    bin_train = _download_split(BASE_URL_BIN, "train")
    bin_val = _download_split(BASE_URL_BIN, "dev")
    bin_test = _download_split(BASE_URL_BIN, "test")

    logger.info(
        f"Downloaded binary: {len(bin_train)} train, "
        f"{len(bin_val)} val, {len(bin_test)} test"
    )

    bin_train_df = _create_binary_df(bin_train)
    bin_val_df = _create_binary_df(bin_val)
    bin_test_df = _create_binary_df(bin_test)

    # Cap each split independently, preserving source split boundaries
    bin_train_df = _cap_split(bin_train_df, MAX_TRAIN)
    bin_val_df = _cap_split(bin_val_df, MAX_VAL)
    bin_test_df = _cap_split(bin_test_df, MAX_TEST)

    logger.info(
        f"Capped binary splits: {len(bin_train_df)} / "
        f"{len(bin_val_df)} / {len(bin_test_df)}"
    )

    bin_dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(bin_train_df[["text", "label"]]),
            "val": Dataset.from_pandas(bin_val_df[["text", "label"]]),
            "test": Dataset.from_pandas(bin_test_df[["text", "label"]]),
        }
    )

    HfApi().delete_repo(
        "EuroEval/ltzglue-la-mini", repo_type="dataset", missing_ok=True
    )
    bin_dataset.push_to_hub("EuroEval/ltzglue-la-mini", private=True)
    logger.info("✓ Uploaded EuroEval/ltzglue-la-mini (binary)")

    logger.info("Downloading ltzGLUE-LA multi-class data from GitHub...")
    mul_train = _download_split(BASE_URL_MULTI, "train")
    mul_val = _download_split(BASE_URL_MULTI, "dev")
    mul_test = _download_split(BASE_URL_MULTI, "test")

    logger.info(
        f"Downloaded multi: {len(mul_train)} train, "
        f"{len(mul_val)} val, {len(mul_test)} test"
    )

    mul_train_df = _create_multi_df(mul_train)
    mul_val_df = _create_multi_df(mul_val)
    mul_test_df = _create_multi_df(mul_test)

    # Cap each split independently, preserving source split boundaries
    mul_train_df = _cap_split(mul_train_df, MAX_TRAIN)
    mul_val_df = _cap_split(mul_val_df, MAX_VAL)
    mul_test_df = _cap_split(mul_test_df, MAX_TEST)

    logger.info(
        f"Capped multi splits: {len(mul_train_df)} / "
        f"{len(mul_val_df)} / {len(mul_test_df)}"
    )

    mul_dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(mul_train_df[["text", "label"]]),
            "val": Dataset.from_pandas(mul_val_df[["text", "label"]]),
            "test": Dataset.from_pandas(mul_test_df[["text", "label"]]),
        }
    )

    HfApi().delete_repo(
        "EuroEval/ltzglue-la-multi-mini", repo_type="dataset", missing_ok=True
    )
    mul_dataset.push_to_hub("EuroEval/ltzglue-la-multi-mini", private=True)
    logger.info("✓ Uploaded EuroEval/ltzglue-la-multi-mini (multi-class)")


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


def _download_split(base_url: str, split: str) -> list[dict]:
    """Download a single split from GitHub.

    Args:
        base_url:
            Base URL for the dataset.
        split:
            Split name (train, dev, test).

    Returns:
        List of records from the JSON file.
    """
    url = f"{base_url}/{split}.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def _create_binary_df(data: list[dict]) -> pd.DataFrame:
    """Create binary LA dataset (correct/incorrect).

    Args:
        data:
            List of records from the JSON file.

    Returns:
        DataFrame with text and label columns.
    """
    return pd.DataFrame(
        [
            {
                "text": item["text"],
                "label": "correct" if item["label"] == 1 else "incorrect",
            }
            for item in data
        ]
    )


def _create_multi_df(data: list[dict]) -> pd.DataFrame:
    """Create multi-class LA dataset (error types).

    Args:
        data:
            List of records from the JSON file.

    Returns:
        DataFrame with text and label columns.
    """
    return pd.DataFrame(
        [
            {"text": item["text"], "label": str(item.get("error_type", "correct"))}
            for item in data
        ]
    )


if __name__ == "__main__":
    main()
