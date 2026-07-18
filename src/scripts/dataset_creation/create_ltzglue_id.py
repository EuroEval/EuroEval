# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the ltzGLUE ID dataset and upload to HF Hub."""

import logging

import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_URL = "https://media.githubusercontent.com/media/plumaj/ltzGLUE/main/data/id"

# Capping limits for mini datasets
MAX_VAL = 256
MAX_TEST = 2048


def main() -> None:
    """Create the ltzGLUE-ID dataset.

    Note that ltzGLUE ID only provides lb.valid.json and lb.test.json.
    No training data is available. We cap the source splits independently
    (val to 256, test to 2048), exposing lb.valid as "val" and lb.test as "test".
    The published dataset ID ends with -mini to indicate capping.
    """
    logger.info("Downloading ltzGLUE-ID data from GitHub...")
    val_data = _download_split("lb.valid")
    test_data = _download_split("lb.test")

    logger.info(
        f"Downloaded: {len(val_data)} val, {len(test_data)} test (no training data)"
    )

    val_df = _load_split(val_data)
    test_df = _load_split(test_data)

    # Cap each split independently, preserving source split boundaries
    val_df = _cap_split(val_df, MAX_VAL)
    test_df = _cap_split(test_df, MAX_TEST)

    logger.info(
        f"Capped splits: {len(val_df)} val, {len(test_df)} test (no training data)"
    )

    dataset = DatasetDict(
        {
            "val": Dataset.from_pandas(val_df[["text", "label"]]),
            "test": Dataset.from_pandas(test_df[["text", "label"]]),
        }
    )

    dataset_id = "EuroEval/ltzglue-id-mini"
    logger.info(f"Uploading to {dataset_id}...")

    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)
    logger.info(f"✓ Uploaded {dataset_id}")


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

    Note:
        Labels are normalized to lowercase, with "/" and "_" replaced by spaces.
    """

    def normalize_label(label: str) -> str:
        """Normalize label: lowercase, replace "/" and "_" with spaces.

        Args:
            label:
                Original label string.

        Returns:
            Normalized label string.
        """
        return str(label).lower().replace("/", " ").replace("_", " ")

    return pd.DataFrame(
        [
            {"text": item["text"], "label": normalize_label(item["label"])}
            for item in data
        ]
    )


if __name__ == "__main__":
    main()
