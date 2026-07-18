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


def main() -> None:
    """Create the ltzGLUE-ID dataset.

    Note that ltzGLUE ID only provides lb.valid.json and lb.test.json.
    No training data is available. We preserve the source splits exactly,
    exposing lb.valid as "val" and lb.test as "test".
    """
    logger.info("Downloading ltzGLUE-ID data from GitHub...")
    val_data = _download_split("lb.valid")
    test_data = _download_split("lb.test")

    logger.info(
        f"Downloaded: {len(val_data)} val, {len(test_data)} test (no training data)"
    )

    val_df = _load_split(val_data)
    test_df = _load_split(test_data)

    dataset = DatasetDict(
        {
            "val": Dataset.from_pandas(val_df[["text", "label"]]),
            "test": Dataset.from_pandas(test_df[["text", "label"]]),
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
