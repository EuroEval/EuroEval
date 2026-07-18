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

"""Create the ltzGLUE Headline Classification dataset and upload to HF Hub."""

import logging

import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_URL = "https://media.githubusercontent.com/media/plumaj/ltzGLUE/main/data/hc"


def main() -> None:
    """Create the ltzGLUE-HC dataset and upload to HF Hub."""
    logger.info("Downloading ltzGLUE-HC data from GitHub...")
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

    # Cap splits to EuroEval standard sizes while preserving source boundaries
    train_df = _cap_split(train_df, 1024, stratify=True)
    val_df = _cap_split(val_df, 256, stratify=True)
    test_df = _cap_split(test_df, 2048, stratify=True)

    logger.info(
        f"Preserved splits (capped): {len(train_df)} train, {len(val_df)} val, "
        f"{len(test_df)} test"
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df[["text", "label"]]),
            "val": Dataset.from_pandas(val_df[["text", "label"]]),
            "test": Dataset.from_pandas(test_df[["text", "label"]]),
        }
    )

    dataset_id = "EuroEval/ltzglue-hc"
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
    """Load headline classification data.

    Args:
        data:
            List of records from the JSON file.

    Returns:
        DataFrame with text and label columns.
    """
    return pd.DataFrame(
        [
            {
                "text": f"{item['title']} | {item['text_long']}",
                "label": "yes" if item["is_correct"] == "True" else "no",
            }
            for item in data
        ]
    )


def _cap_split(df: pd.DataFrame, max_size: int, stratify: bool = False) -> pd.DataFrame:
    """Cap a split to max size while preserving label distribution.

    Args:
        df:
            DataFrame to cap.
        max_size:
            Maximum number of samples to keep.
        stratify:
            Whether to stratify by label when sampling.

    Returns:
        Capped DataFrame with reset index.
    """
    if len(df) <= max_size:
        return df.reset_index(drop=True)

    if stratify and "label" in df.columns:
        # Sample with stratification to preserve label distribution
        df, _ = train_test_split(
            df, train_size=max_size, stratify=df["label"], random_state=42
        )
    else:
        df = df.sample(n=max_size, random_state=42)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    main()
