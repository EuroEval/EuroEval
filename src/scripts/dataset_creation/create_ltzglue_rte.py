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

"""Create the ltzGLUE RTE dataset and upload to HF Hub."""

import io
import logging

import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_URL = "https://raw.githubusercontent.com/plumaj/ltzGLUE/main/data/rte"


def main() -> None:
    """Create the ltzGLUE-RTE dataset and upload to HF Hub."""
    logger.info("Downloading ltzGLUE-RTE data from GitHub...")
    train_df = _download_split("train")
    val_df = _download_split("dev")
    test_df = _download_split("test")

    logger.info(
        f"Downloaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test"
    )

    final_train, final_val, final_test = _create_splits(train_df, val_df, test_df)

    logger.info(
        f"Created splits: {len(final_train)} train, {len(final_val)} val, "
        f"{len(final_test)} test"
    )

    # Create 'text' column combining premise and hypothesis (required by EuroEval)
    for df in [final_train, final_val, final_test]:
        df["text"] = df.apply(
            lambda row: f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}",
            axis=1,
        )

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(final_train[["text", "label"]]),
            "val": Dataset.from_pandas(final_val[["text", "label"]]),
            "test": Dataset.from_pandas(final_test[["text", "label"]]),
        }
    )

    dataset_id = "EuroEval/ltzglue-rte"
    logger.info(f"Uploading to {dataset_id}...")

    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)
    logger.info(f"✓ Uploaded {dataset_id}")


def _download_split(split: str) -> pd.DataFrame:
    """Download a single split from GitHub.

    Args:
        split:
            Split name (train, dev, test).

    Returns:
        DataFrame with premise, hypothesis, and label columns.
    """
    url = f"{BASE_URL}/{split}.tsv"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), sep="\t")
    label_map = {0: "contradiction", 1: "entailment"}
    df["label"] = df["label"].map(label_map)
    df = df.rename(columns={"sentence1": "premise", "sentence2": "hypothesis"})
    return df[["premise", "hypothesis", "label"]]


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

    train_data, temp = train_test_split(
        all_data, train_size=n_train, random_state=42, stratify=all_data["label"]
    )
    val_data, test_data = train_test_split(
        temp, train_size=n_val / len(temp), random_state=42, stratify=temp["label"]
    )

    for df in [train_data, val_data, test_data]:
        df.reset_index(drop=True, inplace=True)

    return train_data, val_data, test_data


if __name__ == "__main__":
    main()
