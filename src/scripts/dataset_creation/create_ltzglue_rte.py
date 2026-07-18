# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the ltzGLUE RTE dataset and upload to HF Hub."""

import io
import logging

import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_URL = "https://media.githubusercontent.com/media/plumaj/ltzGLUE/main/data/rte"


def main() -> None:
    """Create the ltzGLUE-RTE dataset and upload to HF Hub."""
    logger.info("Downloading ltzGLUE-RTE data from GitHub...")
    train_df = _download_split("train")
    val_df = _download_split("dev")
    test_df = _download_split("test")

    logger.info(
        f"Downloaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test"
    )

    logger.info(
        f"Preserved splits exactly: {len(train_df)} train, {len(val_df)} val, "
        f"{len(test_df)} test"
    )

    # Create 'text' column combining premise and hypothesis (required by EuroEval)
    for df in [train_df, val_df, test_df]:
        df["text"] = df.apply(
            lambda row: f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}",
            axis=1,
        )

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df[["text", "label"]]),
            "val": Dataset.from_pandas(val_df[["text", "label"]]),
            "test": Dataset.from_pandas(test_df[["text", "label"]]),
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


if __name__ == "__main__":
    main()
