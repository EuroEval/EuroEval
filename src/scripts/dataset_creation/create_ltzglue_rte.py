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

    # Cap splits to EuroEval standard sizes while preserving source boundaries
    final_train = _cap_split(train_df, 1024, stratify=True)
    final_val = _cap_split(val_df, 256, stratify=True)
    final_test = _cap_split(test_df, 2048, stratify=True)

    logger.info(
        f"Preserved splits (capped): {len(final_train)} train, {len(final_val)} val, "
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
