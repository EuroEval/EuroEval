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

"""Create the ltzGLUE TC dataset and upload to HF Hub."""

import json

import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


BASE_URL = "https://raw.githubusercontent.com/plumaj/ltzGLUE/main/data/tc"


def download_split(split: str) -> list[dict]:
    """Download a single split from GitHub."""
    url = f"{BASE_URL}/{split}.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def load_tc_split(data: list[dict]) -> pd.DataFrame:
    """Load topic classification data."""
    return pd.DataFrame([
        {"text": f"{item['title']}: {item['text']}", "label": str(item["category_name"])}
        for item in data
    ])


def make_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits with stratification."""
    n = len(df)
    n_train = min(1024, int(n * 0.5))
    n_val = min(256, int(n * 0.15))

    train, temp = train_test_split(
        df, train_size=n_train, random_state=42, stratify=df["label"]
    )
    val, test = train_test_split(
        temp, train_size=n_val / len(temp), random_state=42, stratify=temp["label"]
    )

    for d in [train, val, test]:
        d.reset_index(drop=True, inplace=True)
    return train, val, test


def main() -> None:
    """Create the ltzGLUE-TC dataset."""
    print("Downloading ltzGLUE-TC data from GitHub...")
    train_data = download_split("train")
    val_data = download_split("dev")
    test_data = download_split("test")

    print(f"Downloaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    train_df = load_tc_split(train_data)
    val_df = load_tc_split(val_data)
    test_df = load_tc_split(test_data)

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    final_train, final_val, final_test = make_splits(combined)

    print(f"Created splits: {len(final_train)} train, {len(final_val)} val, {len(final_test)} test")

    dataset = DatasetDict({
        "train": Dataset.from_pandas(final_train[["text", "label"]]),
        "val": Dataset.from_pandas(final_val[["text", "label"]]),
        "test": Dataset.from_pandas(final_test[["text", "label"]]),
    })

    dataset_id = "EuroEval/ltzglue-tc"
    print(f"Uploading to {dataset_id}...")

    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)
    print(f"✓ Uploaded {dataset_id}")


if __name__ == "__main__":
    main()
