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

import json
from pathlib import Path

import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


BASE_URL = "https://raw.githubusercontent.com/plumaj/ltzGLUE/main/data/hc"


def download_split(split: str) -> list[dict]:
    """Download a single split from GitHub."""
    url = f"{BASE_URL}/{split}.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def load_ltzglue_split(data: list[dict]) -> pd.DataFrame:
    """Load headline classification data."""
    return pd.DataFrame([
        {"text": f"{item['text_long']} | {item['title']}", "label": "yes" if item["is_correct"] == "True" else "no"}
        for item in data
    ])


def create_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create standardized EuroEval splits (1024/256/2048)."""
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_data = all_data.drop_duplicates(subset=["text"])

    n_train = min(1024, int(len(all_data) * 0.5))
    n_val = min(256, int(len(all_data) * 0.15))

    train_data, temp = train_test_split(
        all_data, train_size=n_train, random_state=42, stratify=all_data["label"]
    )
    val_data, test_data = train_test_split(
        temp,
        train_size=n_val / len(temp),
        random_state=42,
        stratify=temp["label"],
    )

    for df in [train_data, val_data, test_data]:
        df.reset_index(drop=True, inplace=True)

    return train_data, val_data, test_data


def main() -> None:
    """Create the ltzGLUE-HC dataset and upload to HF Hub."""
    # Download data from GitHub
    print("Downloading ltzGLUE-HC data from GitHub...")
    train_data = download_split("train")
    val_data = download_split("dev")
    test_data = download_split("test")

    print(f"Downloaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    train_df = load_ltzglue_split(train_data)
    val_df = load_ltzglue_split(val_data)
    test_df = load_ltzglue_split(test_data)

    # Create capped splits
    final_train, final_val, final_test = create_splits(train_df, val_df, test_df)

    print(f"Created splits: {len(final_train)} train, {len(final_val)} val, {len(final_test)} test")

    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(final_train[["text", "label"]]),
        "val": Dataset.from_pandas(final_val[["text", "label"]]),
        "test": Dataset.from_pandas(final_test[["text", "label"]]),
    })

    dataset_id = "EuroEval/ltzglue-hc"
    print(f"Uploading to {dataset_id}...")

    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset_dict.push_to_hub(dataset_id, private=True)
    print(f"✓ Uploaded {dataset_id}")


if __name__ == "__main__":
    main()
