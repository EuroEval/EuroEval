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

"""Create the ltzGLUE LA dataset (binary and multi-class) and upload to HF Hub."""

import json

import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


BASE_URL_BIN = "https://raw.githubusercontent.com/plumaj/ltzGLUE/refs/heads/main/data/la/binary"
BASE_URL_MULTI = "https://raw.githubusercontent.com/plumaj/ltzGLUE/refs/heads/main/data/la/multi"


def download_split(base_url: str, split: str) -> list[dict]:
    """Download a single split from GitHub."""
    url = f"{base_url}/{split}.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def create_binary_df(data: list[dict]) -> pd.DataFrame:
    """Create binary LA dataset (correct/incorrect)."""
    return pd.DataFrame([
        {"text": item["text"], "label": "correct" if item["label"] == 1 else "incorrect"}
        for item in data
    ])


def create_multi_df(data: list[dict]) -> pd.DataFrame:
    """Create multi-class LA dataset (error types)."""
    return pd.DataFrame([
        {"text": item["text"], "label": str(item.get("error_type", "correct"))}
        for item in data
    ])


def create_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create standardized EuroEval splits (1024/256/2048)."""
    n = len(df)
    n_train = min(1024, int(n * 0.5))
    n_val = min(256, int(n * 0.15))

    train, temp = train_test_split(df, train_size=n_train, random_state=42, stratify=df["label"])
    val, test = train_test_split(temp, train_size=n_val / len(temp), random_state=42, stratify=temp["label"])

    for d in [train, val, test]:
        d.reset_index(drop=True, inplace=True)
    return train, val, test


def main() -> None:
    """Create both LA binary and multi-class datasets."""
    # Binary version
    print("Downloading ltzGLUE-LA binary data from GitHub...")
    bin_train = download_split(BASE_URL_BIN, "train")
    bin_val = download_split(BASE_URL_BIN, "dev")
    bin_test = download_split(BASE_URL_BIN, "test")

    print(f"Downloaded binary: {len(bin_train)} train, {len(bin_val)} val, {len(bin_test)} test")

    binary_df = pd.concat([
        create_binary_df(bin_train),
        create_binary_df(bin_val),
        create_binary_df(bin_test),
    ], ignore_index=True)

    bin_train_split, bin_val_split, bin_test_split = create_splits(binary_df)

    print(f"Created binary splits: {len(bin_train_split)} / {len(bin_val_split)} / {len(bin_test_split)}")

    bin_dataset = DatasetDict({
        "train": Dataset.from_pandas(bin_train_split[["text", "label"]]),
        "val": Dataset.from_pandas(bin_val_split[["text", "label"]]),
        "test": Dataset.from_pandas(bin_test_split[["text", "label"]]),
    })

    HfApi().delete_repo("EuroEval/ltzglue-la", repo_type="dataset", missing_ok=True)
    bin_dataset.push_to_hub("EuroEval/ltzglue-la", private=True)
    print("✓ Uploaded EuroEval/ltzglue-la (binary)")

    # Multi-class version
    print("\nDownloading ltzGLUE-LA multi-class data from GitHub...")
    mul_train = download_split(BASE_URL_MULTI, "train")
    mul_val = download_split(BASE_URL_MULTI, "dev")
    mul_test = download_split(BASE_URL_MULTI, "test")

    print(f"Downloaded multi: {len(mul_train)} train, {len(mul_val)} val, {len(mul_test)} test")

    multi_df = pd.concat([
        create_multi_df(mul_train),
        create_multi_df(mul_val),
        create_multi_df(mul_test),
    ], ignore_index=True)

    mul_train_split, mul_val_split, mul_test_split = create_splits(multi_df)

    print(f"Created multi splits: {len(mul_train_split)} / {len(mul_val_split)} / {len(mul_test_split)}")

    mul_dataset = DatasetDict({
        "train": Dataset.from_pandas(mul_train_split[["text", "label"]]),
        "val": Dataset.from_pandas(mul_val_split[["text", "label"]]),
        "test": Dataset.from_pandas(mul_test_split[["text", "label"]]),
    })

    HfApi().delete_repo("EuroEval/ltzglue-la-multi", repo_type="dataset", missing_ok=True)
    mul_dataset.push_to_hub("EuroEval/ltzglue-la-multi", private=True)
    print("✓ Uploaded EuroEval/ltzglue-la-multi (multi-class)")


if __name__ == "__main__":
    main()
