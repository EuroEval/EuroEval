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

import pandas as pd
import requests
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


BASE_URL = "https://media.githubusercontent.com/media/plumaj/ltzGLUE/main/data/rte"


def download_split(split: str) -> pd.DataFrame:
    """Download a single split from GitHub."""
    url = f"{BASE_URL}/{split}.tsv"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(pd.io.StringIO(response.text), sep="\t")
    label_map = {0: "entailment", 1: "contradiction"}
    df["label"] = df["label"].map(label_map)
    df = df.rename(columns={"sentence1": "premise", "sentence2": "hypothesis"})
    return df[["premise", "hypothesis", "label"]]


def create_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create standardized EuroEval splits (1024/256/2048)."""
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)

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
    """Create the ltzGLUE-RTE dataset and upload to HF Hub."""
    print("Downloading ltzGLUE-RTE data from GitHub...")
    train_df = download_split("train")
    val_df = download_split("dev")
    test_df = download_split("test")

    print(f"Downloaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    final_train, final_val, final_test = create_splits(train_df, val_df, test_df)

    print(f"Created splits: {len(final_train)} train, {len(final_val)} val, {len(final_test)} test")

    dataset = DatasetDict({
        "train": Dataset.from_pandas(final_train[["premise", "hypothesis", "label"]]),
        "val": Dataset.from_pandas(final_val[["premise", "hypothesis", "label"]]),
        "test": Dataset.from_pandas(final_test[["premise", "hypothesis", "label"]]),
    })

    dataset_id = "EuroEval/ltzglue-rte"
    print(f"Uploading to {dataset_id}...")

    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)
    print(f"✓ Uploaded {dataset_id}")


if __name__ == "__main__":
    main()
