# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "scikit-learn==1.6.1",
# ]
# ///

"""Create the ltzGLUE Intent Detection (ID) dataset and upload to HF Hub."""

import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


def load_id_split(file_path: Path) -> pd.DataFrame:
    """Load intent detection data."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame([
        {"text": item["text"], "label": str(item["label"])}
        for item in data
    ])
    return df


def make_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits.
    
    Note: Does not use stratification as ID has classes with very few samples.
    """
    n = len(df)
    n_train = min(1024, int(n * 0.5))
    n_val = min(256, int(n * 0.15))

    train, temp = train_test_split(
        df, train_size=n_train, random_state=42
    )
    val, test = train_test_split(
        temp, train_size=n_val / len(temp), random_state=42
    )

    for d in [train, val, test]:
        d.reset_index(drop=True, inplace=True)
    return train, val, test


def main() -> None:
    """Create the ltzGLUE-ID dataset.

    Note: ltzGLUE ID only provides lb.test.json and lb.valid.json.
    We combine them and create standard splits.
    """
    ltzglue_root = Path(__file__).parent.parent.parent / "ltzGLUE"
    data_dir = ltzglue_root / "data" / "id"

    test_df = load_id_split(data_dir / "lb.test.json")
    val_df = load_id_split(data_dir / "lb.valid.json")

    print(f"Loaded ID: {len(test_df)} test, {len(val_df)} val (no training data)")

    combined = pd.concat([val_df, test_df], ignore_index=True)
    final_train, final_val, final_test = make_splits(combined)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(final_train[["text", "label"]]),
        "val": Dataset.from_pandas(final_val[["text", "label"]]),
        "test": Dataset.from_pandas(final_test[["text", "label"]]),
    })

    dataset_id = "EuroEval/ltzglue-id"
    print(f"Uploading to {dataset_id}...")

    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)
    print(f"✓ Uploaded {dataset_id}")


if __name__ == "__main__":
    main()
