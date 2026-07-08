# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "scikit-learn==1.6.1",
# ]
# ///

"""Create the ltzGLUE Linguistic Acceptability (LA) dataset and upload to HF Hub.

Creates both binary (correct/incorrect) and multi-class (error type) versions.
"""

import json
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


def load_ltzglue_split(file_path: Path) -> list[dict]:
    """Load a single ltzGLUE JSON split file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_binary_split(data: list[dict]) -> pd.DataFrame:
    """Create binary LA dataset (correct/incorrect)."""
    return pd.DataFrame([
        {"text": item["text"], "label": "correct" if item["label"] == 1 else "incorrect"}
        for item in data
    ])


def create_multi_split(data: list[dict]) -> pd.DataFrame:
    """Create multi-class LA dataset (error types)."""
    return pd.DataFrame([
        {"text": item["text"], "label": str(item.get("error_type", "correct"))}
        for item in data
    ])


def make_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits."""
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
    ltzglue_root = Path(__file__).parent.parent.parent / "ltzGLUE"
    data_dir = ltzglue_root / "data" / "la"

    train_data = load_ltzglue_split(data_dir / "train.json")
    val_data = load_ltzglue_split(data_dir / "dev.json")
    test_data = load_ltzglue_split(data_dir / "test.json")

    # Binary version
    print("Creating binary LA dataset...")
    binary_df = pd.concat([
        create_binary_split(train_data),
        create_binary_split(val_data),
        create_binary_split(test_data),
    ], ignore_index=True)
    bin_train, bin_val, bin_test = make_splits(binary_df)

    bin_dataset = DatasetDict({
        "train": Dataset.from_pandas(bin_train),
        "val": Dataset.from_pandas(bin_val),
        "test": Dataset.from_pandas(bin_test),
    })

    HfApi().delete_repo("EuroEval/ltzglue-lab", repo_type="dataset", missing_ok=True)
    bin_dataset.push_to_hub("EuroEval/ltzglue-lab", private=True)
    print(f"✓ Uploaded EuroEval/ltzglue-lab (binary)")

    # Multi-class version
    print("\nCreating multi-class LA dataset...")
    multi_df = pd.concat([
        create_multi_split(train_data),
        create_multi_split(val_data),
        create_multi_split(test_data),
    ], ignore_index=True)
    mul_train, mul_val, mul_test = make_splits(multi_df)

    mul_dataset = DatasetDict({
        "train": Dataset.from_pandas(mul_train),
        "val": Dataset.from_pandas(mul_val),
        "test": Dataset.from_pandas(mul_test),
    })

    HfApi().delete_repo("EuroEval/ltzglue-lam", repo_type="dataset", missing_ok=True)
    mul_dataset.push_to_hub("EuroEval/ltzglue-lam", private=True)
    print(f"✓ Uploaded EuroEval/ltzglue-lam (multi-class)")


if __name__ == "__main__":
    main()
