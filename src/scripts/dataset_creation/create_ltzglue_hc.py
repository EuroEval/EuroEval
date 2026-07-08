# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "scikit-learn==1.6.1",
# ]
# ///

"""Create the ltzGLUE Headline Classification (HC) dataset and upload to HF Hub.

This script processes the Luxembourgish headline classification data from the ltzGLUE
benchmark and uploads it to the EuroEval organization on Hugging Face.
"""

import json
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


def load_ltzglue_split(file_path: Path) -> pd.DataFrame:
    """Load a single ltzGLUE JSON split file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to dataframe with text and label columns
    df = pd.DataFrame([
        {"text": item["text"], "label": str(item["label"])}
        for item in data
    ])
    return df


def process_and_create_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_train: int = 1024,
    target_val: int = 256,
    target_test: int = 2048,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create standardized train/val/test splits."""
    # Combine all data for proper stratified sampling
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Sample target sizes with stratification
    sampled_train, temp = train_test_split(
        combined,
        train_size=target_train,
        random_state=42,
        stratify=combined["label"],
    )

    # Calculate remaining test size
    remaining_test = min(target_test, len(temp))
    sampled_test, remaining = train_test_split(
        temp,
        train_size=remaining_test,
        random_state=42,
        stratify=temp["label"],
    )

    # Use remaining for val (or sample if too large)
    sampled_val = remaining.sample(
        n=min(target_val, len(remaining)),
        random_state=42,
    )

    # Remove split column and reset indices
    for df in [sampled_train, sampled_val, sampled_test]:
        df.drop(columns=["split"], inplace=True, errors="ignore")
        df.reset_index(drop=True, inplace=True)

    return sampled_train, sampled_val, sampled_test


def main() -> None:
    """Create the ltzGLUE-HC dataset and upload to Hugging Face Hub."""
    # Define paths
    ltzglue_root = Path(__file__).parent.parent.parent / "ltzGLUE"
    data_dir = ltzglue_root / "data" / "hc"

    # Load splits from ltzGLUE
    train_df = load_ltzglue_split(data_dir / "train.json")
    val_df = load_ltzglue_split(data_dir / "dev.json")
    test_df = load_ltzglue_split(data_dir / "test.json")

    print(f"Loaded ltzGLUE-HC:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Dev: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Create standardized splits
    final_train, final_val, final_test = process_and_create_splits(
        train_df, val_df, test_df
    )

    print(f"\nCreated standardized splits:")
    print(f"  Train: {len(final_train)} samples")
    print(f"  Val: {len(final_val)} samples")
    print(f"  Test: {len(final_test)} samples")

    # Create DatasetDict
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(final_train),
            "val": Dataset.from_pandas(final_val),
            "test": Dataset.from_pandas(final_test),
        }
    )

    # Upload to Hugging Face
    dataset_id = "EuroEval/ltzglue-hc"
    print(f"\nUploading to {dataset_id}...")

    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset_dict.push_to_hub(dataset_id, private=True)

    print(f"✓ Successfully uploaded {dataset_id}")


if __name__ == "__main__":
    main()
