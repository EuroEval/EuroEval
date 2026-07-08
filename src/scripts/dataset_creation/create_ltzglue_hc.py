# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "json",
#     "pandas==2.2.0",
#     "scikit-learn==1.6.1",
# ]
# ///

"""Create the ltzGLUE Headline Classification dataset.

Processes Luxembourgish headline classification data from the ltzGLUE benchmark
and uploads it to the EuroEval organisation on Hugging Face.
"""

import json

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


def load_ltzglue_split(file_path: str) -> pd.DataFrame:
    """Load a single ltzGLUE JSON split file.

    Args:
        file_path: Path to the JSON file containing ltzGLUE data.

    Returns:
        DataFrame with text and label columns.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame([
        {"text": f"{item['text_long']} | {item['title']}", "label": "yes" if item["is_correct"] == "True" else "no"}
        for item in data
    ])
    return df


def create_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_size: int = 1024,
    val_size: int = 256,
    test_size: int = 2048,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create standardised train/val/test splits.

    Args:
        train_df: Original training data.
        val_df: Original validation data.
        test_df: Original test data.
        train_size: Target number of training samples.
        val_size: Target number of validation samples.
        test_size: Target number of test samples.

    Returns:
        Tuple of (train, val, test) DataFrames.
    """
    combined = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ], ignore_index=True)

    sampled_train, temp = train_test_split(
        combined,
        train_size=train_size,
        random_state=42,
        stratify=combined["label"],
    )

    remaining_test = min(test_size, len(temp))
    sampled_test, remaining = train_test_split(
        temp,
        train_size=remaining_test,
        random_state=42,
        stratify=temp["label"],
    )

    sampled_val = remaining.sample(
        n=min(val_size, len(remaining)),
        random_state=42,
    )

    for df in [sampled_train, sampled_val, sampled_test]:
        df.drop(columns=["split"], inplace=True, errors="ignore")
        df.reset_index(drop=True, inplace=True)

    return sampled_train, sampled_val, sampled_test


def main() -> None:
    """Create the ltzGLUE-HC dataset and upload to Hugging Face.

    Requires the ltzGLUE repository to be cloned with LFS data.
    """
    import os
    from pathlib import Path

    ltzglue_root = Path(os.environ.get("LTZGLUE_ROOT", "../ltzGLUE"))
    data_dir = ltzglue_root / "data" / "hc"

    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        print("Clone ltzGLUE with: git clone https://github.com/plumaj/ltzGLUE.git")
        print("Then: cd ltzGLUE && git lfs pull")
        return

    train_df = load_ltzglue_split(str(data_dir / "train.json"))
    val_df = load_ltzglue_split(str(data_dir / "dev.json"))
    test_df = load_ltzglue_split(str(data_dir / "test.json"))

    print("Loaded ltzGLUE-HC:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Dev: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    final_train, final_val, final_test = create_splits(train_df, val_df, test_df)

    print("\nCreated standardised splits:")
    print(f"  Train: {len(final_train)} samples")
    print(f"  Val: {len(final_val)} samples")
    print(f"  Test: {len(final_test)} samples")

    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(final_train[["text", "label"]]),
        "val": Dataset.from_pandas(final_val[["text", "label"]]),
        "test": Dataset.from_pandas(final_test[["text", "label"]]),
    })

    dataset_id = "EuroEval/ltzglue-hc"
    print(f"\nUploading to {dataset_id}...")

    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset_dict.push_to_hub(dataset_id, private=True)

    print(f"✓ Successfully uploaded {dataset_id}")


if __name__ == "__main__":
    main()
