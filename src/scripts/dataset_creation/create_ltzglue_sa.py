# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "scikit-learn==1.6.1",
# ]
# ///

"""Create the ltzGLUE Sentiment Analysis (SA) dataset and upload to HF Hub."""

import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


def load_ltzglue_split(file_path: Path) -> pd.DataFrame:
    """Load a single ltzGLUE JSON split file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame([
        {"text": item["text"], "label": str(item["label"])}
        for item in data
    ])
    return df


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
    len(all_data) - n_train - n_val

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
    """Create the ltzGLUE-SA dataset and upload to HF Hub."""
    ltzglue_root = Path(__file__).parent.parent.parent / "ltzGLUE"
    data_dir = ltzglue_root / "data" / "sa"

    train_df = load_ltzglue_split(data_dir / "train.json")
    val_df = load_ltzglue_split(data_dir / "dev.json")
    test_df = load_ltzglue_split(data_dir / "test.json")

    print(f"Loaded ltzGLUE-SA: {len(train_df)} train, {len(val_df)} dev, {len(test_df)} test")

    final_train, final_val, final_test = create_splits(train_df, val_df, test_df)

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(final_train[["text", "label"]]),
            "val": Dataset.from_pandas(final_val[["text", "label"]]),
            "test": Dataset.from_pandas(final_test[["text", "label"]]),
        }
    )

    dataset_id = "EuroEval/ltzglue-sa"
    print(f"Uploading to {dataset_id}...")

    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset_dict.push_to_hub(dataset_id, private=True)
    print(f"✓ Uploaded {dataset_id}")


if __name__ == "__main__":
    main()
