# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "scikit-learn==1.6.1",
# ]
# ///

"""Create the ltzGLUE RTE (Recognising Textual Entailment) dataset and upload to HF Hub."""

import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


def load_rte_split(file_path: Path) -> pd.DataFrame:
    """Load RTE data with premise/hypothesis pairs."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        rows.append({
            "premise": item.get("premise", item.get("sentence1", "")),
            "hypothesis": item.get("hypothesis", item.get("sentence2", "")),
            "label": str(item.get("label", item.get("gold_label", ""))),
        })

    return pd.DataFrame(rows)


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
    """Create the ltzGLUE-RTE dataset."""
    ltzglue_root = Path(__file__).parent.parent.parent / "ltzGLUE"
    data_dir = ltzglue_root / "data" / "rte"

    train_df = load_rte_split(data_dir / "train.json")
    val_df = load_rte_split(data_dir / "dev.json")
    test_df = load_rte_split(data_dir / "test.json")

    print(f"Loaded RTE: {len(train_df)} train, {len(val_df)} dev, {len(test_df)} test")

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    final_train, final_val, final_test = make_splits(combined)

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
