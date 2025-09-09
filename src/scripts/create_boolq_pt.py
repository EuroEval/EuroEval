# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "numpy==1.23.2",
#     "pandas==2.2.0",
# ]
# ///

"""Create the BoolQ-pt dataset and upload to HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi

RANDOM_STATE = 4242
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 1024, 256, 2048
ORIGINAL_REPO_ID = "PORTULAN/extraglue"
FINAL_REPO_ID = "EuroEval/boolq-pt"


def main() -> None:
    """Create the dataset and upload to HF Hub."""
    ds_raw = load_dataset(ORIGINAL_REPO_ID, name="boolq_pt-PT")

    # Combine all splits to avoid duplicates across splits
    all_data = []
    for split in ["train", "test", "validation"]:
        df = ds_raw[split].to_pandas()
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Remove duplicates based on passage + question combination
    combined_df = combined_df.drop_duplicates(
        subset=["passage", "question"], keep="first"
    )

    # Shuffle the data
    combined_df = combined_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )

    # Calculate total needed samples
    total_needed = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    if len(combined_df) < total_needed:
        raise ValueError(
            f"Not enough unique samples. Available: {len(combined_df)}, "
            f"needed: {total_needed}"
        )

    # Split the data
    train_df = combined_df[:TRAIN_SIZE]
    val_df = combined_df[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
    test_df = combined_df[TRAIN_SIZE + VAL_SIZE : TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

    # Transform datasets
    train_df = transform_dataset(train_df)
    test_df = transform_dataset(test_df)
    val_df = transform_dataset(val_df)

    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    assert not set(dataset["train"]["text"]) & set(dataset["val"]["text"])
    assert not set(dataset["train"]["text"]) & set(dataset["test"]["text"])
    assert not set(dataset["val"]["text"]) & set(dataset["test"]["text"])

    HfApi().delete_repo(FINAL_REPO_ID, repo_type="dataset", missing_ok=True)

    dataset.push_to_hub(FINAL_REPO_ID, private=True)


def transform_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Transform dataset to multiple choice format.

    Args:
        df: DataFrame with columns 'passage', 'question', and 'label'.

    Returns:
        DataFrame with columns 'text' and 'label'.
    """
    texts = []
    labels = []

    for _, row in df.iterrows():
        text = (
            f"Texto: {row['passage']}\n"
            f"Pergunta: {row['question']}\n"
            f"Opções:\n"
            f"a. sim\n"
            f"b. não"
        )

        # Label 1 (yes) -> "a", Label 0 (no) -> "b"
        correct_label = "a" if row["label"] == 1 else "b"

        texts.append(text)
        labels.append(correct_label)

    return pd.DataFrame({"text": texts, "label": labels})


if __name__ == "__main__":
    main()
