# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
# ]
# ///

"""Create the WiC dataset and upload it to the HF Hub.

The Word in Context (WiC) dataset tests the ability to distinguish word
meanings/senses in context. Given two sentences with the same target word, the task is
to determine whether the word is used with the same sense or a different sense.

The dataset is based on the SuperGLUE WiC task:
  https://super.gluebenchmark.com/tasks
  https://aclanthology.org/N19-1128/
"""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi

# Split sizes
TRAIN_SIZE = 1024
VAL_SIZE = 256
TEST_SIZE = 2048

assert TRAIN_SIZE % 2 == 0, "TRAIN_SIZE must be even to allow per-class balancing."
assert VAL_SIZE % 2 == 0, "VAL_SIZE must be even to allow per-class balancing."
assert TEST_SIZE % 2 == 0, "TEST_SIZE must be even to allow per-class balancing."


def main() -> None:
    """Create the WiC dataset and upload it to the HF Hub."""
    raw_train = load_dataset("super_glue", "wic", split="train")
    raw_val = load_dataset("super_glue", "wic", split="validation")

    train_df = raw_train.to_pandas()
    val_df = raw_val.to_pandas()

    # Combine train and validation (both have labels); test split has no labels
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    combined_df = process_dataframe(df=combined_df)

    train_split, val_split, test_split = make_splits(df=combined_df)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_split, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_split, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_split, split=Split.TEST),
        }
    )

    dataset_id = "EuroEval/wic"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process the raw WiC dataframe into the benchmark format.

    Combines the target word and two context sentences into a single ``text`` column
    structured as::

        Word: {word}
        Context 1: {sentence1}
        Context 2: {sentence2}

    Args:
        df:
            The raw dataframe from the SuperGLUE WiC dataset.

    Returns:
        A dataframe with ``text`` and ``label`` columns.
    """
    df = df.copy()

    df["text"] = (
        "Word: "
        + df["word"].str.strip().astype(str)
        + "\nContext 1: "
        + df["sentence1"].str.strip().astype(str)
        + "\nContext 2: "
        + df["sentence2"].str.strip().astype(str)
    )

    # Map labels: 1 → same_sense, 0 → different_sense
    df["label"] = df["label"].map({1: "same_sense", 0: "different_sense"})

    df = df[["text", "label"]].copy()
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def make_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create balanced train / val / test splits.

    Each split has an equal number of ``same_sense`` and ``different_sense`` samples.

    Args:
        df:
            The full processed dataframe.

    Returns:
        A tuple of (train_df, val_df, test_df).
    """
    same_df = df[df["label"] == "same_sense"].reset_index(drop=True)
    diff_df = df[df["label"] == "different_sense"].reset_index(drop=True)

    train_per_class = TRAIN_SIZE // 2
    val_per_class = VAL_SIZE // 2
    test_per_class = TEST_SIZE // 2

    same_train = same_df.sample(n=train_per_class, random_state=4242)
    diff_train = diff_df.sample(n=train_per_class, random_state=4242)

    same_remaining = same_df.drop(same_train.index)
    diff_remaining = diff_df.drop(diff_train.index)

    same_val = same_remaining.sample(n=val_per_class, random_state=4242)
    diff_val = diff_remaining.sample(n=val_per_class, random_state=4242)

    same_remaining = same_remaining.drop(same_val.index)
    diff_remaining = diff_remaining.drop(diff_val.index)

    same_test = same_remaining.sample(n=test_per_class, random_state=4242)
    diff_test = diff_remaining.sample(n=test_per_class, random_state=4242)

    train_df = (
        pd.concat([same_train, diff_train])
        .sample(frac=1.0, random_state=4242)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat([same_val, diff_val])
        .sample(frac=1.0, random_state=4242)
        .reset_index(drop=True)
    )
    test_df = (
        pd.concat([same_test, diff_test])
        .sample(frac=1.0, random_state=4242)
        .reset_index(drop=True)
    )

    return train_df, val_df, test_df


if __name__ == "__main__":
    main()
