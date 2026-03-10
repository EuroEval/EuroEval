# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
# ]
# ///

"""Create the WiC-ITA dataset and upload it to the HF Hub.

WiC-ITA is the Italian Word-in-Context task from Evalita 2023. Given two sentences
containing the same target word, the task is to determine whether the word carries
the same sense in both sentences.
"""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi

# Split sizes for balanced train, validation and test sets
TRAIN_SIZE = 1024
VAL_SIZE = 256
TEST_SIZE = 2048


def main() -> None:
    """Create the WiC-ITA dataset and upload it to the HF Hub."""
    raw_dataset = load_dataset(path="evalitahf/word_in_context")
    assert isinstance(raw_dataset, DatasetDict)

    train_df = raw_dataset["train"].to_pandas()
    val_df = raw_dataset["dev"].to_pandas()
    test_df = raw_dataset["test"].to_pandas()

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    full_df = (
        pd.concat(
            [
                process_dataframe(df=train_df),
                process_dataframe(df=val_df),
                process_dataframe(df=test_df),
            ]
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    train_df, val_df, test_df = make_splits(df=full_df)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    dataset_id = "EuroEval/wicita"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process the raw WiC-ITA dataframe into the benchmark format.

    Combines the target word and two context sentences into a single ``text`` column
    structured as::

        Parola: {lemma}
        Contesto 1: {sentence1}
        Contesto 2: {sentence2}

    Maps the integer labels (0/1) to string labels (different_sense/same_sense).

    Args:
        df:
            The raw dataframe from the HuggingFace dataset.

    Returns:
        A dataframe with ``text`` and ``label`` columns.
    """
    df = df.copy()

    df["text"] = (
        "Parola: "
        + df["lemma"].str.strip().astype(str)
        + "\nContesto 1: "
        + df["sentence1"].str.strip().astype(str)
        + "\nContesto 2: "
        + df["sentence2"].str.strip().astype(str)
    )

    df["label"] = df["label"].map({0: "different_sense", 1: "same_sense"})

    df = df[["text", "label"]].copy()
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def make_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create balanced train / val / test splits.

    Each split has an equal number of ``same_sense`` and ``different_sense`` samples.
    Samples are drawn without replacement: train is taken first, then val from the
    remainder, then test from what remains.

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
