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

# Split sizes
TRAIN_SIZE = 1024
VAL_SIZE = 256


def main() -> None:
    """Create the WiC-ITA dataset and upload it to the HF Hub."""
    raw_dataset = load_dataset(path="evalitahf/word_in_context")
    assert isinstance(raw_dataset, DatasetDict)

    orig_train_df = raw_dataset["train"].to_pandas()
    orig_val_df = raw_dataset["dev"].to_pandas()
    orig_test_df = raw_dataset["test"].to_pandas()

    assert isinstance(orig_train_df, pd.DataFrame)
    assert isinstance(orig_val_df, pd.DataFrame)
    assert isinstance(orig_test_df, pd.DataFrame)

    orig_train_df = process_dataframe(df=orig_train_df)
    orig_val_df = process_dataframe(df=orig_val_df)
    orig_test_df = process_dataframe(df=orig_test_df)

    # Train and val are sampled from the original training split (stratified on label)
    train_df, val_df = make_train_val_splits(df=orig_train_df)

    # Test is the concatenation of the original dev and test splits
    test_df = (
        pd.concat([orig_val_df, orig_test_df])
        .drop_duplicates()
        .sample(frac=1.0, random_state=4242)
        .reset_index(drop=True)
    )

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


def make_train_val_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create balanced train and val splits from the original training data.

    Each split has an equal number of ``same_sense`` and ``different_sense`` samples,
    stratified so that train is sampled first and val from the remainder.

    Args:
        df:
            The processed original training dataframe.

    Returns:
        A tuple of (train_df, val_df).
    """
    same_df = df[df["label"] == "same_sense"].reset_index(drop=True)
    diff_df = df[df["label"] == "different_sense"].reset_index(drop=True)

    train_per_class = TRAIN_SIZE // 2
    val_per_class = VAL_SIZE // 2

    same_train = same_df.sample(n=train_per_class, random_state=4242)
    diff_train = diff_df.sample(n=train_per_class, random_state=4242)

    same_remaining = same_df.drop(same_train.index)
    diff_remaining = diff_df.drop(diff_train.index)

    same_val = same_remaining.sample(n=val_per_class, random_state=4242)
    diff_val = diff_remaining.sample(n=val_per_class, random_state=4242)

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

    return train_df, val_df


if __name__ == "__main__":
    main()
