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

# Split sizes for balanced train and validation sets
TRAIN_SIZE = 128
VAL_SIZE = 64


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

    train_df = process_dataframe(df=train_df)
    val_df = process_dataframe(df=val_df)
    test_df = process_dataframe(df=test_df)

    train_df = sample_balanced(df=train_df, n=TRAIN_SIZE, random_state=4242)
    val_df = sample_balanced(df=val_df, n=VAL_SIZE, random_state=4242)

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


def sample_balanced(
    df: pd.DataFrame, n: int, random_state: int = 4242
) -> pd.DataFrame:
    """Sample a balanced subset of the dataframe.

    Selects an equal number of ``same_sense`` and ``different_sense`` samples.

    Args:
        df:
            The dataframe to sample from.
        n:
            The total number of samples to select. Must be even.
        random_state:
            The random state to use for sampling.

    Returns:
        A balanced dataframe with ``n`` samples.
    """
    assert n % 2 == 0, "n must be even for balanced sampling"
    per_class = n // 2

    same_df = df[df["label"] == "same_sense"]
    diff_df = df[df["label"] == "different_sense"]

    same_sample = same_df.sample(n=per_class, random_state=random_state)
    diff_sample = diff_df.sample(n=per_class, random_state=random_state)

    return (
        pd.concat([same_sample, diff_sample])
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    main()
