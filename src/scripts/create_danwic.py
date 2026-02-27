# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the DanWiC dataset and upload it to the HF Hub.

The Danish Word in Context dataset (DanWiC) tests the ability to distinguish word
meanings/senses in context. Given two sentences with the same target word, the task is
to determine whether the word is used with the same sense or a different sense.
"""

import io
import zipfile

import pandas as pd
import requests
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi

# The DanWiC dataset is stored in a password-protected zip archive
DANWIC_URL = (
    "https://raw.githubusercontent.com/kuhumcst/"
    "danish-semantic-reasoning-benchmark/main/danwic/danwic.zip"
)
ZIP_PASSWORD = "benchmark"

# Split sizes
TRAIN_SIZE = 128
VAL_SIZE = 64


def main() -> None:
    """Create the DanWiC dataset and upload it to the HF Hub."""
    poly_df = download_and_load_poly()

    poly_df = process_dataframe(df=poly_df)

    train_df, val_df, test_df = make_splits(df=poly_df)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    dataset_id = "EuroEval/danwic"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


def download_and_load_poly() -> pd.DataFrame:
    """Download the DanWiC zip and return the polysemous selection dataframe.

    Returns:
        The polysemous selection dataframe with both same_sense and different_sense
        labels.
    """
    response = requests.get(DANWIC_URL, timeout=30)
    response.raise_for_status()

    zip_bytes = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_bytes) as zf:
        zf.setpassword(ZIP_PASSWORD.encode())
        with zf.open("danwic_poly_selection.tsv") as f:
            poly_df = pd.read_csv(f, sep="\t")

    return poly_df


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process the raw DanWiC dataframe into the benchmark format.

    Combines the target word and two context sentences into a single ``text`` column
    structured as::

        Ord: {target}
        Kontekst 1: {first_context}
        Kontekst 2: {second_context}

    Args:
        df:
            The raw dataframe from the TSV file.

    Returns:
        A dataframe with ``text`` and ``label`` columns.
    """
    df = df.copy()

    df["text"] = (
        "Ord: "
        + df["target"].str.strip()
        + "\nKontekst 1: "
        + df["first_context"].str.strip()
        + "\nKontekst 2: "
        + df["second_context"].str.strip()
    )

    df = df[["text", "label"]].copy()
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def make_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    same_train = same_df.sample(n=train_per_class, random_state=4242)
    diff_train = diff_df.sample(n=train_per_class, random_state=4242)

    same_remaining = same_df.drop(same_train.index)
    diff_remaining = diff_df.drop(diff_train.index)

    same_val = same_remaining.sample(n=val_per_class, random_state=4242)
    diff_val = diff_remaining.sample(n=val_per_class, random_state=4242)

    same_test = same_remaining.drop(same_val.index)
    diff_test = diff_remaining.drop(diff_val.index)

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
