# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "scikit-learn==1.6.1",
# ]
# ///

"""Create the DanWiC dataset and upload it to the HF Hub.

The Danish Word in Context dataset (DanWiC) tests the ability to distinguish word
meanings/senses in context. Given two sentences with the same target word, the task is
to determine whether the word is used with the same sense or a different sense.
"""

import io
import logging
import zipfile

import pandas as pd
import requests
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_danwic")


# The DanWiC dataset is stored in a password-protected zip archive
DANWIC_URL = (
    "https://raw.githubusercontent.com/kuhumcst/"
    "danish-semantic-reasoning-benchmark/main/danwic/danwic.zip"
)
ZIP_PASSWORD = "benchmark"

# Split sizes
TRAIN_SIZE = 1024
VAL_SIZE = 256


def main() -> None:
    """Create the DanWiC dataset and upload it to the HF Hub."""
    df = download_and_load_dataset()
    df = process_dataframe(df=df)
    train_df, val_df, test_df = make_splits(df=df)
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )
    logger.info(
        f"Created dataset with {len(dataset['train'])} train samples, "
        f"{len(dataset['val'])} validation samples and {len(dataset['test'])} "
        f"test samples."
    )
    dataset_id = "EuroEval/danwic"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


def download_and_load_dataset() -> pd.DataFrame:
    """Download the DanWiC zip and return the dataframe.

    Returns:
        The dataframe with both same_sense and different_sense labels.
    """
    logger.info(f"Downloading DanWiC dataset from {DANWIC_URL}...")
    response = requests.get(DANWIC_URL, timeout=30)
    response.raise_for_status()

    logger.info("Loading DanWiC dataset...")
    zip_bytes = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_bytes) as zf:
        zf.setpassword(ZIP_PASSWORD.encode())
        with zf.open("danwic_poly_selection.tsv") as f:
            poly_df = pd.read_csv(f, sep="\t")
            poly_df["type"] = "polysemous"
        with zf.open("danwic_mono_selection.tsv") as f:
            mono_df = pd.read_csv(f, sep="\t")
            mono_df["type"] = "monosemous"

    return pd.concat([poly_df, mono_df]).sample(frac=1.0).reset_index(drop=True)


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
    logger.info("Processing DanWiC dataframe...")
    df = df.copy()
    df["text"] = (
        "Ord: "
        + df["target"].str.strip().astype(str)
        + "\nKontekst 1: "
        + df["first_context"].str.strip().astype(str)
        + "\nKontekst 2: "
        + df["second_context"].str.strip().astype(str)
    )
    df = df[["text", "label", "type", "idx"]]
    return df.drop_duplicates().reset_index(drop=True)


def make_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create balanced train / val / test splits.

    Each split has an equal number of ``same_sense`` and ``different_sense`` samples.

    Args:
        df:
            The full processed dataframe.

    Returns:
        A tuple of (train_df, val_df, test_df).
    """
    logger.info("Making splits...")
    train_val, test = train_test_split(
        df, train_size=TRAIN_SIZE + VAL_SIZE, random_state=4242, stratify=df["label"]
    )
    train, val = train_test_split(
        train_val, test_size=VAL_SIZE, random_state=4242, stratify=train_val["label"]
    )

    train = train.sample(frac=1.0, random_state=4242).reset_index(drop=True)
    val = val.sample(frac=1.0, random_state=4242).reset_index(drop=True)
    test = test.sample(frac=1.0, random_state=4242).reset_index(drop=True)

    return train, val, test


if __name__ == "__main__":
    main()
