# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "pyzipper==0.3.6",
#     "requests==2.32.3",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the DAMETA dataset and upload it to the HF Hub."""

import io

import pandas as pd
import pyzipper
import requests as rq
from constants import CHOICES_MAPPING
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

URL = (
    "https://raw.githubusercontent.com/kuhumcst/danish-semantic-reasoning-benchmark"
    "/main/metaphor_benchmark/DAMETA_shuffled.zip"
)
ZIP_PASSWORD = b"benchmark"


def main() -> None:
    """Create the DAMETA dataset and upload it to the HF Hub."""
    # Download the ZIP file
    response = rq.get(url=URL)
    response.raise_for_status()

    # Extract the TSV file from the password-protected ZIP
    with pyzipper.AESZipFile(file=io.BytesIO(initial_bytes=response.content)) as zf:
        tsv_file_names = [name for name in zf.namelist() if name.endswith(".tsv")]
        assert len(tsv_file_names) == 1, (
            f"Expected exactly one TSV file, found: {tsv_file_names}"
        )
        tsv_content = zf.read(tsv_file_names[0], pwd=ZIP_PASSWORD)

    # Load the TSV data into a dataframe
    df = pd.read_csv(filepath_or_buffer=io.BytesIO(tsv_content), delimiter="\t")

    # Build the `text` column combining the word, sentence and answer options
    df["text"] = [
        "Hvad er den korrekte fortolkning af ordet '"
        + row.word.replace("\n", " ").strip()
        + "' i følgende sætning?\n'"
        + row.sentence.replace("\n", " ").strip()
        + "'\n"
        + f"{CHOICES_MAPPING['da']}:\n"
        + "a. "
        + row.A.replace("\n", " ").strip()
        + "\n"
        + "b. "
        + row.B.replace("\n", " ").strip()
        + "\n"
        + "c. "
        + row.C.replace("\n", " ").strip()
        + "\n"
        + "d. "
        + row.D.replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]

    # Lowercase the label (A -> a, B -> b, etc.)
    df["label"] = df["label"].str.lower()

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create train split
    train_size = 64
    train_arr, remaining_arr = train_test_split(
        df, train_size=train_size, random_state=4242
    )
    train_df = pd.DataFrame(train_arr, columns=df.columns)
    remaining_df = pd.DataFrame(remaining_arr, columns=df.columns)

    # Create validation and test split
    val_size = 128
    val_arr, test_arr = train_test_split(
        remaining_df, train_size=val_size, random_state=4242
    )
    val_df = pd.DataFrame(val_arr, columns=df.columns)
    test_df = pd.DataFrame(test_arr, columns=df.columns)

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    # Create dataset ID
    dataset_id = "EuroEval/dameta"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
