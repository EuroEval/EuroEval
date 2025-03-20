# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the SweReC-mini sentiment dataset and upload it to the HF Hub."""

import io

import pandas as pd
import requests
from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa
from datasets import Split
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from huggingface_hub.hf_api import HfApi
from requests.exceptions import HTTPError
from sklearn.model_selection import train_test_split


def main() -> None:
    """Create the SweReC-mini sentiment dataset and upload it to the HF Hub."""
    # Define the base download URL
    url = (
        "https://github.com/stoffesvensson/DeepLearning-ThesisWork-Convolutional/"
        "raw/master/webscraper/datasetSameAmountPosNeg"
    )

    # Download the dataset
    csv_str = requests.get(url).text.replace("\r", "")
    csv_file = io.StringIO(csv_str)

    # Convert the dataset to a dataframe
    df = pd.read_csv(csv_file, sep=",", usecols=["text", "rating"])
    df.columns = ["text", "label"]

    # Strip trailing whitespace
    df.text = df.text.str.strip()

    # Capitalize the first letter of each sentence
    df.text = df.text.str.capitalize()

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Create new splits
    full_train_df, val_test_df = train_test_split(
        df, test_size=2048 + 256, random_state=4242, stratify=df.label
    )
    val_df, test_df = train_test_split(
        val_test_df, test_size=2048, random_state=4242, stratify=val_test_df.label
    )
    _, train_df = train_test_split(
        full_train_df, test_size=1024, random_state=4242, stratify=full_train_df.label
    )

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Only work with samples where the document is not very large or small
    # We do it after we have made the splits to ensure that the dataset is minimally
    # affected.
    new_train_df = train_df.copy()
    new_train_df["text_len"] = new_train_df.text.str.len()
    new_train_df = new_train_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )
    new_val_df = val_df.copy()
    new_val_df["text_len"] = new_val_df.text.str.len()
    new_val_df = new_val_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )
    new_test_df = test_df.copy()
    new_test_df["text_len"] = new_test_df.text.str.len()
    new_test_df = new_test_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )

    df["text_len"] = df.text.str.len()

    # Add train samples back in, if we removed any
    num_new_train_samples = len(train_df) - len(new_train_df)
    if num_new_train_samples > 0:
        new_samples = (
            df.query("text not in @train_df.text")
            .query("text not in @val_df.text")
            .query("text not in @test_df.text")
            .query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT")
            .query("text_len <= @MAX_NUM_CHARS_IN_DOCUMENT")
            .sample(num_new_train_samples, random_state=4242)
            .drop(columns=["text_len"])
        )
        train_df = (
            pd.concat([new_train_df, new_samples], ignore_index=True)
            .drop(columns=["text_len"])
            .reset_index(drop=True)
        )

    # Add validation samples back in, if we removed any
    num_new_val_samples = len(val_df) - len(new_val_df)
    if num_new_val_samples > 0:
        new_samples = (
            df.query("text not in @train_df.text")
            .query("text not in @val_df.text")
            .query("text not in @test_df.text")
            .query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT")
            .query("text_len <= @MAX_NUM_CHARS_IN_DOCUMENT")
            .sample(num_new_val_samples, random_state=4242)
            .drop(columns=["text_len"])
        )
        val_df = (
            pd.concat([new_val_df, new_samples], ignore_index=True)
            .drop(columns=["text_len"])
            .reset_index(drop=True)
        )

    # Add test samples back in, if we removed any
    num_new_test_samples = len(test_df) - len(new_test_df)
    if num_new_test_samples > 0:
        new_samples = (
            df.query("text not in @train_df.text")
            .query("text not in @val_df.text")
            .query("text not in @test_df.text")
            .query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT")
            .query("text_len <= @MAX_NUM_CHARS_IN_DOCUMENT")
            .sample(num_new_test_samples, random_state=4242)
            .drop(columns=["text_len"])
        )
        test_df = (
            pd.concat([new_test_df, new_samples], ignore_index=True)
            .drop(columns=["text_len"])
            .reset_index(drop=True)
        )

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/swerec-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
