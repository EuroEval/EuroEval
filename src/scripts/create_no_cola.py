# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the NoCoLA linguistic acceptability dataset."""

import logging

import pandas as pd
import requests
from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from requests import HTTPError
from sklearn.model_selection import train_test_split

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_no_cola")


def main() -> None:
    """Create the Jentoft linguistic acceptability dataset and upload to HF Hub."""
    base_url = (
        "https://raw.githubusercontent.com/ltgoslo/nocola/refs/heads/main/datasets/"
        "NoCoLa_class_{}.txt"
    )
    dataset_urls = {
        split: base_url.format(url_split_name)
        for split, url_split_name in dict(train="train", val="dev", test="test").items()
    }

    # Build the dataframes
    train_df = build_dataframe(url=dataset_urls["train"])
    val_df = build_dataframe(url=dataset_urls["val"])
    test_df = build_dataframe(url=dataset_urls["test"])

    # Create the train split
    train_size = 1024
    train_correct = train_df.query("label == 'correct'").sample(
        n=train_size // 2, random_state=4242
    )
    train_incorrect = train_df.query("label == 'incorrect'")
    train_incorrect, _ = train_test_split(
        train_incorrect, train_size=train_size // 2, stratify=train_incorrect.category
    )
    assert isinstance(train_incorrect, pd.DataFrame)
    train_df = pd.concat([train_correct, train_incorrect], ignore_index=True)

    # Create the validation split
    val_size = 256
    val_correct = val_df.query("label == 'correct'").sample(
        n=val_size // 2, random_state=4242
    )
    val_incorrect = val_df.query("label == 'incorrect'")
    val_incorrect, _ = train_test_split(
        val_incorrect, train_size=val_size // 2, stratify=val_incorrect.category
    )
    assert isinstance(val_incorrect, pd.DataFrame)
    val_df = pd.concat([val_correct, val_incorrect], ignore_index=True)

    # Create the test split
    test_size = 2048
    test_correct = test_df.query("label == 'correct'").sample(
        n=test_size // 2, random_state=4242
    )
    test_incorrect = test_df.query("label == 'incorrect'")
    test_incorrect, _ = train_test_split(
        test_incorrect, train_size=test_size // 2, stratify=test_incorrect.category
    )
    assert isinstance(test_incorrect, pd.DataFrame)
    test_df = pd.concat([test_correct, test_incorrect], ignore_index=True)

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

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(
            new_train_df, split=Split.TRAIN, preserve_index=False
        ),
        val=Dataset.from_pandas(
            new_val_df, split=Split.VALIDATION, preserve_index=False
        ),
        test=Dataset.from_pandas(new_test_df, split=Split.TEST, preserve_index=False),
    )

    # Create dataset ID
    dataset_id = "EuroEval/no-cola-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api: HfApi = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def build_dataframe(url: str) -> pd.DataFrame:
    """Build a dataframe from a NoCoLa dataset URL.

    Args:
        url:
            The URL of the NoCoLa dataset.

    Returns:
        A dataframe containing the NoCoLa dataset.
    """
    # Get the raw data
    response = requests.get(url=url)
    response.raise_for_status()
    lines = response.text.split("\n")

    # Extract the data from the lines
    records: list[dict[str, str]] = list()
    for line in lines:
        # Skip header lines and blank lines
        if "Grammatical:" in line or "Ungrammatical:" in line or line.strip() == "":
            continue

        # Extract the text, binary label and error category from the lines. The correct
        # lines only contain the text, so we need to treat this case separately
        components = line.split("\t")
        text = components[0]
        if len(components) == 1:
            label = "correct"
            category = "correct"
        elif len(components) == 2:
            label = "incorrect"
            category = components[1]
        else:
            raise ValueError(f"Unexpected number of components in line: {line}")

        # Store the record
        records.append(dict(text=text, label=label, category=category))

    # Collect the records in a dataframe
    train_df = pd.DataFrame.from_records(records)

    return train_df


if __name__ == "__main__":
    main()
