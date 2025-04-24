# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create a dataset from the Icelandic Linguistic Benchmarks."""

from ast import literal_eval

import pandas as pd
import requests
from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from requests.exceptions import HTTPError


def main() -> None:
    """Create a dataset from the Icelandic Linguistic Benchmarks."""
    dataset_url = (
        "https://raw.githubusercontent.com/stofnun-arna-magnussonar/"
        "ice_linguistic_benchmarks/refs/heads/main/ice_benchmark_set.jsonl"
    )

    # Download the dataset and convert it to a dataframe
    response = requests.get(dataset_url)
    lines = [literal_eval(line) for line in response.text.split("\n") if line.strip()]
    df = pd.DataFrame(lines)

    train_df = prepare_dataframe(df=df)

    # Validation split
    val_size = 32
    val_df = train_df.sample(n=val_size, random_state=4242)
    train_df = train_df.drop(val_df.index.tolist())

    # Test split
    test_size = 256
    test_df = train_df.sample(n=test_size, random_state=4242)
    train_df = train_df.drop(test_df.index.tolist())

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
    dataset_id = "EuroEval/ice-linguistic"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a dataframe from a dataset.

    Args:
        df:
            The dataframe to prepare

    Returns:
        A dataframe with the prepared dataset.
    """
    # Reset the index of the dataframe
    df.reset_index(drop=True, inplace=True)

    # Only extract rows where the task is to determine if a sentence is grammatically
    # correct/incorrect.
    df = df.loc[
        df["input"].str.contains("Is the following Icelandic sentence grammatically")
    ]

    # extract the text between \n\n TEXT \n\n and strip the text
    df["text"] = df["input"].str.extract(r"\n\n(.*)\n\n")
    df["text"] = df["text"].str.strip()

    # Remove samples with too few tokens
    df = df.loc[df["text"].str.split().map(len) > 5]

    def make_label(row: pd.Series) -> str:
        # asking if the sentence is incorrect.
        if "incorrect" in row["input"]:
            return "incorrect" if row["answer"] == "Yes" else "incorrect"

        # ...asking if the sentence is correct
        return "correct" if row["answer"] == "Yes" else "incorrect"

    df["label"] = df.apply(make_label, axis=1)

    # Shuffle the dataframe
    df = df.sample(frac=1.0, random_state=4242).reset_index(drop=True)

    return df


if __name__ == "__main__":
    main()
