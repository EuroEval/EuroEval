# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Icelandic Error Corpus dataset  and upload it to the HF Hub."""

import re
from typing import List

import pandas as pd
from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests.exceptions import HTTPError


def main() -> None:
    """Create the Icelandic Error Corpus dataset  and upload it to the HF Hub."""
    repo_id = "mideind/icelandic-error-corpus-IceEC"
    dataset = load_dataset(path=repo_id, name="category")
    assert isinstance(dataset, DatasetDict)

    train_df = prepare_dataframe(dataset=dataset["train"])
    test_df = prepare_dataframe(dataset=dataset["test"])

    # Make validation split
    val_size = 1024
    val_df = train_df.sample(n=val_size, random_state=4242)
    train_df = train_df.drop(val_df.index)

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
        test=Dataset.from_pandas(new_test_df, split=Split.TEST),
    )

    # Make subset of the dataset. We use `head` instead of `sample` here as the
    # dataframes have already been shuffled.
    dataset_subset = DatasetDict(
        train=Dataset.from_pandas(
            new_train_df.head(1024), split=Split.TRAIN, preserve_index=False
        ),
        val=Dataset.from_pandas(
            new_val_df.head(256), split=Split.VALIDATION, preserve_index=False
        ),
        test=Dataset.from_pandas(
            new_test_df.head(2048), split=Split.TEST, preserve_index=False
        ),
    )

    # Create dataset IDs
    dataset_id = "EuroEval/ice-ec-full"
    dataset_subset_id = "EuroEval/ice-ec"

    for dataset_, dataset_id_ in [
        (dataset, dataset_id),
        (dataset_subset, dataset_subset_id),
    ]:
        # Remove the dataset from Hugging Face Hub if it already exists
        try:
            api = HfApi()
            api.delete_repo(dataset_id_, repo_type="dataset")
        except HTTPError:
            pass

        # Push the dataset to the Hugging Face Hub
        dataset_.push_to_hub(dataset_id_, private=True)


def prepare_dataframe(dataset: Dataset) -> pd.DataFrame:
    """Prepare a dataframe from a dataset.

    Args:
        dataset:
            The dataset to prepare.

    Returns:
        A dataframe with the prepared dataset.
    """
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Reset the index of the dataframe
    df.reset_index(drop=True, inplace=True)

    # Remove samples with five or fewer tokens
    df = df.loc[df.sentence.map(lambda lst: len(lst) > 5)]

    # Join the tokens into a string
    df["text"] = df["sentence"].apply(join_tokens)

    # Make the `label` column` such that it is `já` if there are no errors and `nei` if
    # there are errors
    df["label"] = df.apply(
        lambda row: "correct" if not row["has_error"] else "incorrect", axis=1
    )

    # Keep only relevant features: text and label
    df = df[["text", "label"]]

    # Shuffle the dataframe
    df = df.sample(frac=1.0, random_state=4242).reset_index(drop=True)

    return df


def join_tokens(tokens: List[str]) -> str:
    """Joins a list of tokens into a string.

    Args:
        tokens:
            The list of tokens to join.

    Returns:
        The joined string.
    """
    # Form document
    doc = " ".join(tokens)

    # Remove whitespace around punctuation
    doc = (
        doc.replace(" .", ".")
        .replace(" ,", ",")
        .replace(" ;", ";")
        .replace(" :", ":")
        .replace("( ", "(")
        .replace(" )", ")")
        .replace("[ ", "[")
        .replace(" ]", "]")
        .replace("{ ", "{")
        .replace(" }", "}")
        .replace(" ?", "?")
        .replace(" !", "!")
    )

    # Remove whitespace around quotes
    if doc.count('"') % 2 == 0:
        doc = re.sub('" ([^"]*) "', '"\\1"', doc)

    # Return the document
    return doc


if __name__ == "__main__":
    main()
