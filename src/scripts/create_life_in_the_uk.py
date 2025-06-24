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

"""Create the Life in the UK multiple choice dataset and upload it to the HF Hub."""

from collections import Counter

import pandas as pd
from constants import (
    MAX_NUM_CHARS_IN_INSTRUCTION,
    MAX_NUM_CHARS_IN_OPTION,
    MAX_REPETITIONS,
    MIN_NUM_CHARS_IN_INSTRUCTION,
    MIN_NUM_CHARS_IN_OPTION,
)
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError
from sklearn.model_selection import train_test_split


def main() -> None:
    """Create the Life in the UK multiple choice dataset and upload it to the HF Hub."""
    # Define the dataset ID
    repo_id = "oliverkinch/life-in-the-uk-multiple-choice"

    # Download the dataset
    dataset = load_dataset(path=repo_id)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    df = dataset["train"].to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Rename the columns to match EuroEval format
    df.rename(columns=dict(answer="label"), inplace=True)

    # Remove all samples with a null value in any of the option columns
    df = df[
        df["a"].notnull() & df["b"].notnull() & df["c"].notnull() & df["d"].notnull()
    ]

    # Remove the samples with overly short or long texts
    df = df[
        (df.question.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
        & (df.question.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
        & (df.a.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.a.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.b.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.b.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.c.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.c.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.d.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.d.str.len() <= MAX_NUM_CHARS_IN_OPTION)
    ]

    def is_repetitive(text: str) -> bool:
        """Return True if the text is repetitive."""
        max_repetitions = max(Counter(text.split()).values())
        return max_repetitions > MAX_REPETITIONS

    # Remove overly repetitive samples
    df = df[
        ~df.question.apply(is_repetitive)
        & ~df.a.apply(is_repetitive)
        & ~df.b.apply(is_repetitive)
        & ~df.c.apply(is_repetitive)
        & ~df.d.apply(is_repetitive)
    ]

    # Make a `text` column with all the options in it
    df["text"] = [
        row.question.replace("\n", " ").strip() + "\n"
        "Choices:\n"
        "a. " + row.a.replace("\n", " ").strip() + "\n"
        "b. " + row.b.replace("\n", " ").strip() + "\n"
        "c. " + row.c.replace("\n", " ").strip() + "\n"
        "d. " + row.d.replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]

    # Make the `label` column case-consistent with the `text` column
    df.label = df.label.str.lower()

    # Only keep the `text` and `label` columns
    df = df[["text", "label"]]

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create validation split
    val_size = 256
    traintest_arr, val_arr = train_test_split(df, test_size=val_size, random_state=4242)
    traintest_df = pd.DataFrame(traintest_arr, columns=df.columns)
    val_df = pd.DataFrame(val_arr, columns=df.columns)

    # Create test split
    test_size = 512
    train_arr, test_arr = train_test_split(
        traintest_df, test_size=test_size, random_state=4242
    )
    train_df = pd.DataFrame(train_arr, columns=df.columns)
    test_df = pd.DataFrame(test_arr, columns=df.columns)

    # Create train split
    train_size = 438
    train_df = train_df.sample(train_size, random_state=4242)

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/life-in-the-uk"

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
