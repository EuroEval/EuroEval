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

"""Create the Czech HellaSwag dataset and upload to HF Hub."""

from collections import Counter

import pandas as pd
from constants import (
    CHOICES_MAPPING,
    MAX_NUM_CHARS_IN_INSTRUCTION,
    MAX_NUM_CHARS_IN_OPTION,
    MAX_REPETITIONS,
    MIN_NUM_CHARS_IN_INSTRUCTION,
    MIN_NUM_CHARS_IN_OPTION,
)
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


def main() -> None:
    """Create the Czech HellaSwag dataset and upload to HF Hub."""
    dataset_id = "CZLC/cs_hellaswag"

    # Load the dataset
    dataset = load_dataset(dataset_id, split="test")
    assert isinstance(dataset, Dataset)

    # Convert the dataset to a pandas DataFrame
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Remove the samples with overly short or long texts
    df = df[
        (df["query"].str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)  # Check text
        & (df["query"].str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)  # Check text
        & df.choices.map(
            lambda choices: min(len(choice) for choice in choices)
            >= MIN_NUM_CHARS_IN_OPTION
            and max(len(choice) for choice in choices) <= MAX_NUM_CHARS_IN_OPTION
        )
    ]

    def is_repetitive(text: str) -> bool:
        """Return True if the text is repetitive."""
        max_repetitions = max(Counter(text.split()).values())
        return max_repetitions > MAX_REPETITIONS

    # Remove overly repetitive samples
    df = df[
        ~df["query"].apply(is_repetitive)
        & ~df.choices.map(
            lambda choices: any(is_repetitive(choice) for choice in choices)
        )
    ]

    # Make a `text` column with all the options in it
    df["text"] = [
        row.query.replace("\n", " ").strip() + "\n"
        f"{CHOICES_MAPPING['cs']}:\n"
        "a. " + row.choices[0].replace("\n", " ").strip() + "\n"
        "b. " + row.choices[1].replace("\n", " ").strip() + "\n"
        "c. " + row.choices[2].replace("\n", " ").strip() + "\n"
        "d. " + row.choices[3].replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]

    # Map the `gold` index to a letter for the `label` column
    df["label"] = df["gold"].apply(lambda x: chr(97 + x))

    # Only keep the columns `text` and `label`
    df = df[["text", "label"]]

    # Create train, validation, and test splits
    train_size = 1024
    val_size = 256
    test_size = 2048

    # First, split off the test set
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=4242)

    # Then, split the remaining data into train and validation sets
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=4242
    )

    # Ensure the train set is exactly 1024 samples
    train_df = train_df.sample(n=train_size, random_state=4242)

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

    # Define the dataset ID for the Hugging Face Hub
    dataset_id = "EuroEval/hellaswag-cs-mini"

    # Push the dataset to the Hugging Face Hub
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
