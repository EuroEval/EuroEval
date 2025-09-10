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

"""Create the Swedish skolprov datasets and upload them to the HF Hub."""

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
    """Create the Swedish skolprov datasets and upload them to the HF Hub."""
    # Define the base download URL
    repo_id = "Ekgren/swedish_skolprov"

    # Load the dataset
    try:
        dataset = load_dataset(path=repo_id, name="all")
    except ValueError as e:
        raise e
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    df = dataset["train"].to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Rename the columns to match MMLU format
    df.rename(columns={"answer": "label"}, inplace=True)

    # Remove samples with missing required fields
    df = df.dropna(
        subset=["question", "option_a", "option_b", "option_c", "option_d", "label"]
    )

    # Remove the samples with overly short or long texts
    df = df[
        (df.question.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
        & (df.question.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
        & (df.option_a.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_a.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.option_b.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_b.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.option_c.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_c.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.option_d.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_d.str.len() <= MAX_NUM_CHARS_IN_OPTION)
    ]

    # Handle option_e if it exists and is not null
    if "option_e" in df.columns:
        df = df[df.option_e.isna() | (df.option_e.str.len() <= MAX_NUM_CHARS_IN_OPTION)]

    def is_repetitive(text: str) -> bool:
        """Return True if the text is repetitive."""
        if pd.isna(text):
            return False
        max_repetitions = max(Counter(str(text).split()).values())
        return max_repetitions > MAX_REPETITIONS

    # Remove overly repetitive samples
    df = df[
        ~df.question.apply(is_repetitive)
        & ~df.option_a.apply(is_repetitive)
        & ~df.option_b.apply(is_repetitive)
        & ~df.option_c.apply(is_repetitive)
        & ~df.option_d.apply(is_repetitive)
    ]

    # Create category from test_id and section
    df["category"] = df["test_id"] + "_" + df["section"].fillna("unknown")

    # Make a `text` column with all the options in it
    def create_text(row: pd.Series) -> str:
        """Create the text column.

        Args:
            row: The row of the dataframe.

        Returns:
            The text column.
        """
        # Use question_resource if available, otherwise just question
        instruction = (
            row.question_resource if pd.notna(row.question_resource) else row.question
        )
        instruction = str(instruction).replace("\n", " ").strip()

        text = instruction + f"\n{CHOICES_MAPPING['sv']}:\n"
        text += "a. " + str(row.option_a).replace("\n", " ").strip() + "\n"
        text += "b. " + str(row.option_b).replace("\n", " ").strip() + "\n"
        text += "c. " + str(row.option_c).replace("\n", " ").strip() + "\n"
        text += "d. " + str(row.option_d).replace("\n", " ").strip()

        # Add option_e if it exists and is not null
        if "option_e" in df.columns and pd.notna(row.option_e):
            text += "\ne. " + str(row.option_e).replace("\n", " ").strip()

        return text

    df["text"] = df.apply(create_text, axis=1)

    # Make the `label` column case-consistent with the `text` column
    df.label = df.label.str.lower()

    # Only keep the `text`, `label` and `category` columns
    df = df[["text", "label", "category"]]

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create train and test splits
    train_size = 32
    test_size = 480

    train_arr, test_arr = train_test_split(
        df,
        train_size=train_size,
        test_size=test_size,
        random_state=4242,
        stratify=df.category,
    )

    train_df = pd.DataFrame(train_arr, columns=df.columns)
    test_df = pd.DataFrame(test_arr, columns=df.columns)

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/skolprov"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
