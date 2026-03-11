# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the MMLU-Pro-mini dataset and upload it to the HF Hub."""

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

OPTION_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def main() -> None:
    """Create the MMLU-Pro-mini dataset and upload it to the HF Hub."""
    # Load the MMLU-Pro dataset
    dataset = load_dataset(path="TIGER-Lab/MMLU-Pro", token=True)
    assert isinstance(dataset, DatasetDict)

    # The original validation split contains the few-shot training examples - use it
    # directly as our training split.
    train_df = dataset["validation"].to_pandas()
    test_source_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_source_df, pd.DataFrame)

    train_df = process_df(train_df)
    test_source_df = process_df(test_source_df)

    # Split the original test split into val and test splits (256/2048), stratified on
    # category.
    val_size = 256
    remaining_test_df, val_df = train_test_split(
        test_source_df,
        test_size=val_size,
        random_state=4242,
        stratify=test_source_df.category,
    )

    test_size = 2048
    _, test_df = train_test_split(
        remaining_test_df,
        test_size=test_size,
        random_state=4242,
        stratify=remaining_test_df.category,
    )

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
    dataset_id = "EuroEval/mmlu-pro-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """Process a raw MMLU-Pro dataframe into the EuroEval format.

    Args:
        df:
            The raw dataframe to process.

    Returns:
        The processed dataframe with `text`, `label`, and `category` columns.
    """
    # Rename the columns
    df = df.rename(columns=dict(question="instruction"))

    # Make the answer lowercase
    df["label"] = df["answer"].str.lower()

    # Expand the options list into individual columns (fill None for missing options)
    for i, letter in enumerate(OPTION_LABELS):
        df[f"option_{letter}"] = df["options"].apply(
            lambda opts, idx=i: opts[idx] if idx < len(opts) else None
        )

    # Remove the samples with overly short or long texts
    option_cols = [f"option_{letter}" for letter in OPTION_LABELS]
    length_filter = (df.instruction.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION) & (
        df.instruction.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION
    )
    for col in option_cols:
        present = df[col].notna()
        length_filter = length_filter & (  # pyrefly: ignore[unsupported-operation]
            ~present
            | (
                (df[col].str.len() >= MIN_NUM_CHARS_IN_OPTION)
                & (df[col].str.len() <= MAX_NUM_CHARS_IN_OPTION)
            )
        )
    df = df[length_filter]

    def is_repetitive(text: str) -> bool:
        """Return True if the text is repetitive."""
        max_repetitions = max(Counter(text.split()).values())
        return max_repetitions > MAX_REPETITIONS

    # Remove overly repetitive samples
    repetition_filter = ~df.instruction.apply(is_repetitive)
    for col in option_cols:
        present = df[col].notna()
        repetition_filter = (
            repetition_filter
            & (  # pyrefly: ignore[unsupported-operation]
                ~present
                | ~df[col].apply(lambda x: is_repetitive(x) if pd.notna(x) else False)
            )
        )
    df = df[repetition_filter]

    # Make a `text` column with all the options in it
    df["text"] = [
        row.instruction.replace("\n", " ").strip()
        + "\n"
        + f"{CHOICES_MAPPING['en']}:\n"
        + "\n".join(
            f"{letter.lower()}. " + row[f"option_{letter}"].replace("\n", " ").strip()
            for letter in OPTION_LABELS
            if pd.notna(row[f"option_{letter}"])
        )
        for _, row in df.iterrows()
    ]

    # Only keep the `text`, `label` and `category` columns
    df = df[["text", "label", "category"]]

    # Remove duplicates
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    main()
