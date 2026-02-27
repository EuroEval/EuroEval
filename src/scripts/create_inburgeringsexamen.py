# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "python-dotenv==1.0.1",
# ]
# ///

"""Create the Inburgeringsexamen dataset and upload it to the HF Hub."""

import warnings

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from pandas.errors import SettingWithCopyWarning

from euroeval.utils import get_hf_token

from .constants import CHOICES_MAPPING

load_dotenv()


warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def main() -> None:
    """Create the Inburgeringsexamen dataset and upload it to the HF Hub."""
    # Define the base download URL
    repo_id = "alexandrainst/dutch-citizen-tests"

    # Download the dataset
    dataset = load_dataset(
        path=repo_id, split="train", token=get_hf_token(api_key=None)
    )
    assert isinstance(dataset, Dataset)

    # Convert the dataset to a dataframe
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Rename the columns
    df.rename(columns=dict(answer="label", question="instruction"), inplace=True)

    # Make a `text` column with all the options in it
    texts = list()
    for _, row in df.iterrows():
        text = (
            clean_text(text=row.instruction)
            + f"\n{CHOICES_MAPPING['nl']}:\n"
            + "\n".join(
                [
                    f"{letter}. {clean_text(text=option)}"
                    for letter, option in zip("abcd", row.options)
                ]
            )
        )
        texts.append(text)
    df["text"] = texts

    # Make the `label` column case-consistent with the `text` column
    df.label = df.label.str.lower()

    df = df[["text", "label"]]

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create test split from the newest data
    test_size = 512
    test_df = df.tail(test_size).copy()
    remaining_df = df.iloc[: len(df) - test_size].copy()

    # Create validation split
    val_size = 64
    val_df = remaining_df.tail(val_size).copy()
    train_df = remaining_df.iloc[: len(remaining_df) - val_size].copy()

    assert len(train_df) > 0, "Not enough data for training"

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
    dataset_id = "EuroEval/inburgeringsexamen"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def clean_text(text: str) -> str:
    """Clean some text.

    Args:
        text:
            The text to clean.

    Returns:
        The cleaned text.
    """
    return text.replace("\n", " ").strip()


if __name__ == "__main__":
    main()
