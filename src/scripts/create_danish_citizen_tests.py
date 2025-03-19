# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "python-dotenv==1.0.1",
# ]
# ///

"""Create the Danish Citizen Tests dataset and upload it to the HF Hub."""

import os
import warnings

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from pandas.errors import SettingWithCopyWarning
from requests import HTTPError

load_dotenv()


warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def main() -> None:
    """Create the DanishCitizenTests-mini dataset and upload it to the HF Hub."""
    # Define the base download URL
    repo_id = "alexandrainst/danish-citizen-tests-updated"

    # Download the dataset
    dataset = load_dataset(
        path=repo_id, split="train", token=os.getenv("HUGGINGFACE_API_KEY")
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
            + "\nSvarmuligheder:\n"
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

    df = df[["text", "label", "test_type", "year"]]

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Split data into the two tests
    citizenship_test_df = df.query("test_type == 'indfødsretsprøven'")
    permanent_residence_test_df = df.query("test_type == 'medborgerskabsprøven'")

    # Create test split, containing all the citizenship tests and the newest permanent
    # residence tests
    test_size = 512
    test_df = citizenship_test_df
    for year in sorted(permanent_residence_test_df.year.unique(), reverse=True):
        year_df = permanent_residence_test_df.query("year == @year")
        test_df = pd.concat([test_df, year_df], ignore_index=True)
        if len(test_df) >= test_size:
            break
    permanent_residence_test_df.drop(
        index=test_df.index.tolist(), inplace=True, errors="ignore"
    )

    # Create validation split, containing the newer permanent residence tests (aside
    # from the ones we already added to the test split)
    val_size = 64
    val_df = pd.DataFrame()
    for year in sorted(permanent_residence_test_df.year.unique(), reverse=True):
        year_df = permanent_residence_test_df.query("year == @year")
        val_df = pd.concat([val_df, year_df], ignore_index=True)
        if len(val_df) >= val_size:
            break
    permanent_residence_test_df.drop(
        index=val_df.index.tolist(), inplace=True, errors="ignore"
    )

    # Create train split as the remaining data
    train_df = permanent_residence_test_df
    assert len(train_df) > 256, f"Not enough data for training: {len(train_df):,}"

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
    dataset_id = "EuroEval/danish-citizen-tests-updated"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

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
