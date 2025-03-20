# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the NorCommonSenseQA dataset and upload them to the HF Hub."""

import warnings
from collections import Counter

import pandas as pd
from constants import (
    MAX_NUM_CHARS_IN_INSTRUCTION,
    MAX_REPETITIONS,
    MIN_NUM_CHARS_IN_INSTRUCTION,
)
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from pandas.errors import SettingWithCopyWarning
from requests import HTTPError

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def main() -> None:
    """Create the NorCommonSenseQA dataset and upload them to the HF Hub."""
    # Define the base download URL
    repo_id = "ltg/norcommonsenseqa"

    # Download the dataset
    nb_dataset = load_dataset(path=repo_id, name="nb", token=True, split="train")
    nn_dataset = load_dataset(path=repo_id, name="nn", token=True, split="train")
    assert isinstance(nb_dataset, Dataset) and isinstance(nn_dataset, Dataset)

    # Convert the dataset to a dataframe
    nb_df = nb_dataset.to_pandas()
    nn_df = nn_dataset.to_pandas()
    assert isinstance(nb_df, pd.DataFrame) and isinstance(nn_df, pd.DataFrame)

    # Merge the datasets
    nb_df["language"] = "nb"
    nn_df["language"] = "nn"
    df = pd.concat([nb_df, nn_df], ignore_index=True)

    # Rename the columns
    df.rename(columns=dict(question="instruction", answer="label"), inplace=True)

    # Remove the samples with overly short or long texts
    df = df[
        (df.instruction.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
        & (df.instruction.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
    ]

    def is_repetitive(text: str) -> bool:
        """Return True if the text is repetitive."""
        max_repetitions = max(Counter(text.split()).values())
        return max_repetitions > MAX_REPETITIONS

    # Remove overly repetitive samples
    df = df[~df.instruction.apply(is_repetitive)]
    assert isinstance(df, pd.DataFrame)

    # Make a `text` column with all the options in it
    df["text"] = [
        row.instruction.replace("\n", " ").strip() + "\n"
        "Svaralternativer:\n"
        + "\n".join(
            [
                f"{char}. {clean_text(text=option)}"
                for char, option in zip("abcde", row.choices["text"])
            ]
        )
        for _, row in df.iterrows()
    ]

    # Make the `label` column case-consistent with the `text` column
    df.label = df.label.str.lower()

    # Only keep the `text` and `label` columns
    df = df[["text", "label", "curated"]]
    assert isinstance(df, pd.DataFrame)

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create train split, where we add all the non-curated samples and top up with
    # random curated samples
    train_size = 128
    train_df = df.query("curated == False")
    df.drop(index=train_df.index.tolist(), inplace=True)
    samples_to_add = max(0, train_size - len(train_df))
    new_train_samples = df.sample(samples_to_add, random_state=4242)
    df.drop(index=new_train_samples.index.tolist(), inplace=True)
    train_df = pd.concat([train_df, new_train_samples], ignore_index=True)

    # Create validation split
    val_size = 128
    val_df = df.sample(val_size, random_state=4242)
    df.drop(index=val_df.index.tolist(), inplace=True)

    # Create test split
    test_df = df
    assert len(test_df) > 850

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
    dataset_id = "EuroEval/nor-common-sense-qa"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def clean_text(text: str) -> str:
    """Clean the text.

    Args:
        text:
            The text to be cleaned.

    Returns:
        The cleaned text.
    """
    return text.replace("\n", " ").strip()


if __name__ == "__main__":
    main()
