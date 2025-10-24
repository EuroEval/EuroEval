# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the LR-Sum summarisation datasets."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi

LANGUAGES = {"uk": "ukr", "sr": "srp"}


def main() -> None:
    """Create the LR-Sum summarisation mini datasets and upload to HF Hub."""
    dataset_id = "bltlab/lr-sum"

    for language, subset in LANGUAGES.items():
        # Load the full dataset and filter for the subset
        dataset = load_dataset(dataset_id, subset)
        assert isinstance(dataset, DatasetDict)

        dataset = dataset.map(make_columns)
        train_df = dataset["train"].to_pandas()
        val_df = dataset["validation"].to_pandas()
        test_df = dataset["test"].to_pandas()
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

        train_df = process_df(df=train_df)
        val_df = process_df(df=val_df)
        test_df = process_df(df=test_df)

        # Make splits
        test_size = 2048
        val_size = 256
        train_size = 1024

        # Calculate how many additional samples needed for test split
        additional_test_samples_needed = test_size - len(test_df)

        # Take additional samples from training set for test split
        additional_test_samples = train_df.sample(
            n=additional_test_samples_needed, random_state=4242
        )

        # Combine original test with additional samples from train
        final_test_df = pd.concat([test_df, additional_test_samples], ignore_index=True)

        # Remove the additional samples from train set
        remaining_train_df = train_df[
            ~train_df.index.isin(additional_test_samples.index)
        ]

        # Sample final splits
        final_test_df = final_test_df.sample(
            n=test_size, random_state=4242
        ).reset_index(drop=True)
        final_val_df = val_df.sample(n=val_size, random_state=4242).reset_index(
            drop=True
        )
        final_train_df = remaining_train_df.sample(
            n=train_size, random_state=4242
        ).reset_index(drop=True)

        # Collect datasets in a dataset dictionary
        mini_dataset = DatasetDict(
            train=Dataset.from_pandas(final_train_df, split=Split.TRAIN),
            val=Dataset.from_pandas(final_val_df, split=Split.VALIDATION),
            test=Dataset.from_pandas(final_test_df, split=Split.TEST),
        )

        # Create dataset ID
        mini_dataset_id = f"EuroEval/lr-sum-{language}-mini"

        # Remove the dataset from Hugging Face Hub if it already exists
        HfApi().delete_repo(mini_dataset_id, repo_type="dataset", missing_ok=True)

        # Push the dataset to the Hugging Face Hub
        mini_dataset.push_to_hub(mini_dataset_id, private=True)


def make_columns(sample: dict) -> dict:
    """Map the dataset to have the text and target_text columns.

    Args:
        sample: A sample from the dataset.

    Returns:
        A sample with the text and target_text columns.
    """
    sample["text"] = f"{sample['title']}\n\n{sample['text']}"
    sample["target_text"] = sample["summary"]
    return sample


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe.

    Args:
        df: A dataframe to process.

    Returns:
        Processed dataframe.
    """
    lengths = df.text.str.len()
    lower_bound = MIN_NUM_CHARS_IN_ARTICLE
    upper_bound = MAX_NUM_CHARS_IN_ARTICLE
    df = df[lengths.between(lower_bound, upper_bound)]

    keep_columns = ["text", "target_text"]
    df = df[keep_columns]

    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    main()
