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

"""Create the zebra puzzle datasets and upload them to the HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi

THEMES = ["da_huse_2x3_5rh", "da_huse_4x5_5rh"]
n_train = 128
n_test = 1024


def main() -> None:
    """Create the zebra puzzle datasets and upload them to the HF Hub."""
    # Define the base download URL
    repo_id = "alexandrainst/zebra_puzzles"

    for theme in THEMES:
        # Download the dataset
        train_data: Dataset = load_dataset(
            path=repo_id, name=f"dataset_{theme}", token=True, split="train"
        )
        test_data: Dataset = load_dataset(
            path=repo_id, name=f"dataset_{theme}", token=True, split="test"
        )

        # Check length
        assert len(train_data) == n_train
        assert len(test_data) == n_test

        # Convert the dataset to a dataframe
        train_df: pd.DataFrame = train_data.to_pandas()
        test_df: pd.DataFrame = test_data.to_pandas()

        # Remove unused columns
        train_df = train_df[["introduction", "clues", "format_example", "solution"]]
        test_df = test_df[["introduction", "clues", "format_example", "solution"]]

        # Rename the solution column as label
        train_df.rename(columns={"solution": "label"}, inplace=True)
        test_df.rename(columns={"solution": "label"}, inplace=True)

        # Collect datasets in a dataset dictionary
        dataset = DatasetDict(
            train=Dataset.from_pandas(train_df, split=Split.TRAIN),
            test=Dataset.from_pandas(test_df, split=Split.TEST),
        )

        # Create dataset ID
        dataset_id = f"EuroEval/zebra-puzzles-{theme}"

        # Remove the dataset from Hugging Face Hub if it already exists
        HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

        # Push the dataset to the Hugging Face Hub
        dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
