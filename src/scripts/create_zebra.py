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

import json
import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

THEMES = ["da_huse_2x3_5rh", "da_huse_4x5_5rh"]
n_train = 128
n_val = 128
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
        val_data: Dataset = load_dataset(
            path=repo_id, name=f"dataset_{theme}", token=True, split="val"
        )
        test_data: Dataset = load_dataset(
            path=repo_id, name=f"dataset_{theme}", token=True, split="test"
        )

        # Check length
        assert len(train_data) == n_train
        assert len(val_data) == n_val
        assert len(test_data) == n_test

        # Convert the dataset to a dataframe
        train_df: pd.DataFrame = train_data.to_pandas()
        val_df: pd.DataFrame = val_data.to_pandas()
        test_df: pd.DataFrame = test_data.to_pandas()

        # Remove unused columns
        # TODO: Consider including format_example in the dataset
        train_df = train_df[["introduction", "clues", "solution"]]
        val_df = val_df[["introduction", "clues", "solution"]]
        test_df = test_df[["introduction", "clues", "solution"]]

        # Combine introduction and clues into a single text column
        train_df["text"] = train_df["introduction"] + train_df["clues"].apply(
            lambda clues: "\n".join(clues)
        )
        val_df["text"] = val_df["introduction"] + val_df["clues"].apply(
            lambda clues: "\n".join(clues)
        )
        test_df["text"] = test_df["introduction"] + test_df["clues"].apply(
            lambda clues: "\n".join(clues)
        )

        # Rename the solution column as label
        train_df.rename(columns={"solution": "target_text"}, inplace=True)
        val_df.rename(columns={"solution": "target_text"}, inplace=True)
        test_df.rename(columns={"solution": "target_text"}, inplace=True)


        # Convert numpy arrays in target_text (the values of each dict) to lists
        train_df["target_text"] = train_df["target_text"].apply(
            lambda sol: {k: v.tolist() for k, v in sol.items()}
        )
        val_df["target_text"] = val_df["target_text"].apply(
            lambda sol: {k: v.tolist() for k, v in sol.items()}
        )
        test_df["target_text"] = test_df["target_text"].apply(
            lambda sol: {k: v.tolist() for k, v in sol.items()}
        )
        

        # Convert target_text from dict to string
        train_df["target_text"] = train_df["target_text"].apply(lambda x: json.dumps(x, ensure_ascii=False))
        val_df["target_text"] = val_df["target_text"].apply(lambda x: json.dumps(x, ensure_ascii=False))
        test_df["target_text"] = test_df["target_text"].apply(lambda x: json.dumps(x, ensure_ascii=False))


        # Collect datasets in a dataset dictionary
        dataset = DatasetDict(
            train=Dataset.from_pandas(train_df, split=Split.TRAIN),
            val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
            test=Dataset.from_pandas(test_df, split=Split.TEST),
        )

        # Create dataset ID
        dataset_id = f"EuroEval/zebra-puzzles-{theme}"

        # Remove the dataset from Hugging Face Hub if it already exists
        try:
            HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
        except HfHubHTTPError as e:
            print(f"Could not delete existing dataset {dataset_id}: {e}")

        # Push the dataset to the Hugging Face Hub
        dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
