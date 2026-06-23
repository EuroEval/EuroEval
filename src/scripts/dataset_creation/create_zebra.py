# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==5.0.0",
#     "huggingface-hub==1.20.1",
#     "pandas==3.0.3",
#     "requests==2.34.2",
#     "scikit-learn==1.6.1",
# ]
# ///

"""Create the zebra puzzle datasets and upload them to the HF Hub."""

import json
import typing as t

import pandas as pd
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

# All themes except Danish smørrebrød variants
# Format: (theme, language_code, difficulty)
THEMES = [
    # Danish (house theme)
    ("da_huse_2x3_5rh", "da", "easy"),
    ("da_huse_4x5_5rh", "da", "hard"),
    # German
    ("de_hauser_2x3_5rh", "de", "easy"),
    ("de_hauser_4x5_5rh", "de", "hard"),
    # English
    ("en_houses_2x3_5rh", "en", "easy"),
    ("en_houses_4x5_5rh", "en", "hard"),
    # Faroese
    ("fo_hus_2x3_5rh", "fo", "easy"),
    ("fo_hus_4x5_5rh", "fo", "hard"),
    # Icelandic
    ("is_husum_2x3_5rh", "is", "easy"),
    ("is_husum_4x5_5rh", "is", "hard"),
    # Norwegian Bokmål
    ("nb_hus_2x3_5rh", "nb", "easy"),
    ("nb_hus_4x5_5rh", "nb", "hard"),
    # Dutch
    ("nl_huizen_2x3_5rh", "nl", "easy"),
    ("nl_huizen_4x5_5rh", "nl", "hard"),
    # Norwegian Nynorsk
    ("nn_hus_2x3_5rh", "nn", "easy"),
    ("nn_hus_4x5_5rh", "nn", "hard"),
    # Swedish
    ("sv_hus_2x3_5rh", "sv", "easy"),
    ("sv_hus_4x5_5rh", "sv", "hard"),
]
# Split sizes from original dataset (arXiv:2511.03553)
n_train = 128
n_val = 128
n_test = 1024


def main() -> None:
    """Create the zebra puzzle datasets and upload them to the HF Hub."""
    # Define the base download URL
    repo_id = "alexandrainst/multi-zebra-logic"

    for theme, lang_code, difficulty in THEMES:
        # Load dataset from parquet files directly to avoid deprecated List feature type
        # The source dataset uses List(Value(...)) which was removed in datasets 5.0.0
        train_data: Dataset = Dataset.from_parquet(
            f"hf://datasets/{repo_id}/dataset_{theme}/train-00000-of-00001.parquet"
        )
        val_data: Dataset = Dataset.from_parquet(
            f"hf://datasets/{repo_id}/dataset_{theme}/val-00000-of-00001.parquet"
        )
        test_data: Dataset = Dataset.from_parquet(
            f"hf://datasets/{repo_id}/dataset_{theme}/test-00000-of-00001.parquet"
        )

        # Check length
        assert len(train_data) == n_train
        assert len(val_data) == n_val
        assert len(test_data) == n_test

        # Convert the dataset to a dataframe
        train_df: pd.DataFrame = t.cast(pd.DataFrame, train_data.to_pandas())
        val_df: pd.DataFrame = t.cast(pd.DataFrame, val_data.to_pandas())
        test_df: pd.DataFrame = t.cast(pd.DataFrame, test_data.to_pandas())

        # Remove unused columns
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
        train_df["target_text"] = train_df["target_text"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
        val_df["target_text"] = val_df["target_text"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
        test_df["target_text"] = test_df["target_text"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )

        # Collect datasets in a dataset dictionary
        dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
                "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
                "test": Dataset.from_pandas(test_df, split=Split.TEST),
            }
        )

        # Create dataset ID
        dataset_id = f"EuroEval/zebra-puzzle-{difficulty}-{lang_code}"

        # Remove the dataset from Hugging Face Hub if it already exists
        try:
            HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
        except HfHubHTTPError as e:
            print(f"Could not delete existing dataset {dataset_id}: {e}")

        # Push the dataset to the Hugging Face Hub
        dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
