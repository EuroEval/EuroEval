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

"""Create the CS GEC linguistic acceptability dataset."""

import logging

import pandas as pd
from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("create_cs_gec")


def main() -> None:
    """Create the CS GEC linguistic acceptability dataset and upload to HF Hub."""
    # Load the dataset using Hugging Face datasets library
    dataset = load_dataset("CZLC/cs_gec")

    # Convert the train and test datasets to pandas DataFrames
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # Rename columns to match the expected format
    train_df = train_df.rename(columns={"query": "text", "gold": "label"})
    test_df = test_df.rename(columns={"query": "text", "gold": "label"})
    train_df["label"] = train_df["label"].apply(
        lambda x: "correct" if x == 1 else "incorrect"
    )
    test_df["label"] = test_df["label"].apply(
        lambda x: "correct" if x == 1 else "incorrect"
    )

    # Filter data based on text length
    train_df = filter_by_text_length(train_df)
    test_df = filter_by_text_length(test_df)

    # Sample a larger number of rows to create train and val splits
    sampled_train_df = train_df.sample(n=1280, random_state=42)  # 1024 + 256

    # Create a validation split from the sampled train data
    train_df, val_df = train_test_split(
        sampled_train_df, train_size=1024, random_state=42
    )

    # Sample the desired number of rows for the test split
    test_df = test_df.sample(n=2048, random_state=42)

    # Create a DatasetDict
    dataset_dict = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN, preserve_index=False),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION, preserve_index=False),
        test=Dataset.from_pandas(test_df, split=Split.TEST, preserve_index=False),
    )

    # Upload to Hugging Face Hub
    dataset_id = "EuroEval/cs-gec-mini"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset_dict.push_to_hub(dataset_id, private=True)


def filter_by_text_length(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the dataframe by text length."""
    df["text_len"] = df.text.str.len()
    return df.query(
        "text_len >= @MIN_NUM_CHARS_IN_DOCUMENT and "
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )


if __name__ == "__main__":
    main()
