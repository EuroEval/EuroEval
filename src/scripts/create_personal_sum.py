# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Personal Sum summarisation dataset."""

from logging import getLogger

import pandas as pd
from constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from requests import HTTPError

logger = getLogger(__name__)


def main() -> None:
    """Create the Personal Sum summarisation dataset and upload to HF Hub."""
    dataset_url = "https://raw.githubusercontent.com/SmartmediaAI/PersonalSum/refs/heads/main/dataset/PersonalSum_original.csv"
    df = pd.read_csv(dataset_url)

    # Only keep the columns we need and rename them.
    df = df[["Article", "Worker_summary"]]
    df = df.rename(columns={"Article": "text", "Worker_summary": "target_text"})

    # Strip leading and trailing whitespace
    df["text"] = df["text"].str.strip()
    df["target_text"] = df["target_text"].str.strip()

    # Remove "Oppsummering: " from the start of the target_text
    df["target_text"] = df["target_text"].str.replace(
        r"^Oppsummering: ", "", regex=True
    )

    # Check bounds
    text_lengths = df["text"].str.len()
    lower_bound = MIN_NUM_CHARS_IN_ARTICLE
    upper_bound = MAX_NUM_CHARS_IN_ARTICLE

    df = df[text_lengths.between(lower_bound, upper_bound)]

    # Group by article: each article will now have 1 or more summaries
    df = df.groupby("text")["target_text"].apply(list).reset_index()

    logger.info(f"Total length of dataset: {len(df)}")

    # Make splits
    val_size = 64
    test_size = 256

    val_df = df.sample(val_size, random_state=42)
    df = df.drop(val_df.index)
    test_df = df.sample(test_size, random_state=42)
    train_df = df.drop(test_df.index)
    assert len(train_df) > 100, "The training set should have at least 100 samples."

    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/personal-sum"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api: HfApi = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
