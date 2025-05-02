# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the TydiQA-mini Finnish dataset and upload it to the HF Hub."""

import pandas as pd
from constants import (
    MAX_NUM_CHARS_IN_CONTEXT,
    MAX_NUM_CHARS_IN_QUESTION,
    MIN_NUM_CHARS_IN_CONTEXT,
    MIN_NUM_CHARS_IN_QUESTION,
)
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from huggingface_hub.hf_api import HfApi
from requests.exceptions import HTTPError


def main() -> None:
    """Create the TydiQA-mini Finnish dataset and upload it to the HF Hub."""
    dataset_id = "google-research-datasets/tydiqa"

    # Load the datasets from the `alexandrainst` organisation
    train = load_dataset(dataset_id, "secondary_task", split="train", token=True)
    val = load_dataset(dataset_id, "secondary_task", split="validation", token=True)

    # Ensure that the datasets are indeed datasets
    assert isinstance(train, Dataset)
    assert isinstance(val, Dataset)

    # Make train_df and val_df
    train_df = train.to_pandas()
    val_df = val.to_pandas()

    # Extract all Finnish samples
    train_df, val_df = [df[df.id.str.contains("finnish")] for df in (train_df, val_df)]

    train_df, val_df = [process_df(df=df) for df in (train_df, val_df)]

    # Make splits
    train_size = 1024
    val_size = 256
    test_size = 2048

    # Take val_size samples from val_df for validation
    final_val_df = val_df.sample(n=val_size, random_state=4242)

    # Take train_size samples from train_df for training
    final_train_df = train_df.sample(n=train_size, random_state=4242)

    # Remaining samples
    remaining_val_df = val_df.loc[~val_df.index.isin(final_val_df.index)]
    remaining_train_df = train_df.loc[~train_df.index.isin(final_train_df.index)]

    # Use the remaining samples from val_df + additional samples from
    # train_df for testing
    test_samples = remaining_train_df.sample(
        n=test_size - len(remaining_val_df), random_state=4242
    )
    final_test_df = pd.concat([remaining_val_df, test_samples])

    assert len(final_test_df) == test_size
    assert len(final_train_df) == train_size
    assert len(final_val_df) == val_size

    final_val_df = final_val_df.reset_index(drop=True)
    final_test_df = final_test_df.reset_index(drop=True)
    final_train_df = final_train_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(final_train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(final_val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(final_test_df, split=Split.TEST),
    )

    # Create dataset ID
    mini_dataset_id = "EuroEval/tydiqa-fi-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api: HfApi = HfApi()
        api.delete_repo(mini_dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(mini_dataset_id, private=True)


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers and ensure that the context is not too large or small.

    Args:
        df: The dataframe to process.

    Returns:
        The processed dataframe.
    """
    # Only work with samples where the context is not very large or small
    lengths = df.context.str.len()
    df = df[lengths.between(MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT)]

    # Only work with samples where the context is not very large or small
    lengths = df.question.str.len()
    df_with_no_outliers = df[
        lengths.between(MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION)
    ]

    # Only work with the questions having answers in the context
    has_answer: pd.Series = df_with_no_outliers.answers.map(
        lambda dct: dct["text"][0] != ""
    )
    df_with_answer: pd.DataFrame = df_with_no_outliers.loc[has_answer]
    return df_with_answer


if __name__ == "__main__":
    main()
