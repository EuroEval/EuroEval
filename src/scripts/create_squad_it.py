# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the SQuAD-it-mini dataset and upload it to the HF Hub."""

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
    """Create the SQuAD-it-mini dataset and upload it to the HF Hub."""
    dataset_id = "crux82/squad_it"

    # Load the dataset
    dataset = load_dataset(dataset_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset to a dataframe
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Only work with samples where the context is not very large or small
    train_lengths = train_df.context.str.len()
    test_lengths = test_df.context.str.len()
    train_df = train_df[
        train_lengths.between(MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT)
    ]
    test_df = test_df[
        test_lengths.between(MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT)
    ]

    # Only work with samples where the question is not very large or small
    train_question_lengths = train_df.question.str.len()
    test_question_lengths = test_df.question.str.len()
    train_df = train_df[
        train_question_lengths.between(
            MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION
        )
    ]
    test_df = test_df[
        test_question_lengths.between(
            MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION
        )
    ]

    # Test split has repeat answer text and start indexes, so we just take the first
    test_df["answers"] = test_df.apply(
        lambda row: {
            "text": [row["answers"]["text"][0]],
            "answer_start": [row["answers"]["answer_start"][0]],
        },
        axis=1,
    )
    # Remove rows where the answer text is just a single character
    test_df = test_df[test_df["answers"].apply(lambda x: len(x["text"][0]) > 1)]

    # Create validation split
    val_size = 256
    val_df = test_df.sample(n=val_size, random_state=4242)

    # Create train split
    train_size = 1024
    train_df = train_df.sample(n=train_size, random_state=4242)

    # Create test split
    test_size = 2048
    test_df = test_df.loc[~test_df.index.isin(val_df.index)].sample(
        n=test_size, random_state=4242
    )

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
    mini_dataset_id = "EuroEval/squad-it-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api: HfApi = HfApi()
        api.delete_repo(mini_dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(mini_dataset_id, private=True)


if __name__ == "__main__":
    main()
