# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the SQAD-mini Czech dataset and upload it to the HF Hub."""

import hashlib

from constants import (
    MAX_NUM_CHARS_IN_CONTEXT,
    MAX_NUM_CHARS_IN_QUESTION,
    MIN_NUM_CHARS_IN_CONTEXT,
    MIN_NUM_CHARS_IN_QUESTION,
)
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the SQAD-mini Czech dataset and upload it to the HF Hub."""
    # Load the dataset
    dataset = load_dataset("CZLC/sqad_3.2_filtered")

    # Convert to DataFrames
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # Filter out data points with more than one answer
    train_df = train_df[train_df["answers"].apply(len) == 1]
    test_df = test_df[test_df["answers"].apply(len) == 1]

    # Filter based on context and question length
    train_df = train_df[
        train_df.context.str.len().between(
            MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT
        )
        & train_df.question.str.len().between(
            MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION
        )
    ]
    test_df = test_df[
        test_df.context.str.len().between(
            MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT
        )
        & test_df.question.str.len().between(
            MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION
        )
    ]

    # Extract answer start positions
    train_df["answer_start"] = train_df.apply(
        lambda row: row["context"].find(row["answers"][0]), axis=1
    )
    test_df["answer_start"] = test_df.apply(
        lambda row: row["context"].find(row["answers"][0]), axis=1
    )

    # Make the `answers` column a dictionary with the answer and the answer start
    train_df["answers"] = train_df.apply(
        lambda row: {
            "answer_start": [row["answer_start"]],
            "text": [row["answers"][0]],
        },
        axis=1,
    )
    test_df["answers"] = test_df.apply(
        lambda row: {
            "answer_start": [row["answer_start"]],
            "text": [row["answers"][0]],
        },
        axis=1,
    )

    # Define split sizes
    train_size = 1024
    val_size = 256
    test_size = 2048

    # Create new train and validation splits from the train split
    final_train_df = train_df.sample(n=train_size, random_state=4242)
    remaining_df = train_df.drop(final_train_df.index)
    final_val_df = remaining_df.sample(n=val_size, random_state=4242)

    # Use the existing test split
    final_test_df = test_df.sample(n=test_size, random_state=4242)

    # Reset indices
    final_train_df = final_train_df.reset_index(drop=True)
    final_val_df = final_val_df.reset_index(drop=True)
    final_test_df = final_test_df.reset_index(drop=True)

    # Add ID column
    final_train_df["id"] = [
        hashlib.md5(
            (row.title + row.context + row.question).encode("utf-8")
        ).hexdigest()
        for _, row in final_train_df.iterrows()
    ]
    final_val_df["id"] = [
        hashlib.md5(
            (row.title + row.context + row.question).encode("utf-8")
        ).hexdigest()
        for _, row in final_val_df.iterrows()
    ]
    final_test_df["id"] = [
        hashlib.md5(
            (row.title + row.context + row.question).encode("utf-8")
        ).hexdigest()
        for _, row in final_test_df.iterrows()
    ]

    # Check that the IDs are unique
    assert final_train_df.id.nunique() == len(final_train_df)
    assert final_val_df.id.nunique() == len(final_val_df)
    assert final_test_df.id.nunique() == len(final_test_df)

    # Create DatasetDict
    dataset_dict = DatasetDict(
        train=Dataset.from_pandas(final_train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(final_val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(final_test_df, split=Split.TEST),
    )

    # Push to Hugging Face Hub
    dataset_id = "EuroEval/sqad-mini"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset_dict.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
