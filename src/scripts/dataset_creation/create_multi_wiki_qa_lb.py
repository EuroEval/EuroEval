# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.6.0",
#     "huggingface-hub==0.33.0",
#     "pandas==2.3.0",
#     "requests==2.32.4",
#     "tqdm==4.67.1",
# ]
# ///

"""Create the MultiWikiQA-lb reading comprehension dataset and upload to HF Hub.

This is a Luxembourgish subset extracted from alexandrainst/multi-wiki-qa.
Upload to EuroEval/multi-wiki-qa-lb on Hugging Face.
"""

import logging

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
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Create the MultiWikiQA-lb dataset and upload to HF Hub."""
    dataset_id = "alexandrainst/multi-wiki-qa"
    language = "lb"
    target_dataset_id = "EuroEval/multi-wiki-qa-lb"

    logger.info(f"Loading {language} subset from {dataset_id}...")

    # Load the dataset
    dataset = load_dataset(dataset_id, name=language, token=True, split="train")

    # Convert to dataframe
    df = dataset.to_pandas()

    # Filter by context length
    lengths = df.context.str.len()
    df = df[lengths.between(MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT)]

    # Filter by question length
    question_lengths = df.question.str.len()
    df = df[question_lengths.between(MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION)]

    # Filter to samples with answers
    def has_answer_fn(example: dict) -> bool:
        return len(example["text"]) > 0 and example["text"][0] != ""

    has_answer: pd.Series = df.answers.map(has_answer_fn)
    df_with_answer: pd.DataFrame = df.loc[has_answer]

    logger.info(f"Loaded {len(df_with_answer)} samples with answers")

    # Create validation split
    val_size = 256
    val_df = df_with_answer.sample(n=val_size, random_state=4242)
    df_with_answer = df_with_answer.loc[~df_with_answer.index.isin(val_df.index)]

    # Create test split
    test_size = 2048
    test_df = df_with_answer.sample(n=test_size, random_state=4242)
    df_with_answer = df_with_answer.loc[~df_with_answer.index.isin(test_df.index)]

    # Create train split
    train_size = 1024
    train_df = df_with_answer.sample(n=train_size, random_state=4242)

    # Reset indices
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    logger.info(f"Created splits: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    # Create dataset
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    # Delete and re-upload
    logger.info(f"Uploading to {target_dataset_id}...")
    HfApi().delete_repo(target_dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(target_dataset_id, private=True)
    logger.info(f"✓ Uploaded {target_dataset_id}")


if __name__ == "__main__":
    main()
