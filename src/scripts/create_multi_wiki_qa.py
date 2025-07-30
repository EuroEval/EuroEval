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

"""Create the MultiWikiQA-mini datasets and upload them to the HF Hub."""

import logging
import time

import pandas as pd
from constants import (
    MAX_NUM_CHARS_IN_CONTEXT,
    MAX_NUM_CHARS_IN_QUESTION,
    MIN_NUM_CHARS_IN_CONTEXT,
    MIN_NUM_CHARS_IN_QUESTION,
)
from datasets import disable_progress_bars
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from requests import HTTPError
from tqdm.auto import tqdm

logger = logging.getLogger("create_multi_wiki_qa")


def main() -> None:
    """Create the MultiWikiQA-mini datasets and upload them to the HF Hub."""
    disable_progress_bars()

    dataset_id = "alexandrainst/multi-wiki-qa"

    dataset_languages = [
        cfg["config_name"].split(".")[-1]
        for cfg in HfApi()
        .repo_info(repo_id=dataset_id, repo_type="dataset")
        .card_data.configs
    ]

    api: HfApi = HfApi()
    for language in tqdm(dataset_languages, desc="Generating MultiWikiQA datasets"):
        # Create dataset ID
        mini_dataset_id = f"EuroEval/multi-wiki-qa-{language}-mini"

        # Skip if the dataset already exists
        if api.repo_exists(repo_id=mini_dataset_id, repo_type="dataset"):
            continue

        # Load the dataset
        dataset = load_dataset(dataset_id, name=language, token=True, split="train")
        assert isinstance(dataset, Dataset)

        # Convert the dataset to a dataframe
        df = dataset.to_pandas()
        assert isinstance(df, pd.DataFrame)

        # Only work with samples where the context is not very large or small
        lengths = df.context.str.len()
        df = df[lengths.between(MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT)]

        # Only work with samples where the question is not very large or small
        question_lengths = df.question.str.len()
        df = df[
            question_lengths.between(
                MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION
            )
        ]

        # Extract information on which examples contain an answer
        def has_answer_fn(example: dict) -> bool:
            return len(example["text"]) > 0 and example["text"][0] != ""

        has_answer: pd.Series = df.answers.map(has_answer_fn)

        # Only work with the questions having answers in the context
        df_with_answer: pd.DataFrame = df.loc[has_answer]

        # Determine the size of the splits
        if len(df_with_answer) < 1024 + 32 + 128:
            logger.warning(
                f"Skipping language {language!r} because it has too few samples: "
                f"{len(df_with_answer):,}"
            )
            continue
        elif len(df_with_answer) > 1024 + 256 + 2048:
            train_size = 1024
            val_size = 256
            test_size = 2048
        elif len(df_with_answer) > 1024 + 128 + 1024:
            train_size = 1024
            val_size = 128
            test_size = 1024
        elif len(df_with_answer) > 1024 + 64 + 512:
            train_size = 1024
            val_size = 64
            test_size = 512
        else:
            train_size = 1024
            val_size = 32
            test_size = 128

        # Create validation split
        val_df = df_with_answer.sample(n=val_size, random_state=4242)
        df_with_answer = df_with_answer.loc[~df_with_answer.index.isin(val_df.index)]

        # Create test split
        test_df = df_with_answer.sample(n=test_size, random_state=4242)
        df_with_answer = df_with_answer.loc[~df_with_answer.index.isin(test_df.index)]

        # Create train split
        train_df = df_with_answer.sample(n=train_size, random_state=4242)

        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)

        # Collect datasets in a dataset dictionary
        dataset = DatasetDict(
            train=Dataset.from_pandas(train_df, split=Split.TRAIN),
            val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
            test=Dataset.from_pandas(test_df, split=Split.TEST),
        )

        # Remove the dataset from Hugging Face Hub if it already exists
        try:
            api.delete_repo(mini_dataset_id, repo_type="dataset")
        except HTTPError:
            pass

        # Push the dataset to the Hugging Face Hub
        while True:
            try:
                dataset.push_to_hub(mini_dataset_id, private=True)
                break
            except HfHubHTTPError as e:
                if "too many requests" in str(e).lower():
                    logger.error(
                        "Too many requests to the Hugging Face Hub. Waiting for a "
                        "minute and retrying..."
                    )
                    time.sleep(60)
                else:
                    raise e


if __name__ == "__main__":
    main()
