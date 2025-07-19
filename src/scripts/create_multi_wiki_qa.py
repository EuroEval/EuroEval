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
from requests import HTTPError
from tqdm.auto import tqdm

ALL_LANGUAGES = [
    "da",
    "nl",
    "en",
    "fi",
    "fo",
    "fr",
    "de",
    "is",
    "it",
    "no",
    "nn",
    "es",
    "sv",
    "pt-pt",
]


def main() -> None:
    """Create the MultiWikiQA-mini datasets and upload them to the HF Hub."""
    dataset_id = "alexandrainst/multi-wiki-qa"

    for language in tqdm(ALL_LANGUAGES, desc="Generating datasets"):
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

        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)

        # Collect datasets in a dataset dictionary
        dataset = DatasetDict(
            train=Dataset.from_pandas(train_df, split=Split.TRAIN),
            val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
            test=Dataset.from_pandas(test_df, split=Split.TEST),
        )

        # Create dataset ID
        mini_dataset_id = f"EuroEval/multi-wiki-qa-{language}-mini"

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
