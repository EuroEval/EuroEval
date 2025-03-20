# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the NorQuAD-mini dataset and upload them to the HF Hub."""

import pandas as pd
import requests
from constants import (
    MAX_NUM_CHARS_IN_CONTEXT,
    MAX_NUM_CHARS_IN_QUESTION,
    MIN_NUM_CHARS_IN_CONTEXT,
    MIN_NUM_CHARS_IN_QUESTION,
)
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.splits import Split
from huggingface_hub.hf_api import HfApi
from requests.exceptions import HTTPError


def main() -> None:
    """Create the NorQuAD-mini dataset and upload them to the HF Hub."""
    url = (
        "https://raw.githubusercontent.com/ltgoslo/NorQuAD/main/data/evaluation"
        "/all/{}_dataset_flattened.json"
    )
    train_url = url.format("training")
    val_url = url.format("validation")
    test_url = url.format("test")

    # Download the NorQuAD dataset
    train_raw = requests.get(train_url).json()
    val_raw = requests.get(val_url).json()
    test_raw = requests.get(test_url).json()

    def convert_to_df(raw: dict) -> pd.DataFrame:
        records = [
            dict(
                id=dct["paragraphs"][0]["qas"][0]["id"],
                context=dct["paragraphs"][0]["context"],
                question=dct["paragraphs"][0]["qas"][0]["question"],
                answers=dict(
                    text=[
                        answer["text"]
                        for answer in dct["paragraphs"][0]["qas"][0]["answers"]
                    ],
                    answer_start=[
                        answer["answer_start"]
                        for answer in dct["paragraphs"][0]["qas"][0]["answers"]
                    ],
                ),
            )
            for dct in raw["data"]
        ]
        return pd.DataFrame.from_records(data=records)

    # Convert the splits to Pandas DataFrames
    train_df = convert_to_df(train_raw)
    val_df = convert_to_df(val_raw)
    test_df = convert_to_df(test_raw)

    # Ensure that the dataframes are indeed Pandas DataFrames
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Concatenate all the splits
    df = pd.concat([train_df, val_df, test_df])
    df.reset_index(drop=True, inplace=True)

    # Only work with samples where the context is not very large or small
    lengths = df.context.str.len()
    df = df[lengths.between(MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT)]

    # Only work with samples where the question is not very large or small
    lengths = df.question.str.len()
    df = df[lengths.between(MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION)]

    # Ensure that the `id` column is a string
    df["id"] = df["id"].astype(str)

    # Extract information on which examples contain an answer
    has_answer: pd.Series = df.answers.map(lambda dct: dct["text"][0] != "")

    # Only work with the questions having answers in the context
    df_with_answer: pd.DataFrame = df.loc[has_answer]

    # Create validation split
    val_size = 256
    val_df = df_with_answer.sample(n=val_size, random_state=4242)

    # Create test split
    test_size = 2048
    filtered_df = df.iloc[~df.index.isin(val_df.index)]
    test_df = filtered_df.sample(n=test_size, random_state=4242)

    # Create train split
    train_size = 1024
    filtered_df = filtered_df.iloc[~filtered_df.index.isin(test_df.index)]
    train_df = filtered_df.sample(n=train_size, random_state=4242)

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
    mini_dataset_id = "EuroEval/norquad-mini"

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
