# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Hungarian summarisation dataset based on hun-sum-chatml-5k."""

import json
import os

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import BaseModel

from .constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE

load_dotenv()

CACHE_FILE = "summary_cache.json"


class SummaryValidation(BaseModel):
    """Structured output for the summary validation.

    Args:
        is_valid_summary: True if the summary aligns with the text, False otherwise.
    """

    is_valid_summary: bool


def load_cache() -> dict:
    """Load cache from CACHE_FILE if it exists."""
    try:
        with open(CACHE_FILE, "r") as cache_file:
            return json.load(cache_file)
    except FileNotFoundError:
        return {}


def save_cache(cache: dict) -> None:
    """Save cache to CACHE_FILE."""
    with open(CACHE_FILE, "w") as cache_file:
        json.dump(cache, cache_file, indent=4)


summary_cache = load_cache()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def main() -> None:
    """Create the Hungarian summarisation mini dataset and upload to HF Hub."""
    dataset_id = "ariel-ml/hun-sum-chatml-5k"

    dataset = load_dataset(dataset_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # The dataset has splits: train, val, test
    # Each sample has: title, lead, article
    # We want: text = "{title}\n\n{article}", target_text = lead

    def make_columns(sample: dict) -> dict:
        """Map the dataset to have the text and target_text columns.

        Args:
            sample: A sample from the dataset.

        Returns:
            A sample with the text and target_text columns.
        """
        sample["text"] = f"{sample['title']}\n\n{sample['article']}"
        sample["target_text"] = sample["lead"]
        return sample

    dataset = dataset.map(make_columns)

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    train_df = process(df=train_df)
    val_df = process(df=val_df)
    test_df = process(df=test_df)

    train_df_final, val_df_final, test_df_final = create_splits(
        train_df=train_df, val_df=val_df, test_df=test_df
    )

    # Collect datasets in a dataset dictionary
    mini_dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df_final, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df_final, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df_final, split=Split.TEST),
        }
    )

    # Create dataset ID
    mini_dataset_id = "EuroEval/hun-sum-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(mini_dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    mini_dataset.push_to_hub(mini_dataset_id, private=True)


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe.

    Args:
        df: The dataframe to process.

    Returns:
        The processed dataframe.
    """
    # Validate samples using an LLM
    df["is_valid_summary"] = df.apply(_text_summary_alignment, axis=1)
    df = df.loc[df["is_valid_summary"]]

    keep_columns = ["text", "target_text"]
    df = df[keep_columns]

    # Only work with samples where the text is not very large or small
    lengths = df.text.str.len()
    lower_bound = MIN_NUM_CHARS_IN_ARTICLE
    upper_bound = MAX_NUM_CHARS_IN_ARTICLE
    df = df[lengths.between(lower_bound, upper_bound)]
    df = df.reset_index(drop=True)
    return df


def _text_summary_alignment(row: pd.Series) -> bool:
    """Check if the summary aligns with the text using an LLM, with caching.

    Args:
        row: A row from the dataframe.

    Returns:
        True if the summary aligns with the text, False otherwise.
    """
    text = row["text"]
    summary = row["target_text"]
    if summary in summary_cache:
        return summary_cache[summary]

    messages: list[ChatCompletionUserMessageParam] = list()
    user_message = ChatCompletionUserMessageParam(
        role="user",
        content=(
            f"Does the summary <summary>{summary}</summary> align with the text "
            f"<text>{text}</text> and represent a true summary?"
        ),
    )
    messages.append(user_message)
    completion = client.beta.chat.completions.parse(
        model="gpt-4o", messages=messages, response_format=SummaryValidation
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Parsed response is None")
    is_valid_summary = parsed.is_valid_summary

    # Cache the result
    summary_cache[summary] = is_valid_summary
    save_cache(cache=summary_cache)

    return is_valid_summary


def create_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create splits.

    Args:
        train_df: The training dataframe.
        val_df: The validation dataframe.
        test_df: The test dataframe.

    Returns:
        The final training, validation, and test dataframes.
    """
    # Split sizes
    train_size = 1024
    val_size = 256
    test_size = 2048

    # Create train split
    train_df_final = train_df.sample(n=train_size, random_state=4242)
    train_df_remaining = train_df[~train_df.index.isin(train_df_final.index)]

    # Create validation split
    n_missing_val_samples = val_size - len(val_df)
    val_df_additional = train_df_remaining.sample(
        n=n_missing_val_samples, random_state=4242
    )
    val_df_final = pd.concat([val_df, val_df_additional], ignore_index=True)
    train_df_remaining = train_df_remaining.loc[
        ~train_df_remaining.index.isin(val_df_additional.index)
    ]

    # Create test split
    n_missing_test_samples = test_size - len(test_df)
    test_df_additional = train_df_remaining.sample(
        n=n_missing_test_samples, random_state=4242
    )
    test_df_final = pd.concat([test_df, test_df_additional], ignore_index=True)

    # Reset indices
    train_df_final = train_df_final.reset_index(drop=True)
    val_df_final = val_df_final.reset_index(drop=True)
    test_df_final = test_df_final.reset_index(drop=True)

    assert isinstance(train_df_final, pd.DataFrame)
    assert isinstance(val_df_final, pd.DataFrame)
    assert isinstance(test_df_final, pd.DataFrame)

    return train_df_final, val_df_final, test_df_final


if __name__ == "__main__":
    main()
