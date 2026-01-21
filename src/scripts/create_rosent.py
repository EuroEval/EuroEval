# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "pandas==2.2.0",
#     "requests==2.32.5",
# ]
# ///

"""Create the RoSent Romanian sentiment dataset and upload to HF Hub."""

import json
import os
from typing import Literal

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

CACHE_FILE = "llm_sent_cache.json"


class Sentiment(BaseModel):
    """Sentiment classification."""

    sentiment: Literal["negative", "positive"] = Field(
        description="The sentiment of the text, either 'negative' or 'positive'"
    )


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


label_cache = load_cache()

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def main() -> None:
    """Create the RoSent Romanian sentiment dataset and upload to HF Hub."""
    # Load the dataset
    dataset_id = "dumitrescustefan/ro_sent"
    dataset = load_dataset(dataset_id)
    assert isinstance(dataset, DatasetDict)
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    train_df = process(df=train_df, n_samples=1500)
    test_df = process(df=test_df, n_samples=2500)

    # Make splits
    train_size = 1024
    val_size = 256
    test_size = 2048

    train_df_final = train_df.sample(n=train_size, random_state=4242)
    train_df_remaining = train_df[~train_df.index.isin(train_df_final.index)]
    val_df_final = train_df_remaining.sample(n=val_size, random_state=4242)
    test_df_final = test_df.sample(n=test_size, random_state=4242)

    # Reset indices
    train_df_final = train_df_final.reset_index(drop=True)
    val_df_final = val_df_final.reset_index(drop=True)
    test_df_final = test_df_final.reset_index(drop=True)

    assert isinstance(train_df_final, pd.DataFrame)
    assert isinstance(val_df_final, pd.DataFrame)
    assert isinstance(test_df_final, pd.DataFrame)

    # Create DatasetDict
    mini_dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df_final),
            "val": Dataset.from_pandas(val_df_final),
            "test": Dataset.from_pandas(test_df_final),
        }
    )

    # Push to HF Hub
    dataset_id = "EuroEval/ro-sent-mini"
    HfApi().delete_repo(repo_id=dataset_id, repo_type="dataset", missing_ok=True)
    mini_dataset.push_to_hub(dataset_id, private=True)


def process(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Process the dataframe to ensure required columns and rename.

    Args:
        df: The input dataframe.

    Returns:
        A dataframe with the required columns and renamed.
    """
    renames = {"sentence": "text"}
    df = df.rename(columns=renames)

    # Create uniform label distribution
    df = _create_uniform_label_distribution(df=df)

    # Use train_test_split with stratify to sample n_samples
    # across the label distribution. We don't want to prompt an LLM
    # on more samples than necessary.
    df, _ = train_test_split(
        df, train_size=n_samples, random_state=4242, stratify=df["label"]
    )

    label_mapping = {0: "negative", 1: "positive"}
    df["label"] = df["label"].map(label_mapping)

    # Classify text
    df["label_llm"] = df["text"].apply(_classify_text)

    # Compare labels
    df["label_comparison"] = df.apply(_compare_labels, axis=1)

    # Keep only rows where original and LLM labels are the same
    df = df[df["label_comparison"]]

    # Keep only text and label columns
    keep_columns = ["text", "label"]
    df = df[keep_columns]

    return df


def _classify_text(text: str) -> str:
    """Classify the text using GPT-4o model.

    Args:
        text: The text to classify.

    Returns:
        The sentiment of the text, either 'negative' or 'positive'
    """
    if text in label_cache:
        return label_cache[text]

    messages = [
        ChatCompletionUserMessageParam(
            role="user",
            content=(
                f"Classify the sentiment of the following text: <text>{text}</text>. "
                "Return either 'negative' or 'positive'."
            ),
        )
    ]
    completion = client.beta.chat.completions.parse(
        model="gpt-4o", messages=messages, response_format=Sentiment
    )
    parsed_response = completion.choices[0].message.parsed
    if parsed_response is None:
        raise ValueError("Parsed response is None")
    label = parsed_response.sentiment
    label_cache[text] = label
    save_cache(cache=label_cache)
    return label


def _compare_labels(row: pd.Series) -> bool:
    """Compare the labels from the RoSent model and the original dataset.

    Args:
        row: The row to compare the labels from the RoSent model and the
            original dataset.

    Returns:
        True if labels are the same, False if they are different.
    """
    return row["label"] == row["label_llm"]


def _create_uniform_label_distribution(
    df: pd.DataFrame, random_state: int = 4242
) -> pd.DataFrame:
    """Create a sampled dataset with a uniform label distribution.

    Args:
        df: The input dataframe with a 'label' column.
        random_state: The random state for reproducibility.

    Returns:
        A dataframe with a uniform label distribution.
    """
    # Separate each class
    classes = df["label"].unique()
    class_dfs = [df[df["label"] == label] for label in classes]

    # Find the size of the smallest class
    min_size = min(len(class_df) for class_df in class_dfs)

    # Resample each class to the size of the smallest class
    resampled_dfs = [
        resample(class_df, replace=False, n_samples=min_size, random_state=random_state)
        for class_df in class_dfs
    ]

    # Combine the resampled dataframes (keep original indices!)
    balanced_df = pd.concat(resampled_dfs, ignore_index=False)

    return balanced_df


if __name__ == "__main__":
    main()
