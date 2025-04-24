# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the FoSent sentiment dataset and upload it to the HF Hub."""

import hashlib
import logging
from typing import Literal

import pandas as pd
from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("create_fosent")


def main() -> None:
    """Create the FoSent sentiment dataset and upload it to the HF Hub."""
    # Define the base download URL
    repo_id = "hafsteinn/faroese_sentiment_analysis"

    # Download the dataset
    dataset = load_dataset(path=repo_id, split="train", token=True)
    assert isinstance(dataset, Dataset)

    # Convert the dataset to a dataframe
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    column_mapping = {
        "News article": "news",
        "Selected Sentence": "sentence",
        "Sentence label - Annotator 1": "sentence_label_1",
        "Sentence label - Annotator 2": "sentence_label_2",
        "News label - Annotator 1": "news_label_1",
        "News label - Annotator 2": "news_label_2",
    }
    df.rename(columns=column_mapping, inplace=True)
    df.drop(
        columns=[col for col in df.columns if col not in column_mapping.values()],
        inplace=True,
    )

    df["news_id"] = df["news"].map(
        lambda x: x if x is None else hashlib.md5(string=x.encode()).hexdigest()
    )
    df["sentence_id"] = df["sentence"].map(
        lambda x: x if x is None else hashlib.md5(string=x.encode()).hexdigest()
    )

    # Create news dataframe
    news_df = df[["news_id", "news", "news_label_1", "news_label_2"]].copy()
    assert isinstance(news_df, pd.DataFrame)
    news_df = (
        news_df.rename(
            columns=dict(
                news_id="id",
                news="text",
                news_label_1="label_1",
                news_label_2="label_2",
            )
        )
        .drop_duplicates(subset="id")
        .reset_index(drop=True)
    )

    # Create sentence dataframe
    sentence_df = df[
        ["sentence_id", "news_id", "sentence", "sentence_label_1", "sentence_label_2"]
    ].copy()
    assert isinstance(sentence_df, pd.DataFrame)
    sentence_df = (
        sentence_df.rename(
            columns=dict(
                sentence_id="id",
                sentence="text",
                sentence_label_1="label_1",
                sentence_label_2="label_2",
            )
        )
        .drop_duplicates(subset="id")
        .reset_index(drop=True)
    )

    # Merge the labels
    news_df["label"] = news_df.apply(
        lambda row: merge_labels(label_1=row.label_1, label_2=row.label_2), axis=1
    )
    sentence_df["label"] = sentence_df.apply(
        lambda row: merge_labels(label_1=row.label_1, label_2=row.label_2), axis=1
    )

    news_df = (
        news_df.drop(columns=["label_1", "label_2"]).dropna().reset_index(drop=True)
    )
    sentence_df = (
        sentence_df.drop(columns=["label_1", "label_2"]).dropna().reset_index(drop=True)
    )

    # Select train/val/test news IDs
    all_news_ids = set(news_df.sample(frac=1, random_state=4242).id)
    assert all_news_ids == set(news_df.id) | set(sentence_df.news_id)
    assert len(all_news_ids) == 170

    # Create validation split
    val_size = 16
    val_news_ids = list(all_news_ids)[:val_size]
    val_df = pd.concat(
        objs=[
            news_df[news_df.id.isin(val_news_ids)].drop(columns=["id"]),
            sentence_df[sentence_df.news_id.isin(val_news_ids)].drop(
                columns=["id", "news_id"]
            ),
        ]
    ).sample(frac=1, random_state=4242)
    assert isinstance(val_df, pd.DataFrame)

    # Create train split
    train_size = 32
    train_news_ids = list(all_news_ids)[val_size : val_size + train_size]
    train_df = pd.concat(
        objs=[
            news_df[news_df.id.isin(train_news_ids)].drop(columns=["id"]),
            sentence_df[sentence_df.news_id.isin(train_news_ids)].drop(
                columns=["id", "news_id"]
            ),
        ]
    ).sample(frac=1, random_state=4242)
    assert isinstance(train_df, pd.DataFrame)

    # Create test split
    test_news_ids = list(all_news_ids)[val_size + train_size :]
    test_df = pd.concat(
        objs=[
            news_df[news_df.id.isin(test_news_ids)].drop(columns=["id"]),
            sentence_df[sentence_df.news_id.isin(test_news_ids)].drop(
                columns=["id", "news_id"]
            ),
        ]
    ).sample(frac=1, random_state=4242)
    assert isinstance(test_df, pd.DataFrame)

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Only work with samples where the document is not very large or small
    # We do it after we have made the splits to ensure that the dataset is minimally
    # affected.
    new_train_df = train_df.copy()
    new_train_df["text_len"] = new_train_df.text.str.len()
    new_train_df = new_train_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )
    new_val_df = val_df.copy()
    new_val_df["text_len"] = new_val_df.text.str.len()
    new_val_df = new_val_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )
    new_test_df = test_df.copy()
    new_test_df["text_len"] = new_test_df.text.str.len()
    new_test_df = new_test_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )

    dataset = DatasetDict(
        train=Dataset.from_pandas(new_train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(new_val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(new_test_df, split=Split.TEST),
    )

    dataset_id = "EuroEval/fosent"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def merge_labels(
    label_1: Literal[-1, 0, 1] | float, label_2: Literal[-1, 0, 1] | float
) -> Literal["negative", "neutral", "positive"] | None:
    """Merge two labels.

    This follows the following rules:
        - If one of the labels is missing, return the other label.
        - If both labels are missing, do not use the label.
        - If the labels are the same, return the label.
        - If the labels are adjacent, return the more extreme label.
        - Otherwise, do not use the label.

    Args:
        label_1:
            The first label.
        label_2:
            The second label.

    Returns:
        The merged label.
    """
    labels: list[Literal["negative", "neutral", "positive"]] = [
        "negative",
        "neutral",
        "positive",
    ]

    if label_1 != label_1 and label_2 == label_2:
        return labels[int(label_2) + 1]
    elif label_1 == label_1 and label_2 != label_2:
        return labels[int(label_1) + 1]
    elif label_1 != label_1 and label_2 != label_2:
        return None

    if label_1 == label_2:
        return labels[int(label_1) + 1]
    elif abs(label_1 - label_2) == 1:
        if max(label_1, label_2) == 1:
            return "positive"
        return "negative"

    return None


if __name__ == "__main__":
    main()
