# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the COPA-lv dataset from VTI-Data and upload to HF Hub."""

from collections import Counter

import pandas as pd
from constants import (
    CHOICES_MAPPING,
    MAX_NUM_CHARS_IN_INSTRUCTION,
    MAX_NUM_CHARS_IN_OPTION,
    MAX_REPETITIONS,
    MIN_NUM_CHARS_IN_INSTRUCTION,
    MIN_NUM_CHARS_IN_OPTION,
)
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi


def main() -> None:
    """Create the COPA-lv dataset and upload to HF Hub."""
    url_prefix = (
        "https://raw.githubusercontent.com/LUMII-AILab/VTI-Data/refs/heads/main/copa"
    )
    train_jsonl_url = f"{url_prefix}/mt/train.jsonl"
    val_jsonl_url = f"{url_prefix}/mt/val.jsonl"
    test_jsonl_url = f"{url_prefix}/post-edited/test.json"

    # Load and prepare the dataframes
    train_df = load_and_prepare_dataframe(url=train_jsonl_url)
    val_df = load_and_prepare_dataframe(url=val_jsonl_url)
    test_df = load_and_prepare_dataframe(url=test_jsonl_url)

    # Create dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/copa-lv"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def load_and_prepare_dataframe(url: str) -> pd.DataFrame:
    """Load and prepare the dataframe.

    Args:
        url:
            The url to the JSONL file.

    Returns:
        The prepared dataframe.
    """
    df = pd.read_json(url, lines=True)
    df = df.rename(
        columns=dict(premise="instruction", choice1="option_a", choice2="option_b")
    )

    # Strip texts
    df.instruction = df.instruction.str.strip()
    df.option_a = df.option_a.str.strip()
    df.option_b = df.option_b.str.strip()

    # Remove too short or too long samples
    df = df[
        (df.instruction.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
        & (df.instruction.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
        & (df.option_a.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_a.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.option_b.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_b.str.len() <= MAX_NUM_CHARS_IN_OPTION)
    ]

    # Remove overly repetitive samples
    df = df[
        ~df.instruction.apply(is_repetitive)
        & ~df.option_a.apply(is_repetitive)
        & ~df.option_b.apply(is_repetitive)
    ]
    assert isinstance(df, pd.DataFrame)

    # Create the text column with choices in Latvian
    choices_word = CHOICES_MAPPING["lv"]
    df["text"] = [
        row.instruction.replace("\n", " ").strip() + "\n"
        f"{choices_word}:\n"
        "a. " + row.option_a.replace("\n", " ").strip() + "\n"
        "b. " + row.option_b.replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]

    # Ensure label consistency
    df.label = df.label.map({0: "a", 1: "b"})

    # Only keep the required columns
    df = df[["text", "label"]]

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Shuffle the dataframe
    df = df.sample(frac=1.0, random_state=4242)
    assert isinstance(df, pd.DataFrame)

    return df


def is_repetitive(text: str) -> bool:
    """Check if the text is repetitive.

    Args:
        text:
            The text to check.

    Returns:
        Whether the text is repetitive.
    """
    max_repetitions = max(Counter(text.split()).values()) if text.split() else 0
    return max_repetitions > MAX_REPETITIONS


if __name__ == "__main__":
    main()
