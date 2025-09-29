# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the LT-History knowledge dataset and upload it to the HF Hub."""

from collections import Counter

import pandas as pd
import requests
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
    """Create the Lithuanian History knowledge dataset and upload it to the HF Hub."""
    # Download the JSON data from GitHub
    url = (
        "https://raw.githubusercontent.com/OpenBabylon/NoDaLiDa2025-LT-History-Eval"
        "/refs/heads/main/lit_data.json"
    )
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # Convert to DataFrame
    rows = []
    for item in data:
        # Extract question stem
        question_stem = item["question"]["stem"]

        # Extract choices
        choices = item["question"]["choices"]
        choice_texts = {choice["label"]: choice["text"] for choice in choices}

        # Get answer key and convert to lowercase
        answer_key = item["answerKey"].lower()

        # Create row
        row = {
            "id": item["id"],
            "instruction": question_stem,
            "option_a": choice_texts.get("A", ""),
            "option_b": choice_texts.get("B", ""),
            "option_c": choice_texts.get("C", ""),
            "option_d": choice_texts.get("D", ""),
            "label": answer_key,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Remove samples with null values in any option columns
    df = df[
        df["option_a"].notnull()
        & df["option_b"].notnull()
        & df["option_c"].notnull()
        & df["option_d"].notnull()
    ]

    # Remove samples with overly short or long texts
    df = df[
        (df.instruction.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
        & (df.instruction.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
        & (df.option_a.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_a.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.option_b.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_b.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.option_c.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_c.str.len() <= MAX_NUM_CHARS_IN_OPTION)
        & (df.option_d.str.len() >= MIN_NUM_CHARS_IN_OPTION)
        & (df.option_d.str.len() <= MAX_NUM_CHARS_IN_OPTION)
    ]

    def is_repetitive(text: str) -> bool:
        """Return True if the text is repetitive."""
        max_repetitions = max(Counter(text.split()).values())
        return max_repetitions > MAX_REPETITIONS

    # Remove overly repetitive samples
    df = df[
        ~df.instruction.apply(is_repetitive)
        & ~df.option_a.apply(is_repetitive)
        & ~df.option_b.apply(is_repetitive)
        & ~df.option_c.apply(is_repetitive)
        & ~df.option_d.apply(is_repetitive)
    ]

    # Create the `text` column with all options formatted
    df["text"] = [
        row.instruction.replace("\n", " ").strip() + "\n"
        f"{CHOICES_MAPPING['lt']}:\n"
        "a. " + row.option_a.replace("\n", " ").strip() + "\n"
        "b. " + row.option_b.replace("\n", " ").strip() + "\n"
        "c. " + row.option_c.replace("\n", " ").strip() + "\n"
        "d. " + row.option_d.replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]

    # Keep only the required columns for EuroEval format
    df = df[["text", "label"]]
    assert isinstance(df, pd.DataFrame)

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create train and test splits
    train_size = 64
    val_size = 32
    train_df = df.sample(n=train_size, random_state=42)
    df.drop(index=train_df.index.tolist(), inplace=True)
    val_df = df.sample(n=val_size, random_state=42)
    test_df = df.drop(index=val_df.index.tolist())

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary (no train split)
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/lt-history"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
