# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "polars==1.31.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the MMLU mini Portuguese dataset and upload it to the HF Hub."""

from collections import Counter

import pandas as pd
import polars as pl
from constants import (
    MAX_NUM_CHARS_IN_INSTRUCTION,
    MAX_NUM_CHARS_IN_OPTION,
    MAX_REPETITIONS,
    MIN_NUM_CHARS_IN_INSTRUCTION,
    MIN_NUM_CHARS_IN_OPTION,
)
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from requests import HTTPError
from sklearn.model_selection import train_test_split


def valid_row(question: str, options: list) -> bool:
    """Return True if the row is valid."""
    if not (
        MIN_NUM_CHARS_IN_INSTRUCTION <= len(question) <= MAX_NUM_CHARS_IN_INSTRUCTION  # noqa: E501
    ):
        return False
    if not isinstance(options, list) or len(options) != 4:
        return False
    for option in options:
        if not (MIN_NUM_CHARS_IN_OPTION <= len(option) <= MAX_NUM_CHARS_IN_OPTION):  # noqa: E501
            return False
    return True


def is_repetitive(text: str) -> bool:
    """Return True if the text is repetitive."""
    max_repetitions = max(Counter(str(text).split()).values())
    return max_repetitions > MAX_REPETITIONS


def is_repetitive_options(options: list) -> bool:
    """Return True if any of the options is repetitive."""
    return any(is_repetitive(str(option)) for option in options)


def create_text_row(question: str, choices: list[str]) -> str:
    """Create a text row for the MMLU Portuguese dataset."""
    question_clean = question.replace("\n", " ").strip()
    text = question_clean + "\n" + "Opções:\n"

    letters = ["a", "b", "c", "d"]
    # Ensure we only take first 4 choices
    for i, choice in enumerate(choices[:4]):
        choice_text = choice.replace("\n", " ").strip()
        text += f"{letters[i]}. {choice_text}\n"

    return text.rstrip()  # Remove trailing newline


def load_and_clean_df(url: str) -> pl.DataFrame:
    """Load and clean the MMLU Portuguese dataset."""
    return (
        pl.read_ndjson(url)
        .with_columns(
            pl.col("id").str.split("/").list.get(0).alias("category"),
            pl.col("answer")
            .map_elements(
                lambda x: {0: "a", 1: "b", 2: "c", 3: "d"}[x],
                return_dtype=pl.String,  # noqa: E501
            )
            .alias("label"),
        )
        .filter(
            pl.struct(["question", "choices"]).map_elements(
                lambda row: valid_row(row["question"], row["choices"]),
                return_dtype=pl.Boolean,
            )
        )
        .filter(
            pl.col("question").map_elements(
                lambda x: not is_repetitive(x), return_dtype=pl.Boolean
            )
        )
        .filter(
            pl.col("choices").map_elements(
                lambda x: not is_repetitive_options(x), return_dtype=pl.Boolean
            )
        )
        .with_columns(
            pl.struct(["question", "choices"])
            .map_elements(
                lambda row: create_text_row(row["question"], row["choices"]),
                return_dtype=pl.String,
            )
            .alias("text")
        )
        .select(["text", "label", "category"])
    )


def main() -> None:
    """Create the MMLU Portuguese dataset and upload it to the HF Hub."""
    original_dataset_url = "LumiOpen/opengpt-x_mmlux"

    test_df = load_and_clean_df(
        f"hf://datasets/{original_dataset_url}/*PT-PT*test.jsonl"
    )
    print(f"Test samples: {len(test_df)}")

    val_df = load_and_clean_df(
        f"hf://datasets/{original_dataset_url}/*PT-PT*validation.jsonl"
    )
    print(f"Validation samples: {len(val_df)}")

    train_df = load_and_clean_df(
        f"hf://datasets/{original_dataset_url}/*PT-PT*dev.jsonl"
    )
    print(f"Train samples: {len(train_df)}")

    # Concatenate all splits
    df = pl.concat([test_df, val_df, train_df], how="vertical")
    print(f"Total samples: {len(df)}")

    # Remove duplicates
    df = df.unique(subset=["text"])
    print(f"Samples after deduplication: {len(df)}")

    # Convert to pandas for train_test_split
    df_pandas = df.to_pandas()

    # Create validation split
    val_size = 256
    traintest_arr, val_arr = train_test_split(
        df_pandas, test_size=val_size, random_state=4242, stratify=df_pandas.category
    )
    traintest_df = pd.DataFrame(traintest_arr, columns=df_pandas.columns)
    val_df = pd.DataFrame(val_arr, columns=df_pandas.columns)

    # Create test split
    test_size = 2048
    train_arr, test_arr = train_test_split(
        traintest_df,
        test_size=test_size,
        random_state=4242,
        stratify=traintest_df.category,
    )
    train_df = pd.DataFrame(train_arr, columns=df_pandas.columns)
    test_df = pd.DataFrame(test_arr, columns=df_pandas.columns)

    # Create train split
    train_size = 1024
    train_df = train_df.sample(train_size, random_state=4242)

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Final train samples: {len(train_df)}")
    print(f"Final val samples: {len(val_df)}")
    print(f"Final test samples: {len(test_df)}")

    # assert no intersection in text column
    assert len(set(train_df.text) & set(val_df.text)) == 0
    assert len(set(train_df.text) & set(test_df.text)) == 0
    assert len(set(val_df.text) & set(test_df.text)) == 0

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID for Portuguese
    dataset_id = "EuroEval/mmlu-pt-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=False)
    print(f"Dataset uploaded to {dataset_id}")


if __name__ == "__main__":
    main()
