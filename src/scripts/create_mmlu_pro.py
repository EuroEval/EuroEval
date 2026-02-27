# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the MMLU-Pro-mini dataset and upload it to the HF Hub."""

from collections import Counter

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

from .constants import (
    CHOICES_MAPPING,
    MAX_NUM_CHARS_IN_INSTRUCTION,
    MAX_NUM_CHARS_IN_OPTION,
    MAX_REPETITIONS,
    MIN_NUM_CHARS_IN_INSTRUCTION,
    MIN_NUM_CHARS_IN_OPTION,
)

NUM_OPTIONS = 10
OPTION_LABELS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]


def main() -> None:
    """Create the MMLU-Pro-mini dataset and upload it to the HF Hub."""
    # Load the MMLU-Pro dataset
    dataset = load_dataset(path="TIGER-Lab/MMLU-Pro", token=True)
    assert isinstance(dataset, DatasetDict)

    # Concatenate all available splits
    dfs = []
    for split_name in dataset:
        split_df = dataset[split_name].to_pandas()
        assert isinstance(split_df, pd.DataFrame)
        dfs.append(split_df)
    df = pd.concat(dfs, ignore_index=True)

    # Rename the columns
    df.rename(columns=dict(question="instruction"), inplace=True)

    # Make the answer lowercase
    df["label"] = df["answer"].str.lower()

    # Only keep questions with exactly NUM_OPTIONS options
    df = df[df["options"].apply(len) == NUM_OPTIONS]

    # Expand the options list into individual columns
    for i, letter in enumerate(OPTION_LABELS):
        df[f"option_{letter}"] = df["options"].apply(lambda opts, idx=i: opts[idx])

    # Remove the samples with overly short or long texts
    option_cols = [f"option_{letter}" for letter in OPTION_LABELS]
    length_filter = (
        df.instruction.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION
    ) & (df.instruction.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
    for col in option_cols:
        length_filter &= (df[col].str.len() >= MIN_NUM_CHARS_IN_OPTION) & (
            df[col].str.len() <= MAX_NUM_CHARS_IN_OPTION
        )
    df = df[length_filter]

    def is_repetitive(text: str) -> bool:
        """Return True if the text is repetitive."""
        max_repetitions = max(Counter(text.split()).values())
        return max_repetitions > MAX_REPETITIONS

    # Remove overly repetitive samples
    repetition_filter = ~df.instruction.apply(is_repetitive)
    for col in option_cols:
        repetition_filter &= ~df[col].apply(is_repetitive)
    df = df[repetition_filter]
    assert isinstance(df, pd.DataFrame)

    # Make a `text` column with all the options in it
    lang = "en"
    df["text"] = [
        row.instruction.replace("\n", " ").strip()
        + "\n"
        + f"{CHOICES_MAPPING[lang]}:\n"
        + "\n".join(
            f"{letter}. " + row[f"option_{letter}"].replace("\n", " ").strip()
            for letter in OPTION_LABELS
        )
        for _, row in df.iterrows()
    ]

    # Only keep the `text`, `label` and `category` columns
    df = df[["text", "label", "category"]]

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create validation split
    val_size = 256
    traintest_arr, val_arr = train_test_split(
        df, test_size=val_size, random_state=4242, stratify=df.category
    )
    traintest_df = pd.DataFrame(traintest_arr, columns=df.columns)
    val_df = pd.DataFrame(val_arr, columns=df.columns)

    # Create test split
    test_size = 2048
    train_arr, test_arr = train_test_split(
        traintest_df,
        test_size=test_size,
        random_state=4242,
        stratify=traintest_df.category,
    )
    train_df = pd.DataFrame(train_arr, columns=df.columns)
    test_df = pd.DataFrame(test_arr, columns=df.columns)

    # Create train split
    train_size = min(1024, len(train_df))
    train_df = train_df.sample(train_size, random_state=4242)

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    # Create dataset ID
    dataset_id = "EuroEval/mmlu-pro-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
