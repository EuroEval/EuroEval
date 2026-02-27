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

"""Create the MultiNRC datasets and upload them to the HF Hub."""

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

LANGUAGES = ["en", "es", "fr"]

LANGUAGE_SUBSET_MAPPING = {"en": "english", "es": "spanish", "fr": "french"}


def main() -> None:
    """Create the MultiNRC datasets and upload them to the HF Hub."""
    repo_id = "ScaleAI/MultiNRC"

    for language in LANGUAGES:
        subset = LANGUAGE_SUBSET_MAPPING[language]

        # Load the dataset
        dataset = load_dataset(path=repo_id, name=subset, token=True)
        assert isinstance(dataset, DatasetDict)

        # Concatenate all splits into a single dataframe
        dfs = []
        for split_name in dataset.keys():
            split_df = dataset[split_name].to_pandas()
            assert isinstance(split_df, pd.DataFrame)
            dfs.append(split_df)
        df = pd.concat(dfs, ignore_index=True)

        # Rename the columns to EuroEval format
        df.rename(
            columns=dict(
                question="instruction",
                A="option_a",
                B="option_b",
                C="option_c",
                D="option_d",
                answer="label",
            ),
            inplace=True,
        )

        # Normalise the label column to lowercase letters (a, b, c, d)
        df.label = df.label.str.strip().str.lower()

        # Remove the samples with overly short or long texts
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
        assert isinstance(df, pd.DataFrame)

        # Make a `text` column with all the options in it
        df["text"] = [
            row.instruction.replace("\n", " ").strip() + "\n"
            f"{CHOICES_MAPPING[language]}:\n"
            "a. " + row.option_a.replace("\n", " ").strip() + "\n"
            "b. " + row.option_b.replace("\n", " ").strip() + "\n"
            "c. " + row.option_c.replace("\n", " ").strip() + "\n"
            "d. " + row.option_d.replace("\n", " ").strip()
            for _, row in df.iterrows()
        ]

        # Only keep the `text` and `label` columns
        df = df[["text", "label"]]

        # Remove duplicates
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Create validation split
        val_size = min(256, len(df) // 8)
        traintest_arr, val_arr = train_test_split(
            df, test_size=val_size, random_state=4242
        )
        traintest_df = pd.DataFrame(traintest_arr, columns=df.columns)
        val_df = pd.DataFrame(val_arr, columns=df.columns)

        # Create test split
        test_size = min(2048, len(traintest_df) * 2 // 3)
        train_arr, test_arr = train_test_split(
            traintest_df, test_size=test_size, random_state=4242
        )
        train_df = pd.DataFrame(train_arr, columns=df.columns)
        test_df = pd.DataFrame(test_arr, columns=df.columns)

        # Create train split
        train_size = min(1024, len(train_df))
        if len(train_df) > train_size:
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
        dataset_id = f"EuroEval/multinrc-{language}"

        # Remove the dataset from Hugging Face Hub if it already exists
        HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

        # Push the dataset to the Hugging Face Hub
        dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
