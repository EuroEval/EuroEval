# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the MultiLoKo-mini datasets and upload them to the HF Hub."""

import logging
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

logger = logging.getLogger(__name__)

LANGUAGES = ["nl", "en", "fr", "de", "it", "pt", "es", "sv"]

TARGET_TRAIN_SIZE = 1024
TARGET_VAL_SIZE = 256
TARGET_TEST_SIZE = 2048
VAL_PROPORTION = 0.1
TEST_PROPORTION = 0.6
MIN_SAMPLES_FOR_SPLIT = 10


def main() -> None:
    """Create the MultiLoKo-mini datasets and upload them to the HF Hub."""
    repo_id = "facebook/multiloko"

    for language in LANGUAGES:
        # Load the dataset - use the original-language subset for each language
        dataset = load_dataset(path=repo_id, name=language, token=True)
        assert isinstance(dataset, DatasetDict)

        # Convert the dataset splits to dataframes. MultiLoKo may only have a test
        # split, in which case we create train/val splits from it.
        split_names = list(dataset.keys())
        dfs = {split: dataset[split].to_pandas() for split in split_names}
        for split_df in dfs.values():
            assert isinstance(split_df, pd.DataFrame)

        # Combine all splits into a single dataframe
        df = pd.concat(list(dfs.values()), ignore_index=True)
        assert isinstance(df, pd.DataFrame)

        # Normalise column names. MultiLoKo uses 'question' and 'options' (a list of
        # strings) and 'answer' (0-based integer index of the correct option).
        if "question" in df.columns and "options" not in df.columns:
            # Some variants may use individual option columns
            for old, new in [
                ("option_1", "option_a"),
                ("option_2", "option_b"),
                ("option_3", "option_c"),
                ("option_4", "option_d"),
            ]:
                if old in df.columns:
                    df.rename(columns={old: new}, inplace=True)
        elif "question" in df.columns and "options" in df.columns:
            # Expand the list-typed 'options' column into individual option columns
            options_df = pd.DataFrame(
                df["options"].tolist(),
                columns=["option_a", "option_b", "option_c", "option_d"],
            )
            df = pd.concat([df.drop(columns=["options"]), options_df], axis=1)

        # Rename 'question' -> 'instruction' for consistency with EuroEval format
        if "question" in df.columns:
            df.rename(columns={"question": "instruction"}, inplace=True)

        # Validate that required columns are present
        required_cols = ["instruction", "option_a", "option_b", "option_c", "option_d"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            logger.warning(
                "Skipping language %s: missing columns %s", language, missing_cols
            )
            continue

        # Normalise the answer column. MultiLoKo may use an integer index (0-3) or a
        # letter ('a'-'d'). We convert everything to lowercase letters.
        if "answer" in df.columns:
            non_null = df["answer"].dropna()
            if not non_null.empty and pd.api.types.is_integer_dtype(df["answer"]):
                min_val = int(non_null.min())
                max_val = int(non_null.max())
                if min_val >= 1 and min_val <= max_val <= 4:
                    # 1-based indexing (values are in {1,2,3,4})
                    mapping = {1: "a", 2: "b", 3: "c", 4: "d"}
                else:
                    # Default to 0-based indexing (values are in {0,1,2,3})
                    mapping = {0: "a", 1: "b", 2: "c", 3: "d"}
                df["label"] = df["answer"].map(mapping)
            elif pd.api.types.is_object_dtype(df["answer"]):
                df["label"] = df["answer"].astype(str).str.strip().str.lower()
            else:
                df.rename(columns={"answer": "label"}, inplace=True)
            if "answer" in df.columns:
                df.drop(columns=["answer"], inplace=True)

        # Drop rows with invalid or missing labels
        df = df[df["label"].isin(list("abcd"))].copy()
        assert isinstance(df, pd.DataFrame)

        # Remove the samples with overly short or long texts or options
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

        # Build the 'text' column that EuroEval expects
        choices_label = CHOICES_MAPPING[language]
        df["text"] = [
            row.instruction.replace("\n", " ").strip()
            + f"\n{choices_label}:\n"
            + "a. "
            + str(row.option_a).replace("\n", " ").strip()
            + "\nb. "
            + str(row.option_b).replace("\n", " ").strip()
            + "\nc. "
            + str(row.option_c).replace("\n", " ").strip()
            + "\nd. "
            + str(row.option_d).replace("\n", " ").strip()
            for _, row in df.iterrows()
        ]

        # Keep only columns needed by EuroEval
        df = df[["text", "label"]]

        # Remove duplicates
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Check there are enough samples to create meaningful splits
        available = len(df)
        if available < MIN_SAMPLES_FOR_SPLIT:
            logger.warning(
                "Skipping language %s: only %d samples after filtering",
                language,
                available,
            )
            continue

        # Create splits (capped by available data)
        val_size = max(1, min(TARGET_VAL_SIZE, int(available * VAL_PROPORTION)))
        test_size = max(1, min(TARGET_TEST_SIZE, int(available * TEST_PROPORTION)))

        # Ensure we have enough samples for all three splits (at least 1 train sample)
        while available < val_size + test_size + 1 and val_size > 1:
            val_size -= 1
        while available < val_size + test_size + 1 and test_size > 1:
            test_size -= 1
        if available < val_size + test_size + 1:
            logger.warning(
                "Skipping language %s: not enough samples (%d) to create splits",
                language,
                available,
            )
            continue

        traintest_arr, val_arr = train_test_split(
            df, test_size=val_size, random_state=4242
        )
        traintest_df = pd.DataFrame(traintest_arr, columns=df.columns)
        val_df = pd.DataFrame(val_arr, columns=df.columns)

        remaining = len(traintest_df)
        test_size = min(test_size, remaining - 1)
        train_arr, test_arr = train_test_split(
            traintest_df, test_size=test_size, random_state=4242
        )
        train_df = pd.DataFrame(train_arr, columns=df.columns)
        test_df = pd.DataFrame(test_arr, columns=df.columns)

        train_size = min(TARGET_TRAIN_SIZE, len(train_df))
        train_df = train_df.sample(train_size, random_state=4242)

        # Reset indices
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # Collect into a DatasetDict
        dataset_out = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
                "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
                "test": Dataset.from_pandas(test_df, split=Split.TEST),
            }
        )

        # Build the EuroEval dataset ID
        dataset_id = f"EuroEval/multiloko-{language}-mini"

        # Remove existing dataset from the Hub if present, then push the new one
        HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
        dataset_out.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
