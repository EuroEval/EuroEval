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

LANGUAGES = ["nl", "en", "fr", "de", "it", "pt", "es", "sv"]

NUM_OPTIONS = 4
TARGET_TRAIN_SIZE = 1024
TARGET_VAL_SIZE = 256
TARGET_TEST_SIZE = 2048
VAL_PROPORTION = 0.1
TEST_PROPORTION = 0.6


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

        # Normalise the answer column. MultiLoKo may use an integer index (0-3) or a
        # letter ('a'-'d'). We convert everything to lowercase letters.
        if "answer" in df.columns:
            if pd.api.types.is_integer_dtype(df["answer"]):
                df["answer"] = df["answer"].map({0: "a", 1: "b", 2: "c", 3: "d"})
            elif pd.api.types.is_object_dtype(df["answer"]):
                df["answer"] = df["answer"].str.lower()
            df.rename(columns={"answer": "label"}, inplace=True)

        # Build the 'text' column that EuroEval expects
        choices_label = CHOICES_MAPPING[language]
        df["text"] = [
            row.instruction.replace("\n", " ").strip()
            + "\n"
            + f"{choices_label}:\n"
            + "a. "
            + str(row.option_a).replace("\n", " ").strip()
            + "\n"
            + "b. "
            + str(row.option_b).replace("\n", " ").strip()
            + "\n"
            + "c. "
            + str(row.option_c).replace("\n", " ").strip()
            + "\n"
            + "d. "
            + str(row.option_d).replace("\n", " ").strip()
            for _, row in df.iterrows()
        ]

        # Keep only columns needed by EuroEval
        df = df[["text", "label"]]

        # Remove samples with overly short/long texts or options
        df = df[
            (df.text.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
            & (df.text.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION + NUM_OPTIONS * MAX_NUM_CHARS_IN_OPTION)
        ]

        def is_repetitive(text: str) -> bool:
            """Return True if the text is repetitive."""
            max_repetitions = max(Counter(text.split()).values())
            return max_repetitions > MAX_REPETITIONS

        df = df[~df.text.apply(is_repetitive)]
        assert isinstance(df, pd.DataFrame)

        # Remove duplicates
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Create splits (capped by available data)
        available = len(df)
        test_size = min(TARGET_TEST_SIZE, int(available * TEST_PROPORTION))
        val_size = min(TARGET_VAL_SIZE, int(available * VAL_PROPORTION))

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
