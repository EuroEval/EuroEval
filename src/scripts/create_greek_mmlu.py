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

"""Create the GreekMMLU knowledge dataset and upload it to the HF Hub."""

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
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


def main() -> None:
    """Create the GreekMMLU knowledge dataset and upload it to the HF Hub."""
    repo_id = "dascim/GreekMMLU"

    # Load the "All" subset
    dataset = load_dataset(path=repo_id, name="All")
    assert isinstance(dataset, DatasetDict)

    # Use 'dev' split as training source
    train_source_df = dataset["dev"].to_pandas()
    assert isinstance(train_source_df, pd.DataFrame)

    # Use 'test' split as source for val/test splits
    test_source_df = dataset["test"].to_pandas()
    assert isinstance(test_source_df, pd.DataFrame)

    # Process the dataframes
    train_source_df = process_df(df=train_source_df)
    test_source_df = process_df(df=test_source_df)

    # Sample val and test splits from the test source, stratified by subject
    val_size = 256
    test_size = 2048
    val_df, remaining_df = train_test_split(
        test_source_df,
        train_size=val_size,
        random_state=4242,
        stratify=test_source_df["subject"],
    )
    val_df = pd.DataFrame(val_df, columns=test_source_df.columns)
    remaining_df = pd.DataFrame(remaining_df, columns=test_source_df.columns)

    test_df, _ = train_test_split(
        remaining_df,
        train_size=test_size,
        random_state=4242,
        stratify=remaining_df["subject"],
    )
    test_df = pd.DataFrame(test_df, columns=test_source_df.columns)

    # Use the full dev split as training (it has ~215 samples after filtering)
    train_df = train_source_df.sample(
        n=min(1024, len(train_source_df)), random_state=4242
    )

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    final_dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(test_df, split=Split.TEST),
        }
    )

    dataset_id = "EuroEval/greek-mmlu-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    final_dataset.push_to_hub(dataset_id, private=True)


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """Process the GreekMMLU dataframe into EuroEval format.

    Args:
        df: The input DataFrame from the GreekMMLU dataset.

    Returns:
        The processed DataFrame with `text`, `label`, and `subject` columns.
    """
    label_str = "abcdefghijklmnopqrstuvwxyz"

    # Map the integer answer index to a letter label
    df["label"] = df["answer"].map(lambda x: label_str[int(x)])

    # Filter by text length
    df = df.loc[
        (df["question"].str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
        & (df["question"].str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
        & df["choices"].map(
            lambda opts: (
                min(len(opt) for opt in opts) >= MIN_NUM_CHARS_IN_OPTION
                and max(len(opt) for opt in opts) <= MAX_NUM_CHARS_IN_OPTION
            )
        )
    ]

    # Remove overly repetitive samples
    def is_repetitive(text: str) -> bool:
        """Return True if the text is repetitive.

        Args:
            text: The text to check for repetitions.

        Returns:
            True if any word appears more than MAX_REPETITIONS times,
            False otherwise.
        """
        max_repetitions = max(Counter(text.split()).values())
        return max_repetitions > MAX_REPETITIONS

    df = df.loc[
        ~df["question"].apply(is_repetitive)
        & ~df["choices"].map(lambda opts: any(is_repetitive(opt) for opt in opts))
    ]
    assert isinstance(df, pd.DataFrame)

    # Make a `text` column with all the options in it
    df["text"] = [
        row["question"].replace("\n", " ").strip()
        + "\n"
        + f"{CHOICES_MAPPING['el']}:\n"
        + "\n".join(
            f"{char}. {opt.replace(chr(10), ' ').strip()}"
            for char, opt in zip(label_str, row["choices"])
        )
        for _, row in df.iterrows()
    ]

    # Only keep the `text`, `label`, and `subject` columns
    df = df[["text", "label", "subject"]]

    # Remove duplicates
    df = df.drop_duplicates(inplace=False)
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    main()
