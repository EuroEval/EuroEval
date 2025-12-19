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

"""Create the Global-MMLU knowledge datasets and upload them to the HF Hub."""

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


def main() -> None:
    """Create the Global-MMLU knowledge datasets."""
    # Define the dataset ID
    repo_id = "CohereLabs/Global-MMLU"

    languages = ["uk", "el", "ro", "sq"]

    for language in languages:
        try:
            # Load the dataset
            dataset = load_dataset(path=repo_id, name=language)
            lite = False
        except ValueError:
            # load the lite dataset
            repo_id = "CohereLabs/Global-MMLU-lite"
            dataset = load_dataset(path=repo_id, name=language)
            lite = True
        assert isinstance(dataset, DatasetDict)

        # Convert the dataset to dataframes
        val_df = dataset["dev"].to_pandas()
        test_df = dataset["test"].to_pandas()
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

        # Process the dataframes
        val_df = process_split(df=val_df, language=language)
        test_df = process_split(df=test_df, language=language)

        if lite:
            train_df, val_df, test_df = make_mmlu_lite_splits(
                val_df=val_df, test_df=test_df
            )
        else:
            train_df, test_df, val_df = make_mmlu_splits(val_df=val_df, test_df=test_df)

        # Collect datasets in a dataset dictionary
        dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
                "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
                "test": Dataset.from_pandas(test_df, split=Split.TEST),
            }
        )

        # Create dataset ID
        lite_suffix = "-lite" if lite else ""
        mini_suffix = "-mini" if not lite else ""
        dataset_id = f"EuroEval/global-mmlu{lite_suffix}-{language}{mini_suffix}"
        # Remove the dataset from Hugging Face Hub if it already exists
        HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

        # Push the dataset to the Hugging Face Hub
        dataset.push_to_hub(dataset_id, private=True)


def make_mmlu_lite_splits(
    val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make splits for a lite dataset.

    Args:
        val_df: The validation dataframe.
        test_df: The test dataframe.

    Returns:
        The final training, validation, and test dataframes.
    """
    val_size = 64
    train_size = 128
    train_df_final, test_df = train_test_split(
        test_df, train_size=train_size, random_state=4242, stratify=test_df.category
    )

    val_df_final = val_df.sample(n=val_size, random_state=4242, replace=False)

    # Use remaining val samples in the test split
    remaining_val_df = val_df[~val_df.index.isin(val_df_final.index)]
    test_df_final = pd.concat([test_df, remaining_val_df], ignore_index=True)

    # Reset the index
    train_df_final = train_df_final.reset_index(drop=True)
    val_df_final = val_df_final.reset_index(drop=True)
    test_df_final = test_df_final.reset_index(drop=True)

    return train_df_final, val_df_final, test_df_final


def make_mmlu_splits(
    val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make splits for a full dataset.

    Args:
        val_df: The validation dataframe.
        test_df: The test dataframe.

    Returns:
        The final training, validation, and test dataframes.
    """
    val_size = 256
    val_df_final = val_df.sample(n=val_size, random_state=4242, replace=False)

    train_size = 1024
    test_size = 2048

    train_df_final, test_df = train_test_split(
        test_df, train_size=train_size, random_state=4242, stratify=test_df.category
    )

    test_df_final = test_df.sample(n=test_size, random_state=4242, replace=False)

    # Reset the index
    train_df_final = train_df_final.reset_index(drop=True)
    val_df_final = val_df_final.reset_index(drop=True)
    test_df_final = test_df_final.reset_index(drop=True)
    return train_df_final, test_df_final, val_df_final


def process_split(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """Process a split of the dataset.

    Args:
        df: The input DataFrame.

    Returns:
        The processed DataFrame.
    """
    df["label"] = df["answer"].str.lower()
    df["category"] = df["subject_category"]
    df = filter_by_length(df=df)
    df = filter_repetitive(df=df)
    df = add_text_column(df=df, language=language)
    df = df.loc[:, ["text", "label", "category"]]
    df = df.drop_duplicates(inplace=False)
    df = df.reset_index(drop=True)
    return df


def filter_by_length(df: pd.DataFrame) -> pd.DataFrame:
    """Remove samples with overly short or long texts.

    Args:
        df: The input DataFrame.

    Returns:
        The filtered DataFrame with only rows of acceptable length.
    """
    return df.loc[
        (df.question.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
        & (df.question.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
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
    """Check if a text is repetitive.

    Args:
        text: The input text string.

    Returns:
        True if repetitive, False otherwise.
    """
    max_repetitions = max(Counter(text.split()).values())
    return max_repetitions > MAX_REPETITIONS


def filter_repetitive(df: pd.DataFrame) -> pd.DataFrame:
    """Remove overly repetitive samples.

    Args:
        df: The input DataFrame.

    Returns:
        The filtered DataFrame without overly repetitive texts.
    """
    return df.loc[
        ~df.question.apply(is_repetitive)
        & ~df.option_a.apply(is_repetitive)
        & ~df.option_b.apply(is_repetitive)
        & ~df.option_c.apply(is_repetitive)
        & ~df.option_d.apply(is_repetitive)
    ]


def add_text_column(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """Make a `text` column with all the options.

    Args:
        df: The input DataFrame.

    Returns:
        The DataFrame with the added `text` column.
    """
    df["text"] = [
        row.question.replace("\n", " ").strip() + "\n"
        f"{CHOICES_MAPPING[language]}:\n"
        "a. " + row.option_a.replace("\n", " ").strip() + "\n"
        "b. " + row.option_b.replace("\n", " ").strip() + "\n"
        "c. " + row.option_c.replace("\n", " ").strip() + "\n"
        "d. " + row.option_d.replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]
    return df


if __name__ == "__main__":
    main()
