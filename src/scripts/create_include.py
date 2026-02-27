# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the INCLUDE knowledge datasets and upload them to the HF Hub."""

from collections import Counter

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from sklearn.model_selection import train_test_split

from .constants import (
    CHOICES_MAPPING,
    MAX_NUM_CHARS_IN_INSTRUCTION,
    MAX_NUM_CHARS_IN_OPTION,
    MAX_REPETITIONS,
    MIN_NUM_CHARS_IN_INSTRUCTION,
    MIN_NUM_CHARS_IN_OPTION,
)

# Mapping from ISO language code to full language name used in INCLUDE dataset
INCLUDE_LANGUAGE_MAPPING: dict[str, str] = {
    "sq": "Albanian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "nl": "Dutch",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "it": "Italian",
    "lt": "Lithuanian",
    "pl": "Polish",
    "pt": "Portuguese",
    "sr": "Serbian",
    "es": "Spanish",
    "uk": "Ukrainian",
}


def main() -> None:
    """Create the INCLUDE knowledge datasets."""
    repo_id = "CohereLabs/include-base-44"

    for lang_code, lang_name in INCLUDE_LANGUAGE_MAPPING.items():
        # Load the dataset for this language
        dataset = load_dataset(path=repo_id, name=lang_name)
        assert isinstance(dataset, DatasetDict)

        # Use 'validation' split as training data and 'test' split for val/test
        train_source_df = dataset["validation"].to_pandas()
        test_source_df = dataset["test"].to_pandas()
        assert isinstance(train_source_df, pd.DataFrame)
        assert isinstance(test_source_df, pd.DataFrame)

        # Process the dataframes
        train_source_df = process_split(df=train_source_df, lang_code=lang_code)
        test_source_df = process_split(df=test_source_df, lang_code=lang_code)

        train_df, val_df, test_df = make_splits(
            train_df=train_source_df, test_df=test_source_df
        )

        # Collect datasets in a dataset dictionary
        dataset_out = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df, split=Split.TRAIN),
                "val": Dataset.from_pandas(val_df, split=Split.VALIDATION),
                "test": Dataset.from_pandas(test_df, split=Split.TEST),
            }
        )

        dataset_id = f"EuroEval/include-{lang_code}-mini"

        # Push the dataset to the Hugging Face Hub
        dataset_out.push_to_hub(dataset_id, private=True)


def make_splits(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make train/val/test splits from the validation and test sources.

    The full original validation split becomes the training split (capped at 1,024
    samples). 256 samples are drawn from the original test split for the EuroEval
    validation split, and a further 2,048 samples are drawn from the remainder for the
    EuroEval test split. Both samples are stratified by the 'subject' column.

    Args:
        train_df: The processed validation dataframe (used as training source).
        test_df: The processed test dataframe (used as val/test source).

    Returns:
        The final training, validation, and test dataframes.
    """
    val_size = 256
    test_size = 2048

    stratify_col = test_df["subject"] if "subject" in test_df.columns else None

    # Sample val split from test source, stratified by subject where possible
    n_val = min(val_size, len(test_df))
    try:
        val_df_final, remaining_df = train_test_split(
            test_df,
            train_size=n_val,
            random_state=4242,
            stratify=stratify_col,
        )
    except ValueError:
        val_df_final = test_df.sample(n=n_val, random_state=4242, replace=False)
        remaining_df = test_df[~test_df.index.isin(val_df_final.index)]

    # Sample test split from remaining, stratified by subject where possible
    remaining_stratify = (
        remaining_df["subject"] if "subject" in remaining_df.columns else None
    )
    n_test = min(test_size, max(0, len(remaining_df) - 1))
    try:
        test_df_final, _ = train_test_split(
            remaining_df,
            train_size=n_test,
            random_state=4242,
            stratify=remaining_stratify,
        )
    except ValueError:
        test_df_final = remaining_df.sample(
            n=min(test_size, len(remaining_df)), random_state=4242, replace=False
        )

    # Use the full validation source as training (capped at 1,024)
    train_df_final = train_df.sample(
        n=min(1024, len(train_df)), random_state=4242, replace=False
    )

    # Reset the index
    train_df_final = train_df_final.reset_index(drop=True)
    val_df_final = val_df_final.reset_index(drop=True)
    test_df_final = test_df_final.reset_index(drop=True)

    return train_df_final, val_df_final, test_df_final


def process_split(df: pd.DataFrame, lang_code: str) -> pd.DataFrame:
    """Process a split of the dataset.

    Args:
        df: The input DataFrame.
        lang_code: The ISO 639-1 language code.

    Returns:
        The processed DataFrame.
    """
    df["label"] = df["answer"].str.lower()
    df = filter_by_length(df=df)
    df = filter_repetitive(df=df)
    df = add_text_column(df=df, lang_code=lang_code)
    keep_cols = ["text", "label"]
    if "subject" in df.columns:
        keep_cols.append("subject")
    df = df.loc[:, keep_cols]
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


def add_text_column(df: pd.DataFrame, lang_code: str) -> pd.DataFrame:
    """Make a `text` column with the question (without options).

    Args:
        df: The input DataFrame.
        lang_code: The ISO 639-1 language code.

    Returns:
        The DataFrame with the added `text` column.
    """
    df["text"] = [
        row.question.replace("\n", " ").strip() + "\n"
        f"{CHOICES_MAPPING[lang_code]}:\n"
        "a. " + row.option_a.replace("\n", " ").strip() + "\n"
        "b. " + row.option_b.replace("\n", " ").strip() + "\n"
        "c. " + row.option_c.replace("\n", " ").strip() + "\n"
        "d. " + row.option_d.replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]
    return df


if __name__ == "__main__":
    main()
