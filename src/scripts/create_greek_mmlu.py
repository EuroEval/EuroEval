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
    """Create the GreekMMLU knowledge dataset and upload it to the HF Hub."""
    repo_id = "dascim/GreekMMLU"

    # Load the dataset
    dataset = load_dataset(path=repo_id)
    assert isinstance(dataset, DatasetDict)

    # Combine all available splits into a single dataframe
    dfs = []
    for split_name in dataset:
        split_df = dataset[split_name].to_pandas()
        assert isinstance(split_df, pd.DataFrame)
        dfs.append(split_df)
    df = pd.concat(dfs, ignore_index=True)

    # Process the dataframe
    df = process_df(df=df)

    # Create validation split
    val_size = 256
    traintest_df, val_df = train_test_split(
        df, test_size=val_size, random_state=4242, stratify=df.category
    )
    traintest_df = pd.DataFrame(traintest_df, columns=df.columns)
    val_df = pd.DataFrame(val_df, columns=df.columns)

    # Create test split
    test_size = 2048
    train_df, test_df = train_test_split(
        traintest_df,
        test_size=test_size,
        random_state=4242,
        stratify=traintest_df.category,
    )
    train_df = pd.DataFrame(train_df, columns=df.columns)
    test_df = pd.DataFrame(test_df, columns=df.columns)

    # Create train split
    train_size = 1024
    train_df = train_df.sample(train_size, random_state=4242)

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
        The processed DataFrame with `text`, `label`, and `category` columns.
    """
    # Rename columns to a standardised format
    rename_map: dict[str, str] = {}
    col_lower = {col.lower(): col for col in df.columns}

    for std_name, candidates in [
        ("question", ["question", "instruction"]),
        ("option_a", ["option_a", "a", "choice_a"]),
        ("option_b", ["option_b", "b", "choice_b"]),
        ("option_c", ["option_c", "c", "choice_c"]),
        ("option_d", ["option_d", "d", "choice_d"]),
        ("answer", ["answer", "label", "correct_answer"]),
        ("category", ["subject", "category", "topic", "subject_category"]),
    ]:
        for candidate in candidates:
            if candidate in col_lower:
                original = col_lower[candidate]
                if original != std_name:
                    rename_map[original] = std_name
                break

    df = df.rename(columns=rename_map)

    # Filter by text length
    df = df.loc[
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

    # Remove overly repetitive samples
    def is_repetitive(text: str) -> bool:
        """Return True if the text is repetitive."""
        max_repetitions = max(Counter(text.split()).values())
        return max_repetitions > MAX_REPETITIONS

    df = df.loc[
        ~df.question.apply(is_repetitive)
        & ~df.option_a.apply(is_repetitive)
        & ~df.option_b.apply(is_repetitive)
        & ~df.option_c.apply(is_repetitive)
        & ~df.option_d.apply(is_repetitive)
    ]
    assert isinstance(df, pd.DataFrame)

    # Make the label lowercase
    df["label"] = df.answer.str.lower()

    # Make a `text` column with all the options in it
    df["text"] = [
        row.question.replace("\n", " ").strip() + "\n"
        f"{CHOICES_MAPPING['el']}:\n"
        "a. " + row.option_a.replace("\n", " ").strip() + "\n"
        "b. " + row.option_b.replace("\n", " ").strip() + "\n"
        "c. " + row.option_c.replace("\n", " ").strip() + "\n"
        "d. " + row.option_d.replace("\n", " ").strip()
        for _, row in df.iterrows()
    ]

    # Only keep the `text`, `label`, and `category` columns
    df = df[["text", "label", "category"]]

    # Remove duplicates
    df = df.drop_duplicates(inplace=False)
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    main()
