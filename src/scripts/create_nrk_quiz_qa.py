# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==2.15.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "scikit-learn<1.6.0",
# ]
# ///

"""Create the NRK-Quiz-QA-mini dataset and upload them to the HF Hub."""

import warnings
from collections import Counter

import pandas as pd
from constants import (
    MAX_NUM_CHARS_IN_INSTRUCTION,
    MAX_REPETITIONS,
    MIN_NUM_CHARS_IN_INSTRUCTION,
)
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from pandas.errors import SettingWithCopyWarning
from requests import HTTPError
from sklearn.model_selection import train_test_split

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def main() -> None:
    """Create the NRK-Quiz-QA-mini dataset and upload them to the HF Hub."""
    # Define the base download URL
    repo_id = "ltg/nrk_quiz_qa"

    # Download the dataset
    nb_dataset = load_dataset(path=repo_id, name="nb", token=True, split="test")
    nn_dataset = load_dataset(path=repo_id, name="nn", token=True, split="test")
    assert isinstance(nb_dataset, Dataset) and isinstance(nn_dataset, Dataset)

    # Convert the dataset to a dataframe
    nb_df = nb_dataset.to_pandas()
    nn_df = nn_dataset.to_pandas()
    assert isinstance(nb_df, pd.DataFrame) and isinstance(nn_df, pd.DataFrame)

    # Merge the datasets
    nb_df["language"] = "nb"
    nn_df["language"] = "nn"
    df = pd.concat([nb_df, nn_df], ignore_index=True)

    # Rename the columns
    df.rename(columns=dict(question="instruction", answer="label"), inplace=True)

    # Remove the samples with overly short or long texts
    df = df[
        (df.instruction.str.len() >= MIN_NUM_CHARS_IN_INSTRUCTION)
        & (df.instruction.str.len() <= MAX_NUM_CHARS_IN_INSTRUCTION)
    ]

    def is_repetitive(text: str) -> bool:
        """Return True if the text is repetitive."""
        max_repetitions = max(Counter(text.split()).values())
        return max_repetitions > MAX_REPETITIONS

    # Remove overly repetitive samples
    df = df[~df.instruction.apply(is_repetitive)]
    assert isinstance(df, pd.DataFrame)

    # We can have at most 256 quizzes, as this is the size of the validation split and
    # we are stratifying by quiz, so we keep only the top-256 quizzes by count
    quiz_counts = df.quiz.value_counts()
    top_quizzes = quiz_counts.head(256).index
    df = df[df.quiz.isin(top_quizzes)]

    # Sanity check that there are never more than 4 options
    assert df.choices.apply(lambda x: len(x["text"])).max() <= 4

    # Make a `text` column with all the options in it
    df["text"] = [
        row.instruction.replace("\n", " ").strip() + "\n"
        "Svaralternativer:\n"
        + "\n".join(
            [
                f"{char}. {clean_text(text=option)}"
                for char, option in zip("abcd", row.choices["text"])
            ]
        )
        for _, row in df.iterrows()
    ]

    # Make the `label` column case-consistent with the `text` column
    df.label = df.label.str.lower()

    # Only keep the `text`, `label` and `quiz` columns
    df = df[["text", "label", "quiz"]]
    assert isinstance(df, pd.DataFrame)

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create validation split
    val_size = 256
    traintest_arr, val_arr = train_test_split(
        df, test_size=val_size, random_state=4242, stratify=df.quiz
    )
    traintest_df = pd.DataFrame(traintest_arr, columns=df.columns)
    val_df = pd.DataFrame(val_arr, columns=df.columns)

    # Create test split
    test_size = 2048
    train_arr, test_arr = train_test_split(
        traintest_df, test_size=test_size, random_state=4242, stratify=traintest_df.quiz
    )
    train_df = pd.DataFrame(train_arr, columns=df.columns)
    test_df = pd.DataFrame(test_arr, columns=df.columns)

    # Create train split
    max_train_size = 1024
    if len(train_df) > max_train_size:
        train_df = train_df.sample(max_train_size, random_state=4242)

    # Reset the index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/nrk-quiz-qa-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def clean_text(text: str) -> str:
    """Clean the text.

    Args:
        text:
            The text to be cleaned.

    Returns:
        The cleaned text.
    """
    return text.replace("\n", " ").strip()


if __name__ == "__main__":
    main()
