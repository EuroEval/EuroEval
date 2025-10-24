# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Bulgarian Exams knowledge dataset and upload to HF Hub."""

import json
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests
from constants import CHOICES_MAPPING  # noqa
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi


def main() -> None:
    """Create the Bulgarian Exams knowledge dataset and upload to HF Hub."""
    # URL to the tar.gz file
    url = "https://github.com/bgGLUE/bgglue/raw/refs/heads/main/data/exams.tar.gz"

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        data_dir = download_dataset(url=url, temp_path=temp_path)

        train_df = load_split(file_path=data_dir / "train.jsonl")
        val_df = load_split(file_path=data_dir / "dev.jsonl")
        test_df = load_split(file_path=data_dir / "test.jsonl")

        train_df = process_split(df=train_df)
        val_df = process_split(df=val_df)
        test_df = process_split(df=test_df)

        final_train_df, final_val_df, final_test_df = create_splits(
            train_df=train_df, val_df=val_df, test_df=test_df
        )

        dataset = DatasetDict(
            train=Dataset.from_pandas(final_train_df, split=Split.TRAIN),
            val=Dataset.from_pandas(final_val_df, split=Split.VALIDATION),
            test=Dataset.from_pandas(final_test_df, split=Split.TEST),
        )

        dataset_id = "EuroEval/exams-bg-mini"
        HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
        dataset.push_to_hub(dataset_id, private=True)


def download_dataset(url: str, temp_path: Path) -> Path:
    """Download the dataset.

    Args:
        url: URL to the tar.gz file.
        temp_path: Path to the temporary directory.

    Returns:
        Path to the data directory.
    """
    tar_path = temp_path / "exams.tar.gz"
    response = requests.get(url)
    response.raise_for_status()
    with open(tar_path, "wb") as f:
        f.write(response.content)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(temp_path)

    train_file = list(temp_path.glob("**/train.jsonl"))
    if not train_file:
        raise FileNotFoundError("Could not find train.jsonl in extracted files")

    data_dir = train_file[0].parent
    return data_dir


def load_split(file_path: Path) -> pd.DataFrame:
    """Load a JSONL file into a pandas DataFrame.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        A DataFrame with the data.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    return df


def process_split(df: pd.DataFrame) -> pd.DataFrame:
    """Process a split of the dataset.

    Args:
        df: The dataframe to process.

    Returns:
        The processed dataframe.
    """
    texts = []
    labels = []

    for _, row in df.iterrows():
        # Extract question stem
        question_stem = row["question"]["stem"]

        # Extract choices
        choices = row["question"]["choices"]
        if not len(choices) == 4:
            # Keep only the samples with 4 choices
            continue

        # Sort choices by label to ensure consistent order (A, B, C, D)
        sorted_choices = sorted(choices, key=lambda x: x["label"])

        # Build the text with choices
        choice_lines = []
        for choice in sorted_choices:
            label = choice["label"].lower()
            text = choice["text"]
            choice_lines.append(f"{label}. {text}")

        # Get Bulgarian word for "Choices"
        choices_text = CHOICES_MAPPING.get("bg", "Възможности")

        # Construct the full text
        text = f"{question_stem}\n{choices_text}:\n" + "\n".join(choice_lines)

        # Get the correct answer label (lowercase)
        label = row["answerKey"].lower()

        texts.append(text)
        labels.append(label)

    result_df = pd.DataFrame({"text": texts, "label": labels})

    # Drop duplicates based on text
    result_df = result_df.drop_duplicates(subset="text").reset_index(drop=True)

    return result_df


def create_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create splits for the dataset.

    Args:
        train_df: The training dataframe.
        val_df: The validation dataframe.
        test_df: The test dataframe.

    Returns:
        The final training, validation, and test dataframes.
    """
    train_size = 1024
    test_size = 2048

    final_train_df = train_df.sample(n=train_size, random_state=4242)
    remaining_train_df = train_df.drop(final_train_df.index)
    test_df_with_remaining_train_samples = pd.concat(
        [test_df, remaining_train_df], ignore_index=True
    )

    n_missing_samples = test_size - len(test_df_with_remaining_train_samples)
    additional_val_samples = val_df.sample(n=n_missing_samples, random_state=4242)
    final_test_df = pd.concat(
        [test_df_with_remaining_train_samples, additional_val_samples],
        ignore_index=True,
    )
    final_val_df = val_df.drop(additional_val_samples.index)

    final_train_df = final_train_df.reset_index(drop=True)
    final_val_df = final_val_df.reset_index(drop=True)
    final_test_df = final_test_df.reset_index(drop=True)

    return final_train_df, final_val_df, final_test_df


if __name__ == "__main__":
    main()
