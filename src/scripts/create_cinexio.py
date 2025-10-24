# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Cinexio sentiment dataset and upload to HF Hub."""

import json
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests
from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from sklearn.utils import resample


def main() -> None:
    """Create the Cinexio sentiment dataset and upload to HF Hub."""
    # URL to the tar.gz file
    url = "https://raw.githubusercontent.com/bgGLUE/bgglue/main/data/cinexio.tar.gz"

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

        dataset_id = "EuroEval/cinexio-mini"
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
    tar_path = temp_path / "cinexio.tar.gz"
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
    df = _map_ratings_to_labels(df=df)
    df = df.rename(columns={"Comment": "text"})
    df = df[["text", "label"]]
    df = _filter_by_length(df=df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


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
    test_size = 2048
    val_size = 256
    train_size = 1024

    train_df_uniform = _create_uniform_label_distribution(
        df=train_df, random_state=4242
    )
    train_df_remaining = train_df[
        ~train_df.index.isin(train_df_uniform.index)
    ].reset_index(drop=True)

    # The original test split only contains negative samples
    test_df_with_additional_samples = pd.concat(
        [test_df, train_df_remaining], ignore_index=True
    )
    test_df_uniform = _create_uniform_label_distribution(
        df=test_df_with_additional_samples, random_state=4242
    )

    val_df_uniform = _create_uniform_label_distribution(df=val_df, random_state=4242)

    final_train_df = train_df_uniform.sample(
        n=train_size, random_state=4242
    ).reset_index(drop=True)
    final_test_df = test_df_uniform.sample(n=test_size, random_state=4242).reset_index(
        drop=True
    )
    final_val_df = val_df_uniform.sample(n=val_size, random_state=4242).reset_index(
        drop=True
    )

    return final_train_df, final_val_df, final_test_df


def _create_uniform_label_distribution(
    df: pd.DataFrame, random_state: int = 4242
) -> pd.DataFrame:
    """Create a sampled dataset with a uniform label distribution.

    Args:
        df: The input dataframe with a 'label' column.
        random_state: The random state for reproducibility.

    Returns:
        A dataframe with a uniform label distribution.
    """
    # Separate each class
    classes = df["label"].unique()
    class_dfs = [df[df["label"] == label] for label in classes]

    # Find the size of the smallest class
    min_size = min(len(class_df) for class_df in class_dfs)

    # Resample each class to the size of the smallest class
    resampled_dfs = [
        resample(class_df, replace=False, n_samples=min_size, random_state=random_state)
        for class_df in class_dfs
    ]

    # Combine the resampled dataframes (keep original indices!)
    balanced_df = pd.concat(resampled_dfs, ignore_index=False)

    return balanced_df


def _map_ratings_to_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map User_Rating to sentiment labels.

    Rating scale: 0-5 (with 0.5 increments)
    - Negative: 0, 0.5, 1.0, 1.5
    - Neutral: 2.0, 2.5, 3.0, 3.5
    - Positive: 4.0, 4.5, 5.0

    Args:
        df: DataFrame with User_Rating column.

    Returns:
        DataFrame with label column.
    """
    df = df.copy()

    # Convert User_Rating to numeric if it's a string
    df["User_Rating"] = pd.to_numeric(df["User_Rating"])

    # Map ratings to labels
    def rating_to_label(rating: float) -> str:
        if rating <= 1.5:
            return "negative"
        elif rating <= 3.5:
            return "neutral"
        else:  # rating >= 4.0
            return "positive"

    df["label"] = df["User_Rating"].apply(rating_to_label)
    return df


def _filter_by_length(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe by text length.

    Args:
        df: The dataframe to filter.

    Returns:
        The filtered dataframe.
    """
    new_df = df.copy()
    new_df["text_len"] = new_df["text"].str.len()
    filtered_df = new_df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )
    return filtered_df.drop(columns=["text_len"]).reset_index(drop=True)


if __name__ == "__main__":
    main()
