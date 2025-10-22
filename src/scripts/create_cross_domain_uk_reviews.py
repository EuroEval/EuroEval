# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Cross-Domain UK Reviews dataset and upload it to the HF Hub."""

from pathlib import Path

import pandas as pd
import requests
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from sklearn.utils import resample


def main() -> None:
    """Create the Cross-Domain UK Reviews dataset and upload it to the HF Hub."""
    url = "https://huggingface.co/datasets/vkovenko/cross_domain_uk_reviews/resolve/main/processed_data.csv"
    file_path = Path("processed_data.csv")
    download_file(url=url, file_path=file_path)

    # Read and process the data
    df = pd.read_csv(file_path)
    df.rename(columns={"review_translate": "text"}, inplace=True)
    df["label"] = df["rating"].map(
        {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
    )
    df = df[["text", "label"]]
    df.drop_duplicates(inplace=True)
    df = create_uniform_label_distribution(df=df, random_state=4242)
    df = df.reset_index(drop=True)

    # Make splits
    train_size = 1024
    val_size = 256
    test_size = 2048

    train_df = df[:train_size]
    val_df = df[train_size : train_size + val_size]
    test_df = df[train_size + val_size : train_size + val_size + test_size]

    # Create dataset
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Push the dataset to the Hugging Face Hub
    dataset_id = "EuroEval/cross-domain-uk-reviews-mini"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)

    file_path.unlink(missing_ok=True)


def download_file(url: str, file_path: Path) -> None:
    """Download a file from a URL and save it to a specified location.

    Args:
        url: The URL of the file to download.
        file_path: The path to the file to save the downloaded file to.
    """
    if file_path.exists():
        return
    response = requests.get(url)
    response.raise_for_status()

    with open(file_path, "wb") as file:
        file.write(response.content)


def create_uniform_label_distribution(
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

    # Combine the resampled dataframes
    balanced_df = pd.concat(resampled_dfs, ignore_index=True)

    # Shuffle the dataframe
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )

    return balanced_df


if __name__ == "__main__":
    main()
