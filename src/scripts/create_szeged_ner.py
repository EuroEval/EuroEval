# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "pandas==2.2.0",
#     "requests==2.32.5",
# ]
# ///


"""Create the Hungarian NER dataset SzegedNER and upload it to the HF Hub."""

import ast

import pandas as pd
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from sklearn.utils import resample


def main() -> None:
    """Create the Hungarian NER dataset SzegedNER and upload it."""
    train_df, val_df, test_df = download_dataset()

    column_to_keep = ["tokens", "labels"]
    train_df = train_df[column_to_keep]
    val_df = val_df[column_to_keep]
    test_df = test_df[column_to_keep]

    # Make splits
    train_size = 1024
    val_size = 256
    test_size = 2048

    final_train_df = train_df.sample(n=train_size, random_state=4242)
    remaining_train_df = train_df[~train_df.index.isin(final_train_df.index)]
    assert isinstance(remaining_train_df, pd.DataFrame)

    final_val_df = val_df.sample(n=val_size, random_state=4242)

    n_missing_test_samples = test_size - len(test_df)
    additional_test_samples = remaining_train_df.sample(
        n=n_missing_test_samples, random_state=4242
    )
    final_test_df = pd.concat([test_df, additional_test_samples], ignore_index=True)

    # Reset the index
    final_train_df = final_train_df.reset_index(drop=True)
    final_val_df = final_val_df.reset_index(drop=True)
    final_test_df = final_test_df.reset_index(drop=True)

    assert isinstance(final_train_df, pd.DataFrame)
    assert isinstance(final_val_df, pd.DataFrame)
    assert isinstance(final_test_df, pd.DataFrame)

    # Create dataset dictionary
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(final_train_df, split=Split.TRAIN),
            "val": Dataset.from_pandas(final_val_df, split=Split.VALIDATION),
            "test": Dataset.from_pandas(final_test_df, split=Split.TEST),
        }
    )

    # Create dataset ID
    dataset_id = "EuroEval/szeged-ner-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def download_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download the SzegedNER dataset.

    Download both business and criminal subsets and concatenate them by split.

    Returns:
        A tuple of DataFrames for the training, validation, and test splits.
    """
    base_urls = {
        "business": "https://huggingface.co/datasets/ficsort/SzegedNER/resolve/main/data/business",
        "criminal": "https://huggingface.co/datasets/ficsort/SzegedNER/resolve/main/data/criminal",
    }
    # Types of datasets to download
    splits = ["train", "validation", "test"]

    # Iterate through base URLs and dataset types to download data
    datasets = {}
    for dataset_name, base_url in base_urls.items():
        dataset = {}
        for split in splits:
            df = pd.read_csv(f"{base_url}/{split}.csv")
            dataset[split] = _process_df(df=df, dataset_name=dataset_name)
        datasets[dataset_name] = dataset

    # Merge splits into a single dataset
    train_df = pd.concat([datasets["business"]["train"], datasets["criminal"]["train"]])
    val_df = pd.concat(
        [datasets["business"]["validation"], datasets["criminal"]["validation"]]
    )
    test_df = pd.concat([datasets["business"]["test"], datasets["criminal"]["test"]])

    train_df = _create_uniform_dataset_distribution(df=train_df)
    val_df = _create_uniform_dataset_distribution(df=val_df)
    test_df = _create_uniform_dataset_distribution(df=test_df)

    return train_df, val_df, test_df


def _process_df(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Process the dataframe.

    Args:
        df: The dataframe.
        dataset_name: The name of the dataset.

    Returns:
        The processed dataframe.
    """
    df["original-dataset"] = dataset_name

    df = df.rename(columns={"ner": "labels"})
    df["labels"] = df["labels"].apply(ast.literal_eval)
    df["tokens"] = df["tokens"].apply(ast.literal_eval)
    return df


def _create_uniform_dataset_distribution(
    df: pd.DataFrame, random_state: int = 4242
) -> pd.DataFrame:
    """Create a dataset with a uniform dataset distribution.

    We have merged the business and criminal datasets,
    and want to have equally many samples from each dataset in each split.

    Args:
        df: The dataframe.
        random_state: The random state for reproducibility.

    Returns:
        The dataframe with a uniform dataset distribution.
    """
    # Separate each class
    dataset_names = df["original-dataset"].unique()
    dataset_dfs = [
        df[df["original-dataset"] == dataset_name] for dataset_name in dataset_names
    ]

    # Find the size of the smallest class
    min_size = min(len(dataset_df) for dataset_df in dataset_dfs)

    # Resample each class to the size of the smallest class
    resampled_dfs: list[pd.DataFrame] = [
        resample(
            dataset_df, replace=False, n_samples=min_size, random_state=random_state
        )
        for dataset_df in dataset_dfs
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
