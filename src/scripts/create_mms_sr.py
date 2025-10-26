# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Serbian MMS Sentiment dataset and upload to HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from sklearn.utils import resample


def main() -> None:
    """Create the Serbian MMS sentiment dataset and upload to HF Hub."""
    # Define the dataset repository
    repo_id = "Brand24/mms"

    # Download the dataset
    dataset = load_dataset(path=repo_id)
    assert isinstance(dataset, DatasetDict)

    # Get the train split (MMS only has train split)
    df = dataset["train"].to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Filter for Serbian language only
    df = df[df["language"] == "sr"].reset_index(drop=True)

    # Create a uniform distribution based on the original dataset and label columns
    df = create_uniform_distribution(df=df, column="original_dataset")
    df = create_uniform_distribution(df=df, column="label")

    # Map numeric labels to string labels
    # 0: negative, 1: neutral, 2: positive
    df["label"] = df["label"].map({0: "negative", 1: "neutral", 2: "positive"})

    # Keep only text and label columns
    df = df[["text", "label"]]

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Define split sizes
    test_size = 2048
    val_size = 256
    train_size = 1024

    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=4242).reset_index(drop=True)

    # Create splits
    test_df = df.iloc[:test_size].reset_index(drop=True)
    val_df = df.iloc[test_size : test_size + val_size].reset_index(drop=True)
    train_df = df.iloc[
        test_size + val_size : test_size + val_size + train_size
    ].reset_index(drop=True)

    # Create dataset dictionary with custom splits
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Push the dataset to the Hugging Face Hub
    dataset_id = "EuroEval/mms-sr-mini"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


def create_uniform_distribution(
    df: pd.DataFrame, column: str = "label", random_state: int = 4242
) -> pd.DataFrame:
    """Create a sampled dataset with a uniform distribution for the given column.

    Args:
        df: The input dataframe.
        column: The name of the column to create a uniform label distribution for
        random_state: The random state for reproducibility.

    Returns:
        A dataframe with a uniform label distribution.
    """
    # Separate each class
    classes = df[column].unique()
    class_dfs = [df[df[column] == label] for label in classes]

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


if __name__ == "__main__":
    main()
