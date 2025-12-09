# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "pandas==2.2.0",
#     "requests==2.32.5",
#     "scikit-learn==1.6.1",
# ]
# ///

"""Create the HuSST Hungarian sentiment dataset and upload to HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from sklearn.utils import resample


def main() -> None:
    """Create the HuSST Hungarian sentiment dataset and upload to HF Hub."""
    # Load the dataset
    dataset_id = "NYTK/HuSST"
    dataset = load_dataset(dataset_id)
    assert isinstance(dataset, DatasetDict)
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)

    train_df = process(df=train_df)
    val_df = process(df=val_df)

    # Create uniform label distribution
    train_df_uniform = create_uniform_label_distribution(df=train_df)
    val_df_uniform = create_uniform_label_distribution(df=val_df)
    val_df_remaining = val_df[~val_df.index.isin(val_df_uniform.index)]

    # Make splits
    train_size = 1024
    val_size = 256
    test_size = 2048

    train_df_final = train_df_uniform.sample(
        n=train_size, random_state=4242
    ).reset_index(drop=True)
    train_df_remaining = train_df_uniform[
        ~train_df_uniform.index.isin(train_df_final.index)
    ]

    test_df_final = train_df_remaining.sample(
        n=test_size, random_state=4242
    ).reset_index(drop=True)

    n_missing_val_samples = val_size - len(val_df_uniform)
    val_df_additional = val_df_remaining.sample(
        n=n_missing_val_samples, random_state=4242
    )
    val_df_final = pd.concat([val_df_uniform, val_df_additional], ignore_index=True)

    assert isinstance(train_df_final, pd.DataFrame)
    assert isinstance(val_df_final, pd.DataFrame)
    assert isinstance(test_df_final, pd.DataFrame)

    # Create DatasetDict
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df_final),
            "val": Dataset.from_pandas(val_df_final),
            "test": Dataset.from_pandas(test_df_final),
        }
    )

    # Push to HF Hub
    dataset_id = "EuroEval/husst-mini"
    HfApi().delete_repo(repo_id=dataset_id, repo_type="dataset", missing_ok=True)
    dataset_dict.push_to_hub(dataset_id, private=True)


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe to ensure required columns and rename.

    Args:
        df: The input dataframe.

    Returns:
        A dataframe with the required columns and renamed.
    """
    renames = {"sentence": "text"}
    df = df.rename(columns=renames)
    keep_columns = ["text", "label"]
    df = df.loc[keep_columns]
    return df


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

    # Combine the resampled dataframes (keep original indices!)
    balanced_df = pd.concat(resampled_dfs, ignore_index=False)

    return balanced_df


if __name__ == "__main__":
    main()
