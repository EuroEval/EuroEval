# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Slovenian SentiNews sentiment dataset and upload to HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from sklearn.utils import resample


def main() -> None:
    """Create the Slovenian SentiNews sentiment dataset and upload to HF Hub."""
    # Define the dataset repository
    repo_id = "cjvt/sentinews"

    # Download the dataset
    dataset = load_dataset(repo_id, "sentence_level", split="train")

    # Convert to dataframe
    df = dataset.to_pandas()

    # Fix columns
    df = df.rename(columns={"content": "text", "sentiment": "label"})
    df = df[["text", "label"]]

    # Make splits
    test_size = 2048
    val_size = 256
    train_size = 1024

    # Create uniform label distribution
    df = create_uniform_label_distribution(df=df)

    # Sample final splits
    final_test_df = df.sample(n=test_size, random_state=42).reset_index(drop=True)
    df = df[~df.index.isin(final_test_df.index)]
    final_val_df = df.sample(n=val_size, random_state=42).reset_index(drop=True)
    df = df[~df.index.isin(final_val_df.index)]
    final_train_df = df.sample(n=train_size, random_state=42).reset_index(drop=True)

    # Create a dataset dictionary with custom splits
    dataset_dict = DatasetDict(
        train=Dataset.from_pandas(final_train_df),
        val=Dataset.from_pandas(final_val_df),
        test=Dataset.from_pandas(final_test_df),
    )

    # Push the dataset to the Hugging Face Hub
    dataset_id = "EuroEval/sentinews-mini"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset_dict.push_to_hub(dataset_id, private=True)


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
