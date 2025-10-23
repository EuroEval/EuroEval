# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Greek sentiment analysis dataset and upload it to the HF Hub."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from sklearn.utils import resample


def main() -> None:
    """Create the Greek sentiment analysis dataset and upload it to the HF Hub."""
    # Define the source dataset
    repo_id = "DGurgurov/greek_sa"

    dataset_dict = load_dataset(path=repo_id, token=True)
    assert isinstance(dataset_dict, DatasetDict)

    # Process splits
    train_df = process_split(dataset_dict["train"])
    val_df = process_split(dataset_dict["validation"])
    test_df = process_split(dataset_dict["test"])

    # Make splits
    train_size = 1024
    val_size = 256
    test_size = 2048

    # Calculate how many additional samples needed for test split
    additional_test_samples_needed = test_size - len(test_df)

    # Take additional samples from training set for test split
    additional_test_samples = train_df.sample(
        n=additional_test_samples_needed, random_state=4242
    )

    # Combine original test with additional samples from train
    final_test_df = pd.concat([test_df, additional_test_samples], ignore_index=True)

    # Remove the additional samples from train set
    remaining_train_df = train_df[~train_df.index.isin(additional_test_samples.index)]

    # Sample final splits
    final_test_df = final_test_df.sample(n=test_size, random_state=4242).reset_index(
        drop=True
    )
    final_val_df = val_df.sample(n=val_size, random_state=4242).reset_index(drop=True)

    # The distributions are already fairly uniform in the val and test splits
    final_train_df = create_uniform_label_distribution(df=remaining_train_df)
    final_train_df = final_train_df.sample(n=train_size, random_state=4242).reset_index(
        drop=True
    )

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(final_train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(final_val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(final_test_df, split=Split.TEST),
    )

    # Push the dataset to the Hugging Face Hub
    dataset_id = "EuroEval/greek-sa-mini"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


def process_split(dataset: Dataset) -> pd.DataFrame:
    """Process a split of the dataset.

    Args:
        dataset: The dataset to process.

    Returns:
        A dataframe with the processed split.
    """
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)
    df["label"] = df["label"].map({0: "negative", 1: "positive"})
    df["text_len"] = df["text"].str.len()
    df = df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
        "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
    )
    df = df.drop(columns=["text_len"]).reset_index(drop=True)
    keep_columns = ["text", "label"]
    return df[keep_columns]


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

    return balanced_df


if __name__ == "__main__":
    main()
