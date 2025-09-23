# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Lithuanian Emotions dataset and upload it to the HF Hub."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT  # noqa
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def main() -> None:
    """Create the Lithuanian Emotions dataset and upload it to the HF Hub."""
    # Define the repository ID
    repo_id = "SkyWater21/lt_emotions"

    # Download the dataset
    dataset = load_dataset(path=repo_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Convert the dataset splits to dataframes
    train_df = dataset["comb_train"].to_pandas()
    val_df = dataset["comb_validation"].to_pandas()
    test_df1 = dataset["lt_go_emotions_test"].to_pandas()
    test_df2 = dataset["lt_twitter_emotions_test"].to_pandas()

    # Combine the two test splits
    test_df = pd.concat([test_df1, test_df2], ignore_index=True)

    def process_split(df: pd.DataFrame) -> pd.DataFrame:
        """Process a split by filtering single labels and mapping them.

        Args:
            df: The dataframe to process.

        Returns:
            The processed dataframe.
        """
        # Filter to only keep samples with single labels (ignore multi-label samples)
        df = df[df["labels"].apply(lambda x: len(x) == 1)].copy()

        # Extract the single integer from the labels list
        df["label_int"] = df["labels"].apply(lambda x: x[0])

        # Map the labels to the EuroEval sentiment labels
        label_mapping = {
            0: "negative",  # anger
            1: "negative",  # disgust
            2: "negative",  # fear
            3: "positive",  # joy
            4: "negative",  # sadness
            5: "positive",  # surprise
            6: "neutral",  # neutral
        }
        df["label"] = df["label_int"].map(label_mapping)

        # Remove any unmapped labels
        df = df.dropna(subset=["label"])

        # Keep only the required columns: lt_text -> text, label
        df = df[["lt_text", "label"]].rename(columns={"lt_text": "text"})

        # Filter by text length
        df["text_len"] = df.text.str.len()
        df = df.query("text_len >= @MIN_NUM_CHARS_IN_DOCUMENT").query(
            "text_len <= @MAX_NUM_CHARS_IN_DOCUMENT"
        )

        # Drop the temporary text_len column
        df = df.drop(columns=["text_len"])

        # Create a uniform label distribution
        df = _create_uniform_label_distribution(df=df)
        return df

    # Process each split
    train_df = process_split(df=train_df)
    val_df = process_split(df=val_df)
    test_df = process_split(df=test_df)

    train_size = 1024
    val_size = 256
    test_size = 2048

    # Assuming train_df, val_df, and test_df are your original DataFrames
    train_df, _ = train_test_split(
        train_df, train_size=train_size, stratify=train_df["label"], random_state=4242
    )
    val_df, _ = train_test_split(
        val_df, train_size=val_size, stratify=val_df["label"], random_state=4242
    )
    test_df, _ = train_test_split(
        test_df, train_size=test_size, stratify=test_df["label"], random_state=4242
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Create DatasetDict
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    dataset_id = "EuroEval/lithuanian-emotions-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


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

    # Combine the resampled dataframes
    balanced_df = pd.concat(resampled_dfs, ignore_index=True)

    return balanced_df


if __name__ == "__main__":
    main()
