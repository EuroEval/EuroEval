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

        return df

    # Process each split
    train_df = process_split(df=train_df)
    val_df = process_split(df=val_df)
    test_df = process_split(df=test_df)

    train_size = 1024
    val_size = 256
    test_size = 2048

    train_df = create_balanced_split(df=train_df, target_total_size=train_size)
    val_df = create_balanced_split(df=val_df, target_total_size=val_size)
    test_df = create_balanced_split(df=test_df, target_total_size=test_size)

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


def create_balanced_split(
    df: pd.DataFrame, target_total_size: int, random_state: int = 4242
) -> pd.DataFrame:
    """Create a balanced split with equal samples per class.

    This is used to create a balanced split with equal samples per class.

    Args:
        df: The dataframe to create a balanced split from.
        target_total_size: The total number of samples to create.
        random_state: The random state to use for sampling.

    Returns:
        The balanced split dataframe.
    """
    # Calculate samples per class (divide by 3 for the 3 classes)
    samples_per_class = target_total_size // 3

    # Check if we have enough samples in each class
    label_counts = df["label"].value_counts()
    min_samples_available = min(
        label_counts["negative"], label_counts["positive"], label_counts["neutral"]
    )
    assert min_samples_available > samples_per_class, "Not enough samples in each class"

    # Sample equal amounts from each class
    sampled_class_dfs = []
    remaining_samples = []  # Keep track of unused samples for padding

    for label in ["negative", "positive", "neutral"]:
        class_df = df[df["label"] == label]
        sampled_df = class_df.sample(n=samples_per_class, random_state=random_state)
        sampled_class_dfs.append(sampled_df)

        # Keep remaining samples for padding
        unused_df = class_df.drop(sampled_df.index)
        remaining_samples.append(unused_df)

    # Combine the balanced samples
    balanced_df = pd.concat(sampled_class_dfs, ignore_index=True)

    # Add remaining samples to reach exact target size
    current_size = len(balanced_df)
    needed_samples = target_total_size - current_size
    if needed_samples > 0:
        all_remaining = pd.concat(remaining_samples, ignore_index=True)
        extra_samples = all_remaining.sample(
            n=needed_samples, random_state=random_state
        )
        balanced_df = pd.concat([balanced_df, extra_samples], ignore_index=True)

    # Final shuffle and reset index
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )
    return balanced_df


if __name__ == "__main__":
    main()
