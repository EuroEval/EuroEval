# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
#     "scikit-learn==1.6.1",
# ]
# ///

"""Create the Atsiliepimai sentiment dataset and upload to HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split


def main() -> None:
    """Create the Atsiliepimai sentiment dataset and upload to HF Hub."""
    # Define the dataset repository
    repo_id = "alexandrainst/lithuanian-sentiment-analysis"

    # Download the dataset
    dataset = load_dataset(repo_id, split="train")
    assert isinstance(dataset, Dataset)

    # Convert to dataframe
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Fix columns
    df = df.rename(columns={"rating": "label"})
    df = df.loc[["text", "label"]]

    # Map labels
    label_mapping = {
        5: "positive",
        4: "neutral",
        3: "neutral",
        2: "negative",
        1: "negative",
    }
    df["label"] = df["label"].map(label_mapping)

    # Make splits
    # Dataset has 1796 samples, so we will have 1796 - 512 - 256 = 1028 samples
    # for the test split.
    val_size = 256
    train_size = 512

    # Sample final splits
    df, final_train_df = train_test_split(
        df, test_size=train_size, random_state=42, stratify=df["label"]
    )
    df, final_val_df = train_test_split(
        df, test_size=val_size, random_state=42, stratify=df["label"]
    )
    final_test_df = df

    # Reset indices
    final_train_df = final_train_df.reset_index(drop=True)
    final_val_df = final_val_df.reset_index(drop=True)
    final_test_df = final_test_df.reset_index(drop=True)

    # Create a dataset dictionary with custom splits
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(final_train_df),
            "val": Dataset.from_pandas(final_val_df),
            "test": Dataset.from_pandas(final_test_df),
        }
    )

    # Push the dataset to the Hugging Face Hub
    dataset_id = "EuroEval/atsiliepimai"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset_dict.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
