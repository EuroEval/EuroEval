# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Slovak CSFD sentiment dataset and upload to HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the Slovak CSFD sentiment dataset and upload to HF Hub."""
    # Define the dataset repository
    repo_id = "fewshot-goes-multilingual/sk_csfd-movie-reviews"

    # Download the dataset
    dataset = load_dataset(path=repo_id)
    assert isinstance(dataset, DatasetDict)

    # Process each split separately
    processed_splits = {}

    for split_name in ["train", "validation", "test"]:
        # Convert to dataframe
        df = dataset[split_name].to_pandas()
        assert isinstance(df, pd.DataFrame)

        # Keep only columns comment and rating_int for sentiment analysis
        df = df[["comment", "rating_int"]]

        # Map rating_int to sentiment labels
        # ratings: 0-1 = negative, 2-3 = neutral, 4-5 = positive
        df["label"] = df["rating_int"].apply(
            lambda x: "negative" if x <= 1 else ("neutral" if x <= 3 else "positive")
        )

        # Rename comment column to text
        df.rename(columns={"comment": "text"}, inplace=True)

        # Keep only columns text and label
        df = df[["text", "label"]]

        # Remove duplicates within each split
        df = df.drop_duplicates().reset_index(drop=True)

        # Strip trailing whitespace
        df.text = df.text.str.strip()

        # Store processed split
        processed_splits[split_name] = df

    # Make splits
    train_df = processed_splits["train"]
    val_df = processed_splits["validation"]
    test_df = processed_splits["test"]

    # Define split sizes
    test_size = 2048
    val_size = 256
    train_size = 1024

    # Sample final splits
    final_test_df = test_df.sample(n=test_size, random_state=4242).reset_index(
        drop=True
    )
    final_val_df = val_df.sample(n=val_size, random_state=4242).reset_index(drop=True)
    final_train_df = train_df.sample(n=train_size, random_state=4242).reset_index(
        drop=True
    )

    # Create dataset dictionary with custom splits
    dataset = DatasetDict(
        train=Dataset.from_pandas(final_train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(final_val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(final_test_df, split=Split.TEST),
    )

    # Push the dataset to the Hugging Face Hub
    dataset_id = "EuroEval/csfd-sentiment-sk-mini"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
