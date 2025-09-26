# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the CSFD Sentiment dataset and upload to HF Hub."""

import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the Czech CSFD sentiment dataset and upload to HF Hub."""
    # Define the dataset repository
    repo_id = "CZLC/csfd_sentiment_balanced"

    # Download the dataset
    dataset = load_dataset(path=repo_id)
    assert isinstance(dataset, DatasetDict)

    # Process each split separately
    processed_splits = {}

    for split_name in ["train", "validation", "test"]:
        # Convert to dataframe
        df = dataset[split_name].to_pandas()
        assert isinstance(df, pd.DataFrame)

        # Rename query -> text
        df.rename(columns={"query": "text"}, inplace=True)

        # Create the label column based on gold index
        # gold: 0 = negativní, 1 = neutrální, 2 = pozitivní
        df["label"] = df["gold"].map({0: "negative", 1: "neutral", 2: "positive"})

        # Keep only columns text and label
        df = df[["text", "label"]]

        # Remove duplicates within each split
        df = df.drop_duplicates().reset_index(drop=True)

        # Store processed split
        processed_splits[split_name] = df

    # Make splits
    train_df = processed_splits["train"]
    val_df = processed_splits["validation"]
    test_df = processed_splits["test"]

    test_size = 2048
    val_size = 256
    train_size = 1024

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
    final_train_df = remaining_train_df.sample(
        n=train_size, random_state=4242
    ).reset_index(drop=True)

    # Create dataset dictionary with custom splits
    dataset = DatasetDict(
        train=Dataset.from_pandas(final_train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(final_val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(final_test_df, split=Split.TEST),
    )

    # Push the dataset to the Hugging Face Hub
    dataset_id = "EuroEval/csfd-sentiment-mini"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
