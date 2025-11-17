# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Hungarian summarisation dataset based on hun-sum-chatml-5k."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the Hungarian summarisation mini dataset and upload to HF Hub."""
    dataset_id = "ariel-ml/hun-sum-chatml-5k"

    dataset = load_dataset(dataset_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # The dataset has splits: train, val, test
    # Each sample has: title, lead, article
    # We want: text = "{title}\n\n{article}", target_text = lead

    def make_columns(sample: dict) -> dict:
        """Map the dataset to have the text and target_text columns.

        Args:
            sample: A sample from the dataset.

        Returns:
            A sample with the text and target_text columns.
        """
        sample["text"] = f"{sample['title']}\n\n{sample['article']}"
        sample["target_text"] = sample["lead"]
        return sample

    dataset = dataset.map(make_columns)

    # The val and test splits only have 200 samples each.
    df_train = dataset["train"].to_pandas()
    df_val = dataset["validation"].to_pandas()
    df_test = dataset["test"].to_pandas()
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    keep_columns = ["text", "target_text"]
    df = df[keep_columns]

    # Only work with samples where the text is not very large or small
    lengths = df.text.str.len()
    lower_bound = MIN_NUM_CHARS_IN_ARTICLE
    upper_bound = MAX_NUM_CHARS_IN_ARTICLE
    df = df[lengths.between(lower_bound, upper_bound)]
    df = df.reset_index(drop=True)

    # Create validation split
    val_size = 256
    val_df = df.sample(n=val_size, random_state=4242)
    remaining = df.drop(index=val_df.index)
    val_df = val_df.reset_index(drop=True)

    # Create test split
    test_size = 2048
    test_df = remaining.sample(n=test_size, random_state=4242)
    remaining2 = remaining.drop(index=test_df.index)
    test_df = test_df.reset_index(drop=True)

    # Create train split
    train_size = 1024
    train_df = remaining2.sample(n=train_size, random_state=4242)
    train_df = train_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    mini_dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    # Create dataset ID
    mini_dataset_id = "EuroEval/hun-sum-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    HfApi().delete_repo(mini_dataset_id, repo_type="dataset", missing_ok=True)

    # Push the dataset to the Hugging Face Hub
    mini_dataset.push_to_hub(mini_dataset_id, private=True)


if __name__ == "__main__":
    main()
