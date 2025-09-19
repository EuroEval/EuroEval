# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the PSC (Polish Summaries Corpus) summarization dataset."""

import pandas as pd
from constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the PSC summarization dataset and upload to HF Hub."""
    dataset_id = "community-datasets/psc"

    dataset = load_dataset(dataset_id, token=True)
    assert isinstance(dataset, DatasetDict)

    # Rename columns to match expected format
    dataset = dataset.rename_columns(
        column_mapping={"extract_text": "text", "summary_text": "target_text"}
    )

    # Work primarily with the train split since test split has no proper labels
    train_df = dataset["train"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)

    # Remove the label column since we don't need it for summarization
    if "label" in train_df.columns:
        train_df = train_df.drop(columns=["label"])

    # Only work with samples where the text is not very large or small
    text_lengths = train_df.text.str.len()
    lower_bound = MIN_NUM_CHARS_IN_ARTICLE
    upper_bound = MAX_NUM_CHARS_IN_ARTICLE
    train_df = train_df[text_lengths.between(lower_bound, upper_bound)]

    # Create validation split from train data
    val_size = 256
    val_df = train_df.sample(n=val_size, random_state=4242)
    val_df = val_df.reset_index(drop=True)

    # Remove validation samples from train data and reset index
    remaining_train = train_df.drop(val_df.index).reset_index(drop=True)

    # Create test split from remaining train data
    test_size = 2048
    test_df = remaining_train.sample(
        n=min(test_size, len(remaining_train)), random_state=4242
    )
    test_df = test_df.reset_index(drop=True)

    # Remove test samples from train data
    remaining_train = remaining_train.drop(test_df.index).reset_index(drop=True)

    # Create final train split
    train_size = 1024
    train_df = remaining_train.sample(
        n=min(train_size, len(remaining_train)), random_state=4242
    )
    train_df = train_df.reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    dataset_id = "EuroEval/psc-mini"

    # Push the dataset to the Hugging Face Hub
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
