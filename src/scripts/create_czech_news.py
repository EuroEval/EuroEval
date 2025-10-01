# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Czech News Summarization dataset."""

from constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE
from datasets import Dataset, DatasetDict, Split, load_dataset
from huggingface_hub import HfApi


def main() -> None:
    """Create the Czech News Summarization dataset and upload to HF Hub."""
    dataset_id = "hynky/czech_news_dataset_v2"

    # Load the dataset with all splits
    dataset = load_dataset(dataset_id, split=None, token=True)
    assert isinstance(dataset, DatasetDict)

    # Rename columns
    dataset = dataset.rename_columns(
        column_mapping=dict(content="text", brief="target_text")
    )
    dataset = dataset.select_columns(["text", "target_text"])

    # Convert each split to pandas DataFrame
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # Sample from each split, to avoid processing the full dataset
    train_df = train_df.sample(n=2000, random_state=4242).reset_index(drop=True)
    val_df = val_df.sample(n=500, random_state=4242).reset_index(drop=True)
    test_df = test_df.sample(n=3000, random_state=4242).reset_index(drop=True)

    # Only work with samples where the text is not very large or small
    train_lengths = train_df.text.str.len()
    val_lengths = val_df.text.str.len()
    test_lengths = test_df.text.str.len()
    lower_bound = MIN_NUM_CHARS_IN_ARTICLE
    upper_bound = MAX_NUM_CHARS_IN_ARTICLE
    train_df = train_df[train_lengths.between(lower_bound, upper_bound)]
    val_df = val_df[val_lengths.between(lower_bound, upper_bound)]
    test_df = test_df[test_lengths.between(lower_bound, upper_bound)]

    # Make final splits
    train_size = 1024
    val_size = 256
    test_size = 2048

    train_df = train_df.sample(n=train_size, random_state=4242).reset_index(drop=True)
    val_df = val_df.sample(n=val_size, random_state=4242).reset_index(drop=True)
    test_df = test_df.sample(n=test_size, random_state=4242).reset_index(drop=True)

    # Collect datasets in a dataset dictionary
    dataset = DatasetDict(
        train=Dataset.from_pandas(train_df, split=Split.TRAIN),
        val=Dataset.from_pandas(val_df, split=Split.VALIDATION),
        test=Dataset.from_pandas(test_df, split=Split.TEST),
    )

    dataset_id = "EuroEval/czech-news-mini"
    # Push the dataset to the Hugging Face Hub
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
