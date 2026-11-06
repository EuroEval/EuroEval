# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the SB10k-mini sentiment dataset and upload it to the HF Hub."""

from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT
from datasets import DatasetDict, load_dataset

if __package__:
    from .dataset_utils import preflight_private_upload
else:
    from dataset_utils import preflight_private_upload

if __package__:
    from .dataset_processing import process_multilingual_tweet_sentiment
else:
    from dataset_processing import process_multilingual_tweet_sentiment


def main() -> None:
    """Create the SB10k-mini sentiment dataset and upload it to the HF Hub."""
    # Define the base download URL
    repo_id = "cardiffnlp/tweet_sentiment_multilingual"

    # Download the dataset
    dataset = load_dataset(path=repo_id, name="german", token=True)
    assert isinstance(dataset, DatasetDict)

    dataset = process_multilingual_tweet_sentiment(
        source=dataset,
        source_label_mapping={0: "negative", 1: "neutral", 2: "positive"},
        minimum_text_length=MIN_NUM_CHARS_IN_DOCUMENT,
        maximum_text_length=MAX_NUM_CHARS_IN_DOCUMENT,
    )

    dataset_id = "EuroEval/sb10k-mini"
    # Push the dataset to the Hugging Face Hub
    dataset_id = "EuroEval/sb10k-mini"
    preflight_private_upload(repository_id=dataset_id)
    dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
