"""Shared multilingual tweet sentiment creation workflow."""

from datasets import load_dataset

if __package__:
    from .constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT
    from .dataset_processing import process_multilingual_tweet_sentiment
    from .dataset_utils import create_and_upload_dataset
else:
    from constants import MAX_NUM_CHARS_IN_DOCUMENT, MIN_NUM_CHARS_IN_DOCUMENT
    from dataset_processing import process_multilingual_tweet_sentiment
    from dataset_utils import create_and_upload_dataset


def create_tweet_sentiment_dataset(language: str, repository_id: str) -> None:
    """Load, normalise, filter, cap and privately upload multilingual tweets.

    Args:
        language:
            Source configuration in the multilingual tweet sentiment dataset.
        repository_id:
            Target EuroEval repository for that language's benchmark.
    """
    create_and_upload_dataset(
        load_source=lambda: load_dataset(
            path="cardiffnlp/tweet_sentiment_multilingual", name=language, token=True
        ),
        process_source=lambda source: process_multilingual_tweet_sentiment(
            source=source,
            source_label_mapping={0: "negative", 1: "neutral", 2: "positive"},
            minimum_text_length=MIN_NUM_CHARS_IN_DOCUMENT,
            maximum_text_length=MAX_NUM_CHARS_IN_DOCUMENT,
        ),
        repository_id=repository_id,
    )
