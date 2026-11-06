# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the sentipolc16-mini sentiment dataset and upload it to the HF Hub."""

if __package__:
    from .tweet_sentiment import create_tweet_sentiment_dataset
else:
    from tweet_sentiment import create_tweet_sentiment_dataset


def main() -> None:
    """Create the sentipolc16 benchmark using the shared tweet workflow."""
    create_tweet_sentiment_dataset(
        language="italian", repository_id="EuroEval/sentipolc16-mini"
    )


if __name__ == "__main__":
    main()
