# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==2.19.1",
#     "nltk==3.8.1",
#     "huggingface-hub==0.24.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the Público-mini summarisation dataset."""

import random

import nltk
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub.hf_api import HfApi
from nltk.tokenize import sent_tokenize  # noqa: E402
from requests.exceptions import HTTPError

nltk.download("punkt")

TOTAL = 1024 + 256 + 2048


def main() -> None:
    """Create the Público-mini dataset."""
    raw = load_dataset("duarteocarmo/cc_news_publico", split="train")
    processed = [_extract_fields(x) for x in raw]
    processed = [x for x in processed if x]

    random.seed(42)
    random.shuffle(processed)
    processed = processed[:TOTAL]

    train = Dataset.from_list(processed[:1024])
    val = Dataset.from_list(processed[1024 : 1024 + 256])
    test = Dataset.from_list(processed[1024 + 256 :])

    dataset = DatasetDict({"train": train, "val": val, "test": test})

    dataset_id = "EuroEval/publico-mini"

    # Remove the dataset from Hugging Face Hub if it already exists
    try:
        api = HfApi()
        api.delete_repo(dataset_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_id, private=True)


def _extract_fields(example: dict) -> dict | None:
    """Extracts fields from a dataset example.

    Args:
        example: Dictionary with keys 'title' and 'plain_text'.

    Returns:
        dict with 'text' and 'target_text', or None if invalid.
    """
    title = example.get("title", "").strip()
    text = example.get("plain_text", "").strip()
    if not title or not text:
        return None
    sentences = sent_tokenize(text, language="portuguese")
    if len(sentences) < 3:
        return None
    return {
        "text": f"{title}\n\n" + " ".join(sentences[2:]),
        "target_text": " ".join(sentences[:2]),
    }


if __name__ == "__main__":
    main()
