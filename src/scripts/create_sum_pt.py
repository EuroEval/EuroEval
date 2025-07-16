# ///
# requires-python = ">=3.10"
# dependencies = ["datasets==2.19.1", "nltk==3.8.1"]
# ///


"""Create the Publico-pt summarisation dataset."""

import random

import nltk
from datasets import Dataset, DatasetDict, load_dataset

nltk.download("punkt")
from nltk.tokenize import sent_tokenize  # noqa: E402

TOTAL = 1024 + 256 + 2048


def _extract_fields(example: dict) -> dict | None:
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


def main() -> None:
    """This script creates the Publico-pt dataset.

    The dataset is created from the cc_news_publico dataset, which is a dataset of
    news articles from the Publico newspaper.
    """
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

    dataset.push_to_hub("EuroEval/sum-pt-publico")


if __name__ == "__main__":
    main()
