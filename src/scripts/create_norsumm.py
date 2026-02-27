# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.24.0",
#     "requests==2.32.3",
# ]
# ///

"""Create the NorSumm summarisation dataset."""

import requests
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi

from .constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE

BASE_URL = (
    "https://raw.githubusercontent.com/SamiaTouileb/NorSumm/refs/heads/main/Data/"
)


def load_norsumm(split: str) -> list[dict]:
    """Load the raw NorSumm data for a given split.

    Args:
        split:
            The split to load, either "dev" or "test".

    Returns:
        A list of dictionaries with the raw data.
    """
    url = f"{BASE_URL}NorSumm_{split}.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def process_records(records: list[dict], lang: str) -> list[dict[str, str | list[str]]]:
    """Process raw NorSumm records into EuroEval format.

    Args:
        records:
            A list of raw NorSumm records.
        lang:
            The language variant to use, either "nb" or "nn".

    Returns:
        A list of processed records with "text" and "target_text" columns.
    """
    key = f"summaries_{lang}"
    processed = []
    for item in records:
        text = item["article"].strip()
        text_len = len(text)
        # Only keep articles within reasonable length bounds for summarisation
        if text_len < MIN_NUM_CHARS_IN_ARTICLE or text_len > MAX_NUM_CHARS_IN_ARTICLE:
            continue
        summaries = [
            entry[f"summary{i + 1}"].strip()
            for i, entry in enumerate(item[key])
            if f"summary{i + 1}" in entry
        ]
        processed.append({"text": text, "target_text": summaries})
    return processed


def create_dataset(lang: str) -> DatasetDict:
    """Create the NorSumm dataset for a given language variant.

    Args:
        lang:
            The language variant, either "nb" or "nn".

    Returns:
        A DatasetDict with train, val, and test splits.
    """
    dev_records = process_records(load_norsumm("dev"), lang=lang)
    test_records = process_records(load_norsumm("test"), lang=lang)

    # Split the dev set into train and val
    train_size = 14
    train_records = dev_records[:train_size]
    val_records = dev_records[train_size:]

    return DatasetDict(
        {
            "train": Dataset.from_list(train_records, split=Split.TRAIN),
            "val": Dataset.from_list(val_records, split=Split.VALIDATION),
            "test": Dataset.from_list(test_records, split=Split.TEST),
        }
    )


def main() -> None:
    """Create the NorSumm summarisation datasets and upload to the HF Hub."""
    for lang in ("nb", "nn"):
        dataset = create_dataset(lang=lang)
        dataset_id = f"EuroEval/norsumm-{lang}"
        HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
        dataset.push_to_hub(dataset_id, private=True)


if __name__ == "__main__":
    main()
