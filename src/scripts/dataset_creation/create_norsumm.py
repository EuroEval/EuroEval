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
from constants import MAX_NUM_CHARS_IN_ARTICLE, MIN_NUM_CHARS_IN_ARTICLE
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi

BASE_URL = (
    "https://raw.githubusercontent.com/SamiaTouileb/NorSumm/refs/heads/main/Data/"
)


def main() -> None:
    """Create the NorSumm summarisation datasets and upload to the HF Hub."""
    for lang in ("nb", "nn"):
        dataset = create_dataset(lang=lang)
        dataset_id = f"EuroEval/norsumm-{lang}"
        HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
        dataset.push_to_hub(dataset_id, private=True)


def create_dataset(lang: str) -> DatasetDict:
    """Create the NorSumm dataset for a given language variant.

    Args:
        lang:
            The language variant, either "nb" or "nn".

    Returns:
        A DatasetDict with train and test splits.
    """
    dev_records = process_records(load_norsumm("dev"), lang=lang)
    test_records = process_records(load_norsumm("test"), lang=lang)

    # Use 8 samples from the dev set for training; combine the rest with test
    train_size = 8
    train_records = dev_records[:train_size]
    remaining_dev_records = dev_records[train_size:]

    return DatasetDict(
        {
            "train": Dataset.from_list(train_records, split=Split.TRAIN),
            "test": Dataset.from_list(
                remaining_dev_records + test_records, split=Split.TEST
            ),
        }
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


def process_records(records: list[dict], lang: str) -> list[dict[str, str]]:
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
        summaries = item[key]
        if not summaries:
            continue
        first_summary_key = "summary1"
        if first_summary_key not in summaries[0]:
            continue
        target_text = summaries[0][first_summary_key].strip()
        processed.append({"text": text, "target_text": target_text})
    return processed


if __name__ == "__main__":
    main()
