"""Create the bfcl datasets and upload them to the HF Hub."""

import json
import os
import random
from pathlib import Path
from urllib.request import urlopen

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv


def split_dataset_to_dict(items: list[dict]) -> DatasetDict:
    """Split dataset into train/validation/test with deterministic random sampling.

    Args:
        items: List of dataset items to split

    Returns:
        DatasetDict with train (10%), validation (10%), and test (80%) splits
    """
    random.seed(42)
    shuffled_items = items.copy()
    random.shuffle(shuffled_items)

    total = len(shuffled_items)
    train_size = int(0.1 * total)
    val_size = int(0.1 * total)

    train_items = shuffled_items[:train_size]
    val_items = shuffled_items[train_size : train_size + val_size]
    test_items = shuffled_items[train_size + val_size :]

    return DatasetDict(
        {
            "train": Dataset.from_list(train_items),
            "validation": Dataset.from_list(val_items),
            "test": Dataset.from_list(test_items),
        }
    )


def _load_jsonl(x: str | Path) -> list:
    if isinstance(x, Path):
        x = x.read_text()
    return [json.loads(line) for line in x.splitlines()]


def _load_jsonl_from_url(url: str) -> list:
    with urlopen(url) as r:
        return _load_jsonl(r.read().decode())


def main() -> None:
    """Create bfcl dataset."""
    items: list = []
    for subset_name in [
        "live_multiple",
        "live_parallel_multiple",
        "live_parallel",
        "live_simple",
        "multiple",
        "parallel_multiple",
        "parallel",
        "simple_java",
        "simple_javascript",
        "simple_python",
    ]:
        url_prefix = (
            "https://raw.githubusercontent.com/ShishirPatil/gorilla"
            "/refs/heads/main/berkeley-function-call-leaderboard/bfcl_eval/data"
        )
        input_url = f"{url_prefix}/BFCL_v4_{subset_name}.json"
        ground_truth_url = f"{url_prefix}/possible_answer/BFCL_v4_{subset_name}.json"
        print(f"Loading dataset '{subset_name}'")
        inputs = _load_jsonl_from_url(input_url)
        ground_truth = _load_jsonl_from_url(ground_truth_url)
        # Join input and ground_truth by 'id' key
        ground_truth_by_id = {item["id"]: item for item in ground_truth}
        for item in inputs:
            item_id = item["id"]
            gt: dict = ground_truth_by_id.get(item_id, {})
            joined: dict = item | gt
            items.append(joined)

    for item in items:
        item["function"] = json.dumps(item["function"])
        item["ground_truth"] = json.dumps(item["ground_truth"])

    load_dotenv()
    dataset_dict = split_dataset_to_dict(items)
    dataset_dict.push_to_hub(
        "EuroEval/bfcl-en", private=True, token=os.getenv("HF_TOKEN")
    )


if __name__ == "__main__":
    main()
