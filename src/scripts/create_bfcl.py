"""Create the bfcl datasets and upload them to the HF Hub."""

import json
import os
import random
from pathlib import Path
from urllib.request import urlopen

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv


def main() -> None:
    """Create BFCL dataset (the part used by EuroEval) and push to hugging face."""
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
        function_str = json.dumps(item["function"])
        question_str = item["question"][0][0]["content"]
        item: dict
        item["text"] = f"Functions:\n{function_str}\nQuestion: {question_str}"
        item["function"] = function_str
        item["question"] = question_str
        item["target_text"] = json.dumps(item.pop("ground_truth"))

    load_dotenv()
    dataset_dict = _split_dataset_to_dict(items)
    dataset_dict.push_to_hub(
        "EuroEval/bfcl-en", private=True, token=os.getenv("HF_TOKEN")
    )


def _split_dataset_to_dict(items: list[dict]) -> DatasetDict:
    """Split dataset into train/validation/test with deterministic random sampling.

    Args:
        items:
            List of dataset items to split

    Returns:
        DatasetDict with train, validation, and test splits
    """
    random.seed(42)
    shuffled_items = items.copy()
    random.shuffle(shuffled_items)

    train_size = 250
    val_size = 250

    train_items = shuffled_items[:train_size]
    val_items = shuffled_items[train_size : train_size + val_size]
    test_items = shuffled_items[train_size + val_size :]

    return DatasetDict(
        {
            "train": Dataset.from_list(train_items),
            "val": Dataset.from_list(val_items),
            "test": Dataset.from_list(test_items),
        }
    )


def _load_jsonl(path: str | Path) -> list:
    """Load jsonl.

    Args:
        path: string with serialized JSONL or Path to JSONL file

    Returns:
        List of deserialized objects
    """
    if isinstance(path, Path):
        path = path.read_text()
    return [json.loads(line) for line in path.splitlines()]


def _load_jsonl_from_url(url: str) -> list:
    """Load jsonl from url.

    Args:
        url: url to JSONL

    Returns:
        List of deserialized objects
    """
    with urlopen(url) as r:
        return _load_jsonl(r.read().decode())


if __name__ == "__main__":
    main()
