# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
# ]
# ///
"""Create the Estonian IFEval instruction-following dataset and upload to HF Hub."""

import random

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi


def stratified_split(
    dataset: Dataset, label_key: str, test_size: float, seed: int
) -> DatasetDict:
    """Split ensuring all label types appear in both splits."""
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_idx, test_idx = [], []
    labels_in_train = set()
    labels_in_test = set()

    for idx in indices:
        labels = set(dataset[idx][label_key])

        missing_in_test = labels - labels_in_test
        missing_in_train = labels - labels_in_train

        if missing_in_test and len(test_idx) < len(dataset) * test_size:
            test_idx.append(idx)
            labels_in_test.update(labels)
        elif missing_in_train:
            train_idx.append(idx)
            labels_in_train.update(labels)
        elif len(test_idx) < len(dataset) * test_size:
            test_idx.append(idx)
            labels_in_test.update(labels)
        else:
            train_idx.append(idx)
            labels_in_train.update(labels)

    return DatasetDict(
        {"val": dataset.select(train_idx), "test": dataset.select(test_idx)}
    )


def verify_split(split_ds: DatasetDict, label_key: str) -> bool:
    """Verify all label types appear in both splits. Returns True if valid."""
    val_labels: set[str] = set()
    test_labels: set[str] = set()

    for row in split_ds["val"]:
        val_labels.update(row[label_key])
    for row in split_ds["test"]:
        test_labels.update(row[label_key])

    missing_in_val = test_labels - val_labels
    missing_in_test = val_labels - test_labels

    if missing_in_val or missing_in_test:
        print(f"Missing in val: {missing_in_val}, missing in test: {missing_in_test}")
        return False

    print(f"Val: {len(split_ds['val'])} samples, {len(val_labels)} label types")
    print(f"Test: {len(split_ds['test'])} samples, {len(test_labels)} label types")
    return True


def stratified_split_with_retry(
    dataset: Dataset,
    label_key: str,
    test_size: float,
    seed: int,
    max_retries: int = 100,
) -> DatasetDict:
    """Split with retry on different seeds until verification passes."""
    for attempt in range(max_retries):
        current_seed = seed + attempt
        split_ds = stratified_split(dataset, label_key, test_size, current_seed)
        if verify_split(split_ds, label_key):
            if attempt > 0:
                print(f"Succeeded on attempt {attempt + 1} with seed {current_seed}")
            return split_ds
        print(f"Attempt {attempt + 1} failed, retrying...")

    raise ValueError(f"Failed to create valid split after {max_retries} attempts")


def main() -> None:
    """Create the Estonian IFEval dataset and upload to HF Hub."""
    source_repo_id = "tartuNLP/ifeval_et"
    target_repo_id = "EuroEval/ifeval-et"

    ds = load_dataset(source_repo_id, split="test")

    split_ds = stratified_split_with_retry(
        ds, label_key="instruction_id_list", test_size=0.75, seed=42
    )

    def transform(row: dict) -> dict:
        return {
            "text": row["prompt"],
            "target_text": {
                "instruction_id_list": row["instruction_id_list"],
                "kwargs": row["kwargs"],
            },
        }

    split_ds = split_ds.map(transform).select_columns(["text", "target_text"])

    HfApi().delete_repo(target_repo_id, repo_type="dataset", missing_ok=True)
    split_ds.push_to_hub(target_repo_id, private=True)


if __name__ == "__main__":
    main()
