"""Create a Dutch sentence-level acceptability dataset from BLiMP-NL."""

import hashlib

from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset
from huggingface_hub import HfApi

CAPS = {"train": 1024, "val": 256, "test": 2048}


def main() -> None:
    """Create the BLiMP-NL dataset and upload it to the Hugging Face Hub."""
    rows: list[dict[str, str]] = []
    for config_name in get_dataset_config_names("juletxara/blimp-nl"):
        source = load_dataset("juletxara/blimp-nl", config_name)
        for record in source["train"]:
            rows.append(
                {
                    "sentence_good": record["sentence_good"],
                    "sentence_bad": record["sentence_bad"],
                    "linguistic_phenomenon": record["linguistic_phenomenon"],
                    "paradigm": record["paradigm"],
                    "item_nr": str(record.get("item_nr", record.get("item_id", ""))),
                }
            )

    dataset = process_dataset(rows=rows)
    repo_id = "EuroEval/blimp-nl"
    HfApi().create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    dataset.push_to_hub(repo_id, private=True)


def process_dataset(
    *, rows: list[dict[str, str]], source_splits: dict[str, str] | None = None
) -> DatasetDict:
    """Convert minimal pairs to balanced single-sentence examples.

    Args:
        rows:
            Source minimal-pair records.
        source_splits (optional): Split assignment by pair identifier. When omitted,
            pairs are deterministically assigned to train, validation and test.

    Returns:
        Capped, balanced dataset with ``text`` and ``label`` columns.
    """
    assignment = source_splits or _deterministic_pair_splits(rows=rows)
    pairs_by_split: dict[str, list[dict[str, str]]] = {split: [] for split in CAPS}
    for index, pair in enumerate(rows):
        pair_id = _pair_id(pair=pair, index=index)
        split = assignment[pair_id]
        pairs_by_split[split].append(pair)

    datasets: dict[str, Dataset] = {}
    for split, pairs in pairs_by_split.items():
        pairs = pairs[: CAPS[split]]
        examples = [
            example
            for pair in pairs
            for example in (
                {"text": pair["sentence_good"], "label": "correct"},
                {"text": pair["sentence_bad"], "label": "incorrect"},
            )
        ]
        datasets[split] = Dataset.from_list(examples)
    return DatasetDict(datasets)


def _deterministic_pair_splits(*, rows: list[dict[str, str]]) -> dict[str, str]:
    """Assign complete pairs to deterministic splits when upstream has none.

    Args:
        rows:
            Source minimal-pair records.

    Returns:
        Mapping from pair identifier to split name.
    """
    identifiers = sorted(
        _pair_id(pair=row, index=index) for index, row in enumerate(rows)
    )
    result: dict[str, str] = {}
    for identifier in identifiers:
        bucket = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16) % 100
        result[identifier] = (
            "train" if bucket < 31 else "val" if bucket < 39 else "test"
        )
    return result


def _pair_id(*, pair: dict[str, str], index: int) -> str:
    """Return a stable identifier for a source pair."""
    phenomenon = pair.get("linguistic_phenomenon", "")
    paradigm = pair.get("paradigm", "")
    item_nr = pair.get("item_nr", index)
    return f"{phenomenon}:{paradigm}:{item_nr}"


if __name__ == "__main__":
    main()
