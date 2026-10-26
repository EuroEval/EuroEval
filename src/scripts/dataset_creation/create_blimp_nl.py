"""Create a Dutch sentence-level acceptability dataset from BLiMP-NL."""

import hashlib
import tempfile
from pathlib import Path

from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset
from huggingface_hub import HfApi

CAPS = {"train": 1024, "val": 256, "test": 2048}
PAIR_CAPS = {split: size // 2 for split, size in CAPS.items()}
SOURCE_REPO = "juletxara/blimp-nl"
TARGET_REPO = "EuroEval/blimp-nl-mini"


def main() -> None:
    """Create and publish BLiMP-NL-mini.

    Raises:
        ValueError:
            If upstream contains unexpected splits or invalid records.
    """
    rows: list[dict[str, str]] = []
    for config_name in sorted(get_dataset_config_names(SOURCE_REPO)):
        source = load_dataset(SOURCE_REPO, config_name)
        if set(source) != {"train"}:
            raise ValueError(f"Unexpected split shape in {config_name}: {set(source)}")
        for record in source["train"]:
            rows.append(
                {
                    "sentence_good": record["sentence_good"],
                    "sentence_bad": record["sentence_bad"],
                    "linguistic_phenomenon": record["linguistic_phenomenon"],
                    "paradigm": record["paradigm"],
                    "item_nr": str(record.get("item_nr", record.get("item_id", ""))),
                    "config_name": config_name,
                }
            )

    dataset = process_dataset(rows=rows)
    api = HfApi()
    api.create_repo(
        repo_id=TARGET_REPO, repo_type="dataset", exist_ok=True, private=False
    )
    api.update_repo_settings(repo_id=TARGET_REPO, repo_type="dataset", private=False)
    dataset.push_to_hub(TARGET_REPO, private=False)
    card = """---
language:
- nl
license: cc-by-4.0
task_categories:
- text-classification
pretty_name: BLiMP-NL-mini
---

# BLiMP-NL-mini

A capped Dutch grammatical acceptability dataset derived from the BLiMP-NL
minimal-pair benchmark. Each source pair becomes two examples: the grammatical
sentence is labelled `correct` and the ungrammatical sentence `incorrect`.
Both members of a pair are kept in the same split.

Source: [BLiMP-NL](https://huggingface.co/datasets/juletxara/blimp-nl),
licensed under CC BY 4.0.

Please cite the original BLiMP-NL paper when using this dataset:

```bibtex
@article{suijkerbuijk2023blimp,
  title={BLiMP-NL: The Benchmark of Linguistic Minimal Pairs for Dutch},
  author={Suijkerbuijk, M.J.P.F. and Prins, Zoë and de Heer Kloots, Marianne
    and Zuidema, Willem and Frank, Stefan},
  year={2023}
}
```

This derived dataset is distributed under the upstream CC BY 4.0 licence.
"""
    with tempfile.TemporaryDirectory(prefix="blimp-nl-card-") as temporary_directory:
        card_path = Path(temporary_directory) / "README.md"
        card_path.write_text(card, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=TARGET_REPO,
            repo_type="dataset",
        )


def process_dataset(
    *, rows: list[dict[str, str]], source_splits: dict[str, str] | None = None
) -> DatasetDict:
    """Convert minimal pairs to balanced single-sentence examples.

    Args:
        rows:
            Source minimal-pair records.
        source_splits (optional): Split assignment by pair identifier.

    Returns:
        Capped, balanced dataset with ``text`` and ``label`` columns.

    Raises:
        ValueError:
            If records are incomplete, identifiers collide, or splits are invalid.
    """
    identifiers = [_pair_id(pair=pair, index=index) for index, pair in enumerate(rows)]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("BLiMP-NL pair identifiers are not unique")
    for pair in rows:
        required = (
            "sentence_good",
            "sentence_bad",
            "linguistic_phenomenon",
            "paradigm",
        )
        if any(not pair.get(field, "").strip() for field in required):
            raise ValueError("BLiMP-NL records must contain all pair fields")

    assignment = source_splits or _deterministic_pair_splits(
        rows=rows, identifiers=identifiers
    )
    if set(assignment) != set(identifiers) or any(
        split not in CAPS for split in assignment.values()
    ):
        raise ValueError("Split assignment must cover every pair with a valid split")

    pairs_by_split: dict[str, list[tuple[str, dict[str, str]]]] = {
        split: [] for split in CAPS
    }
    for identifier, pair in zip(identifiers, rows, strict=True):
        pairs_by_split[assignment[identifier]].append((identifier, pair))

    datasets: dict[str, Dataset] = {}
    for split, pairs in pairs_by_split.items():
        selected = _stratified_sample(pairs=pairs, limit=PAIR_CAPS[split])
        examples = [
            example
            for _, pair in selected
            for example in (
                {"text": pair["sentence_good"], "label": "correct"},
                {"text": pair["sentence_bad"], "label": "incorrect"},
            )
        ]
        datasets[split] = Dataset.from_list(examples)
    return DatasetDict(datasets)


def _deterministic_pair_splits(
    *, rows: list[dict[str, str]], identifiers: list[str]
) -> dict[str, str]:
    """Assign pairs to reproducible splits within each phenomenon/paradigm.

    Returns:
        Mapping from pair identifiers to split names.
    """
    strata: dict[tuple[str, str], list[str]] = {}
    for identifier, row in zip(identifiers, rows, strict=True):
        key = (row["linguistic_phenomenon"], row["paradigm"])
        strata.setdefault(key, []).append(identifier)

    result: dict[str, str] = {}
    for members in strata.values():
        ordered = sorted(members, key=_stable_hash)
        for index, identifier in enumerate(ordered):
            result[identifier] = (
                "train" if index % 10 < 8 else "val" if index % 10 == 8 else "test"
            )
    return result


def _pair_id(*, pair: dict[str, str], index: int) -> str:
    """Return a stable identifier for a source pair.

    Raises:
        ValueError:
            If the record lacks an item identifier.
    """
    phenomenon = pair.get("linguistic_phenomenon", "")
    paradigm = pair.get("paradigm", "")
    item_nr = pair.get("item_nr", pair.get("item_id", ""))
    config_name = pair.get("config_name", "")
    if not item_nr:
        raise ValueError(f"Missing item identifier in source row {index}")
    return f"{config_name}:{phenomenon}:{paradigm}:{item_nr}"


def _stratified_sample(
    *, pairs: list[tuple[str, dict[str, str]]], limit: int
) -> list[tuple[str, dict[str, str]]]:
    """Select a deterministic, round-robin sample across linguistic strata.

    Returns:
        Selected pair identifiers and records.
    """
    strata: dict[tuple[str, str], list[tuple[str, dict[str, str]]]] = {}
    for identifier, pair in pairs:
        key = (pair["linguistic_phenomenon"], pair["paradigm"])
        strata.setdefault(key, []).append((identifier, pair))
    ordered_groups = [
        sorted(group, key=lambda item: _stable_hash(item[0]))
        for _, group in sorted(strata.items())
    ]
    selected: list[tuple[str, dict[str, str]]] = []
    for round_index in range(max((len(group) for group in ordered_groups), default=0)):
        for group in ordered_groups:
            if round_index < len(group) and len(selected) < limit:
                selected.append(group[round_index])
    return selected


def _stable_hash(identifier: str) -> str:
    """Return a stable ordering key for a pair identifier."""
    return hashlib.sha256(identifier.encode()).hexdigest()


if __name__ == "__main__":
    main()
