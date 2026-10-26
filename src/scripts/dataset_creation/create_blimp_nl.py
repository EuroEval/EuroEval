"""Create a Dutch sentence-level acceptability dataset from BLiMP-NL."""

import hashlib

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
            if record["sentence_good"] == record["sentence_bad"]:
                continue
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


def process_dataset(
    *, rows: list[dict[str, str]], source_splits: dict[str, str] | None = None
) -> DatasetDict:
    """Convert minimal pairs to balanced single-sentence examples.

    Args:
        rows:
            Source minimal-pair records.
        source_splits (optional):
            Split assignment by pair identifier. Defaults to None.

    Returns:
        Capped, balanced dataset with ``text`` and ``label`` columns.

    Raises:
        ValueError:
            If records are incomplete, identifiers collide, or splits are invalid.
    """
    unique_rows: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for pair in rows:
        required = (
            "sentence_good",
            "sentence_bad",
            "linguistic_phenomenon",
            "paradigm",
        )
        if any(not pair.get(field, "").strip() for field in required):
            raise ValueError("BLiMP-NL records must contain all pair fields")
        good = pair["sentence_good"]
        bad = pair["sentence_bad"]
        if good == bad:
            raise ValueError("BLiMP-NL pairs cannot contain identical sentences")
        pair_content = (good, bad)
        if pair_content not in seen_pairs:
            seen_pairs.add(pair_content)
            unique_rows.append(pair)

    unique_rows.sort(
        key=lambda pair: (
            pair.get("config_name", ""),
            pair["linguistic_phenomenon"],
            pair["paradigm"],
            pair.get("item_nr", pair.get("item_id", "")),
            pair["sentence_good"],
            pair["sentence_bad"],
        )
    )
    _validate_sentence_labels(rows=unique_rows)
    identifiers = [
        _pair_id(pair=pair, index=index) for index, pair in enumerate(unique_rows)
    ]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("BLiMP-NL pair identifiers are not unique")

    assignment = source_splits or _deterministic_pair_splits(
        rows=unique_rows, identifiers=identifiers
    )
    if set(assignment) != set(identifiers) or any(
        split not in CAPS for split in assignment.values()
    ):
        raise ValueError("Split assignment must cover every pair with a valid split")

    pairs_by_split: dict[str, list[tuple[str, dict[str, str]]]] = {
        split: [] for split in CAPS
    }
    for identifier, pair in zip(identifiers, unique_rows, strict=True):
        pairs_by_split[assignment[identifier]].append((identifier, pair))

    _validate_split_texts(pairs_by_split=pairs_by_split)
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
    """Assign pairs to reproducible splits without separating shared sentences.

    Args:
        rows:
            Source pair records.
        identifiers:
            Stable identifiers corresponding to the source pairs.

    Returns:
        Mapping from pair identifiers to split names.
    """
    identifiers_by_sentence: dict[str, list[str]] = {}
    for identifier, row in zip(identifiers, rows, strict=True):
        for sentence in (row["sentence_good"], row["sentence_bad"]):
            identifiers_by_sentence.setdefault(sentence, []).append(identifier)
    connected: dict[str, set[str]] = {
        identifier: {identifier} for identifier in identifiers
    }
    for members in identifiers_by_sentence.values():
        union = set().union(*(connected[identifier] for identifier in members))
        for identifier in union:
            connected[identifier] = union

    components = sorted(
        {tuple(sorted(members)) for members in connected.values()},
        key=lambda component: _stable_hash("|".join(component)),
    )
    # No source split exists: target a 70/10/20 distribution so the test
    # partition comfortably supplies its 1,024-pair cap before sampling.
    proportions = {"train": 0.7, "val": 0.1, "test": 0.2}
    result: dict[str, str] = {}
    counts = dict.fromkeys(proportions, 0)
    for component in components:
        split = max(
            proportions,
            key=lambda name: proportions[name] * len(identifiers) - counts[name],
        )
        result.update(dict.fromkeys(component, split))
        counts[split] += len(component)
    return result


def _stable_hash(identifier: str) -> str:
    """Return a stable ordering key for a pair identifier.

    Args:
        identifier:
            Identifier to hash.

    Returns:
        Hexadecimal SHA-256 digest of the identifier.
    """
    return hashlib.sha256(identifier.encode()).hexdigest()


def _pair_id(*, pair: dict[str, str], index: int) -> str:
    """Return a stable identifier for a source pair.

    Args:
        pair:
            Source pair record and its linguistic metadata.
        index:
            Source row index used when reporting missing identifiers.

    Returns:
        Identifier combining the source config, phenomenon, paradigm and item number.

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

    Args:
        pairs:
            Pair identifiers and records assigned to this split.
        limit:
            Maximum number of pairs to select.

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


def _validate_sentence_labels(*, rows: list[dict[str, str]]) -> None:
    """Reject source sentences assigned both acceptability labels.

    Args:
        rows:
            Deduplicated source pair records.

    Raises:
        ValueError:
            If any sentence occurs with contradictory labels.
    """
    labels_by_sentence: dict[str, str] = {}
    for pair in rows:
        for sentence, label in (
            (pair["sentence_good"], "correct"),
            (pair["sentence_bad"], "incorrect"),
        ):
            previous_label = labels_by_sentence.setdefault(sentence, label)
            if previous_label != label:
                raise ValueError(
                    f"BLiMP-NL sentence has contradictory labels: {sentence!r}"
                )


def _validate_split_texts(
    *, pairs_by_split: dict[str, list[tuple[str, dict[str, str]]]]
) -> None:
    """Reject any sentence text appearing in multiple assigned splits.

    Args:
        pairs_by_split:
            Source pairs grouped by their assigned split.

    Raises:
        ValueError:
            If sentence text occurs in more than one split.
    """
    split_by_sentence: dict[str, str] = {}
    for split, pairs in pairs_by_split.items():
        for _, pair in pairs:
            for sentence in (pair["sentence_good"], pair["sentence_bad"]):
                previous_split = split_by_sentence.setdefault(sentence, split)
                if previous_split != split:
                    raise ValueError(
                        f"BLiMP-NL sentence occurs in multiple splits: {sentence!r}"
                    )


if __name__ == "__main__":
    main()
