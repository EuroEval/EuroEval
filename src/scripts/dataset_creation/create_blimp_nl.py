"""Create a Dutch sentence-level acceptability dataset from BLiMP-NL."""

import hashlib

from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset

CAPS = {"train": 1024, "val": 256, "test": 2048}
SOURCE_REPO = "juletxara/blimp-nl"
TARGET_REPO = "EuroEval/blimp-nl-mini"


def main() -> None:
    """Create and publish BLiMP-NL-mini.

    Raises:
        ValueError:
            If the source splits or sentence labels are invalid.
    """
    pairs: dict[tuple[str, str], dict[str, str]] = {}
    for config in sorted(get_dataset_config_names(SOURCE_REPO)):
        source = load_dataset(SOURCE_REPO, config)
        if set(source) != {"train"}:
            raise ValueError(f"Unexpected split shape in {config}: {set(source)}")
        for record in source["train"]:
            good, bad = record["sentence_good"], record["sentence_bad"]
            if good == bad:
                continue
            pair = {
                "sentence_good": good,
                "sentence_bad": bad,
                "linguistic_phenomenon": record["linguistic_phenomenon"],
                "paradigm": record["paradigm"],
            }
            if not all(value.strip() for value in pair.values()):
                raise ValueError("BLiMP-NL records must contain all pair fields")
            key = (good, bad)
            if key not in pairs or tuple(pair.values()) < tuple(pairs[key].values()):
                pairs[key] = pair

    rows = sorted(pairs.values(), key=lambda pair: tuple(pair.values()))
    labels: dict[str, str] = {}
    for pair in rows:
        for text, label in (
            (pair["sentence_good"], "correct"),
            (pair["sentence_bad"], "incorrect"),
        ):
            if text in labels and labels[text] != label:
                raise ValueError(
                    f"BLiMP-NL sentence has contradictory labels: {text!r}"
                )
            labels[text] = label

    # Connected components keep every repeated sentence inside one split.
    by_sentence: dict[str, list[int]] = {}
    for index, pair in enumerate(rows):
        for text in (pair["sentence_good"], pair["sentence_bad"]):
            by_sentence.setdefault(text, []).append(index)
    remaining = set(range(len(rows)))
    components = []
    while remaining:
        component, pending = set(), [remaining.pop()]
        while pending:
            index = pending.pop()
            if index in component:
                continue
            component.add(index)
            pair = rows[index]
            for text in (pair["sentence_good"], pair["sentence_bad"]):
                pending.extend(by_sentence[text])
            remaining.difference_update(component)
        components.append(sorted(component))

    proportions = {"train": 0.7, "val": 0.1, "test": 0.2}
    counts = dict.fromkeys(proportions, 0)
    assigned = {split: [] for split in proportions}
    components.sort(
        key=lambda group: hashlib.sha256(
            "|".join(str(index) for index in group).encode()
        ).hexdigest()
    )
    # Sorting row content above makes component indices independent of source order.
    for component in components:
        split = max(
            proportions, key=lambda name: proportions[name] * len(rows) - counts[name]
        )
        counts[split] += len(component)
        assigned[split].extend(rows[index] for index in component)

    result = {}
    for split, limit in CAPS.items():
        strata: dict[tuple[str, str], list[dict[str, str]]] = {}
        for pair in assigned[split]:
            key = (pair["linguistic_phenomenon"], pair["paradigm"])
            strata.setdefault(key, []).append(pair)
        groups = [
            sorted(
                group,
                key=lambda pair: hashlib.sha256(
                    f"{pair['sentence_good']}\0{pair['sentence_bad']}".encode()
                ).hexdigest(),
            )
            for _, group in sorted(strata.items())
        ]
        selected = []
        for index in range(max(map(len, groups), default=0)):
            selected.extend(group[index] for group in groups if index < len(group))
            if len(selected) >= limit // 2:
                break
        examples = [
            {"text": pair[key], "label": label}
            for pair in selected[: limit // 2]
            for key, label in (
                ("sentence_good", "correct"),
                ("sentence_bad", "incorrect"),
            )
        ]
        result[split] = Dataset.from_list(examples)

    dataset = DatasetDict(result)
    dataset.push_to_hub(TARGET_REPO, private=False)


if __name__ == "__main__":
    main()
