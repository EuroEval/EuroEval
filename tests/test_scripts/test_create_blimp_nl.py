"""Tests for the BLiMP-NL dataset creation script."""

import pytest

from src.scripts.dataset_creation.create_blimp_nl import process_dataset


def test_contradictory_pair_sentences_are_rejected() -> None:
    """A sentence cannot be both grammatical and ungrammatical."""
    row = {
        "sentence_good": "Dit is dezelfde zin.",
        "sentence_bad": "Dit is dezelfde zin.",
        "linguistic_phenomenon": "syntax",
        "paradigm": "word_order",
        "item_nr": "1",
    }

    with pytest.raises(ValueError, match="identical sentences"):
        process_dataset(rows=[row])


def test_exact_duplicate_pairs_across_configs_are_removed() -> None:
    """Pair content, rather than source configuration, defines duplicates."""
    row = {
        "sentence_good": "Dit is goed.",
        "sentence_bad": "Dit goed is.",
        "linguistic_phenomenon": "syntax",
        "paradigm": "word_order",
        "item_nr": "1",
    }
    rows = [{**row, "config_name": "config-a"}, {**row, "config_name": "config-b"}]

    result = process_dataset(rows=rows)

    assert sum(dataset.num_rows for dataset in result.values()) == 2


def test_explicit_split_assignment_keeps_pairs_in_their_requested_splits() -> None:
    """Each pair is assigned to the requested split without separating members."""
    rows = [
        {
            "sentence_good": f"goed {index}",
            "sentence_bad": f"fout {index}",
            "linguistic_phenomenon": "syntax",
            "paradigm": "word_order",
            "item_nr": str(index),
        }
        for index in range(3)
    ]
    source_splits = {
        f":syntax:word_order:{index}": split
        for index, split in enumerate(("train", "val", "test"))
    }

    result = process_dataset(rows=rows, source_splits=source_splits)

    for split, index in zip(("train", "val", "test"), range(3), strict=True):
        assert result[split].to_list() == [
            {"text": f"goed {index}", "label": "correct"},
            {"text": f"fout {index}", "label": "incorrect"},
        ]


def test_pair_level_split_is_deterministic_and_caps_examples() -> None:
    """Pair-level splits retain both members and respect example caps."""
    rows = [
        {
            "sentence_good": f"goed {index}",
            "sentence_bad": f"fout {index}",
            "linguistic_phenomenon": f"phenomenon-{index % 3}",
            "paradigm": f"paradigm-{index % 7}",
            "item_nr": str(index),
        }
        for index in range(20_000)
    ]

    first = process_dataset(rows=rows)
    second = process_dataset(rows=rows)

    assert {split: data.to_list() for split, data in first.items()} == {
        split: data.to_list() for split, data in second.items()
    }
    assert {split: data.num_rows for split, data in first.items()} == {
        "train": 1024,
        "val": 256,
        "test": 2048,
    }
    for dataset in first.values():
        assert dataset.num_rows % 2 == 0
        for index in range(0, dataset.num_rows, 2):
            assert dataset[index]["label"] == "correct"
            assert dataset[index + 1]["label"] == "incorrect"
            assert (
                dataset[index]["text"].replace("goed", "fout")
                == dataset[index + 1]["text"]
            )


def test_pairs_are_converted_to_balanced_sentence_examples() -> None:
    """Both members of each source pair remain together with opposite labels."""
    rows = [
        {
            "sentence_good": "Dit is goed.",
            "sentence_bad": "Dit goed is.",
            "linguistic_phenomenon": "syntax",
            "paradigm": "word_order",
            "item_nr": "1",
        }
    ]

    result = process_dataset(rows=rows)

    assert (
        result["train"].num_rows + result["val"].num_rows + result["test"].num_rows == 2
    )
    nonempty = [dataset.to_list() for dataset in result.values() if dataset.num_rows]
    assert nonempty == [
        [
            {"text": "Dit is goed.", "label": "correct"},
            {"text": "Dit goed is.", "label": "incorrect"},
        ]
    ]


def test_sentence_text_cannot_overlap_across_splits() -> None:
    """A shared sentence cannot leak between explicitly assigned splits."""
    rows = [
        {
            "sentence_good": "Dit is gedeeld.",
            "sentence_bad": "Dit is fout 1.",
            "linguistic_phenomenon": "syntax",
            "paradigm": "word_order",
            "item_nr": "1",
        },
        {
            "sentence_good": "Dit is gedeeld.",
            "sentence_bad": "Dit is fout 2.",
            "linguistic_phenomenon": "syntax",
            "paradigm": "word_order",
            "item_nr": "2",
        },
    ]
    source_splits = {":syntax:word_order:1": "train", ":syntax:word_order:2": "test"}

    with pytest.raises(ValueError, match="multiple splits"):
        process_dataset(rows=rows, source_splits=source_splits)
