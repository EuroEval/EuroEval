"""Tests for the BLiMP-NL dataset creation script."""

from src.scripts.dataset_creation.create_blimp_nl import process_dataset


def test_pair_level_split_is_deterministic_and_caps_pairs() -> None:
    """Pair-level assignment and per-split caps are stable for repeated runs."""
    rows = [
        {
            "sentence_good": f"goed {index}",
            "sentence_bad": f"fout {index}",
            "linguistic_phenomenon": "syntax",
            "paradigm": "word_order",
            "item_nr": str(index),
        }
        for index in range(5000)
    ]

    first = process_dataset(rows=rows)
    second = process_dataset(rows=rows)

    assert {split: data.num_rows for split, data in first.items()} == {
        split: data.num_rows for split, data in second.items()
    }
    assert all(data.num_rows % 2 == 0 for data in first.values())
    assert first["train"].num_rows <= 2048
    assert first["val"].num_rows <= 512
    assert first["test"].num_rows <= 4096


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
