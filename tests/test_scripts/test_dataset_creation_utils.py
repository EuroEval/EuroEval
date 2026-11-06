"""Tests for shared dataset-creation processing interfaces."""

import random

import pandas as pd

from src.scripts.dataset_creation.dataset_processing import (
    cap_dataframe,
    dataset_dict_from_frames,
    filter_by_character_lengths,
    format_multiple_choice_record,
    process_ner_frame,
)


def test_cap_dataframe_preserves_source_order_and_resets_index() -> None:
    """Capping one source frame does not shuffle or mix source rows."""
    frame = pd.DataFrame({"text": ["a", "b", "c"]}, index=[5, 6, 7])

    result = cap_dataframe(frame=frame, maximum=2)

    assert result["text"].tolist() == ["a", "b"]
    assert result.index.tolist() == [0, 1]


def test_dataset_dict_from_frames_retains_requested_columns() -> None:
    """Frame conversion sets the output splits and column order explicitly."""
    frames = {
        "train": pd.DataFrame({"text": ["one"], "label": ["yes"], "extra": [0]}),
        "val": pd.DataFrame({"text": ["two"], "label": ["no"], "extra": [1]}),
        "test": pd.DataFrame({"text": ["three"], "label": ["yes"], "extra": [2]}),
    }

    result = dataset_dict_from_frames(frames=frames, columns=["text", "label"])

    assert list(result) == ["train", "val", "test"]
    assert result["train"].column_names == ["text", "label"]
    assert result["val"][0]["text"] == "two"


def test_filter_by_character_lengths_applies_all_inclusive_bounds() -> None:
    """Rows must satisfy every requested inclusive text-length range."""
    frame = pd.DataFrame(
        {"context": ["x" * 2, "x" * 3, "x" * 4], "question": ["q", "q", "q"]}
    )

    result = filter_by_character_lengths(
        frame=frame, bounds={"context": (3, 4), "question": (1, 1)}
    )

    assert result["context"].str.len().tolist() == [3, 4]


def test_format_multiple_choice_record_tracks_shuffled_correct_option() -> None:
    """The label identifies the correct answer after deterministic shuffling."""
    result = format_multiple_choice_record(
        prompt="Pick one",
        correct_answer="right",
        distractors=["wrong one", "wrong two"],
        letters=["a", "b", "c"],
        choices_label="Choices",
        rng=random.Random(4242),
    )

    assert result["text"].startswith("Pick one\nChoices:\n")
    answer_line = next(
        line for line in result["text"].splitlines() if line.startswith(result["label"])
    )
    assert answer_line.endswith("right")


def test_process_ner_frame_maps_dataset_specific_tags() -> None:
    """NER conversion retains tokens and applies the provided label schema."""
    frame = pd.DataFrame(
        {"tokens": [["Euro", "Eval"]], "ner_tags": [[0, 1]], "other": ["drop"]}
    )

    result = process_ner_frame(frame=frame, label_mapping={0: "O", 1: "B-PER"})

    assert result.to_dict(orient="records") == [
        {"tokens": ["Euro", "Eval"], "text": "Euro Eval", "labels": ["O", "B-PER"]}
    ]
