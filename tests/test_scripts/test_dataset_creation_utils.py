"""Tests for shared dataset-creation processing interfaces."""

import importlib
import random
from pathlib import Path

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from src.scripts.dataset_creation.dataset_processing import (
    build_split_dataset,
    cap_dataframe,
    dataset_dict_from_frames,
    filter_by_character_lengths,
    format_multiple_choice_record,
    process_multilingual_tweet_sentiment,
    process_ner_frame,
    sample_dataframe,
    split_generated_dataset,
)


def test_build_split_dataset_keeps_short_source_splits_separate() -> None:
    """Dataset construction caps each downloaded split without pooling rows."""
    records = {
        "train": pd.DataFrame({"text": ["train-a"], "label": ["x"]}),
        "dev": pd.DataFrame({"text": ["dev-a", "dev-b"], "label": ["x", "x"]}),
        "test": pd.DataFrame({"text": ["test-a"], "label": ["x"]}),
    }

    result = build_split_dataset(
        load_split=lambda split: records[split],
        source_splits={"train": "train", "val": "dev", "test": "test"},
        limits={"train": 2, "val": 1, "test": 2},
        columns=["text", "label"],
    )

    assert result["train"]["text"] == ["train-a"]
    assert result["val"]["text"] == ["dev-a"]
    assert result["test"]["text"] == ["test-a"]


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


def test_generated_dataset_caps_each_proportional_split() -> None:
    """Oversized generated data is capped per split; small data is not renamed."""
    oversized = pd.DataFrame(
        {"text": [str(index) for index in range(12_000)], "label": "a"}
    )
    dataset, truncated = split_generated_dataset(frame=oversized)

    assert truncated
    assert len(dataset["train"]) == 1024
    assert len(dataset["val"]) == 256
    assert len(dataset["test"]) == 2048
    source_rows = set().union(*(set(dataset[name]["text"]) for name in dataset))
    assert len(source_rows) == 3328

    small = pd.DataFrame({"text": [str(index) for index in range(10)], "label": "a"})
    small_dataset, small_truncated = split_generated_dataset(frame=small)
    assert not small_truncated
    assert sum(len(split) for split in small_dataset.values()) == 10


def test_ner_uk_entrypoint_preserves_undersized_test_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Ukrainian builder never fills test from source training examples."""
    scripts_directory = Path("src/scripts/dataset_creation").resolve()
    monkeypatch.syspath_prepend(str(scripts_directory))
    module = importlib.import_module("create_ner_uk")
    source = DatasetDict(
        {
            split: Dataset.from_pandas(
                pd.DataFrame({"tokens": [[f"{split}-sample"]], "ner_tags": [[0]]})
            )
            for split in ("train", "validation", "test")
        }
    )
    captured: dict[str, DatasetDict] = {}
    monkeypatch.setattr(module, "load_dataset", lambda **kwargs: source)
    monkeypatch.setattr(module, "preflight_private_upload", lambda **kwargs: None)
    monkeypatch.setattr(
        DatasetDict,
        "push_to_hub",
        lambda dataset, *args, **kwargs: captured.update(dataset=dataset),
    )

    module.main()

    assert captured["dataset"]["train"][0]["tokens"] == ["train-sample"]
    assert captured["dataset"]["test"][0]["tokens"] == ["test-sample"]
    assert len(captured["dataset"]["test"]) == 1


def test_process_ner_frame_maps_dataset_specific_tags() -> None:
    """NER conversion retains tokens and applies the provided label schema."""
    frame = pd.DataFrame(
        {"tokens": [["Euro", "Eval"]], "ner_tags": [[0, 1]], "other": ["drop"]}
    )

    result = process_ner_frame(frame=frame, label_mapping={0: "O", 1: "B-PER"})

    assert result.to_dict(orient="records") == [
        {"tokens": ["Euro", "Eval"], "text": "Euro Eval", "labels": ["O", "B-PER"]}
    ]


def test_sample_dataframe_handles_undersized_split_without_duplication() -> None:
    """A small source split remains intact rather than causing an oversample."""
    frame = pd.DataFrame({"source": ["train-a", "train-b"]})

    result = sample_dataframe(frame=frame, maximum=4)

    assert result["source"].tolist() == ["train-a", "train-b"]


def test_squad_entrypoint_scripts_import_processing_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQuAD scripts import the character filter from the processing module."""
    scripts_directory = Path("src/scripts/dataset_creation").resolve()
    monkeypatch.syspath_prepend(str(scripts_directory))

    for name in (
        "create_squad",
        "create_squad_it",
        "create_squad_nl",
        "create_squad_nl_old",
    ):
        module = importlib.import_module(name)
        assert callable(module.main)


def test_squad_nl_entrypoint_keeps_validation_out_of_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dutch SQuAD development rows remain validation-only."""
    scripts_directory = Path("src/scripts/dataset_creation").resolve()
    monkeypatch.syspath_prepend(str(scripts_directory))
    module = importlib.import_module("create_squad_nl")

    def split_frame(prefix: str) -> Dataset:
        return Dataset.from_pandas(
            pd.DataFrame(
                {
                    "context": [f"{prefix} context with enough characters"],
                    "question": [f"{prefix} question long enough"],
                    "answers": [{"answer_start": [0], "text": [prefix]}],
                }
            )
        )

    source = DatasetDict(
        {
            "train": split_frame("training"),
            "dev": split_frame("development"),
            "test": split_frame("testing"),
        }
    )
    captured: dict[str, DatasetDict] = {}
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: source)
    monkeypatch.setattr(module, "preflight_private_upload", lambda **kwargs: None)
    monkeypatch.setattr(
        DatasetDict,
        "push_to_hub",
        lambda dataset, *args, **kwargs: captured.update(dataset=dataset),
    )

    module.main()

    assert captured["dataset"]["val"][0]["context"].startswith("development")
    assert captured["dataset"]["test"][0]["context"].startswith("testing")


def test_tweet_processing_keeps_source_split_provenance() -> None:
    """Tweet deduplication and capping never transfer examples across splits."""
    source = DatasetDict(
        {
            split: Dataset.from_pandas(
                pd.DataFrame(
                    {"text": [f"{split}-one", f"{split}-two"], "label": [0, 1]}
                )
            )
            for split in ("train", "validation", "test")
        }
    )

    result = process_multilingual_tweet_sentiment(
        source=source,
        source_label_mapping={0: "negative", 1: "positive"},
        minimum_text_length=1,
        maximum_text_length=100,
    )

    assert set(result["train"]["text"]) == {"train-one", "train-two"}
    assert set(result["val"]["text"]) == {"validation-one", "validation-two"}
    assert set(result["test"]["text"]) == {"test-one", "test-two"}
