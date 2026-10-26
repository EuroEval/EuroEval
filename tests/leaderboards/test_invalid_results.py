"""Tests for the public list of invalid (model, dataset) results."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from euroeval.dataset_configs.danish import ANGRY_TWEETS_CONFIG, DALA_CONFIG
from leaderboards.invalid_results import (
    InvalidResult,
    invalid_datasets_by_model,
    load_invalid_results,
)

VALID_ENTRY = (
    "- model: org/model\n"
    f"  dataset: {ANGRY_TWEETS_CONFIG.name}\n"
    "  reason: Trained on the test split.\n"
    "  source: https://example.com/model-card\n"
    "  date: 2026-01-13\n"
)


def test_bundled_file_loads() -> None:
    """The bundled file passes validation, whatever entries it holds."""
    invalid_results = load_invalid_results()

    assert all(isinstance(result, InvalidResult) for result in invalid_results)


@pytest.mark.parametrize("content", ["", "# Only a comment.\n", "[]\n"])
def test_empty_file_gives_no_entries(tmp_path: Path, content: str) -> None:
    """An empty, comment-only or empty-list file has no entries.

    Args:
        tmp_path:
            Temporary directory for files created by the test.
        content:
            Content supplied to the test.
    """
    assert load_invalid_results(path=_write(tmp_path=tmp_path, content=content)) == []


def _write(tmp_path: Path, content: str) -> Path:
    """Write an invalid results file.

    Args:
        tmp_path:
            The directory to write the file in.
        content:
            The file content.

    Returns:
        The path to the written file.
    """
    path = tmp_path / "invalid_results.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_entry_matches_every_variant_of_the_model() -> None:
    """An entry covers split, shot, revision and parameter variants only."""
    invalid_result = InvalidResult(
        model="org/model",
        dataset=ANGRY_TWEETS_CONFIG.name,
        reason="Trained on the test split.",
        source="https://example.com/model-card",
        date=dt.date(2026, 1, 13),
    )
    model_ids = [
        "org/model",
        "org/model (val)",
        "org/model (zero-shot, val)",
        "org/model@main",
        "org/model#no-thinking",
        "org/model-large",
        "org/other",
    ]

    model_datasets = {model_id: {ANGRY_TWEETS_CONFIG.name} for model_id in model_ids}

    assert invalid_datasets_by_model(
        invalid_results=[invalid_result], model_datasets=model_datasets
    ) == {
        model_id: {ANGRY_TWEETS_CONFIG.name}
        for model_id in model_ids
        if model_id not in {"org/model-large", "org/other"}
    }


def test_entry_without_result_on_dataset_raises() -> None:
    """An entry for a dataset that no variant of the model has a result on fails."""
    invalid_result = InvalidResult(
        model="org/model",
        dataset=DALA_CONFIG.name,
        reason="Trained on the test split.",
        source="https://example.com/model-card",
        date=dt.date(2026, 1, 13),
    )

    with pytest.raises(ValueError, match=f"org/model, {DALA_CONFIG.name}"):
        invalid_datasets_by_model(
            invalid_results=[invalid_result],
            model_datasets={
                "org/model": {ANGRY_TWEETS_CONFIG.name},
                "org/model (val)": {ANGRY_TWEETS_CONFIG.name},
            },
        )


@pytest.mark.parametrize(
    "date", ["'13/01/2026'", "'20260113'", "'2026-W03-2'", "2026-01-13 10:00:00"]
)
def test_malformed_date_raises(tmp_path: Path, date: str) -> None:
    """A date that is not exactly YYYY-MM-DD is rejected.

    Args:
        tmp_path:
            Temporary directory for files created by the test.
        date:
            Date supplied to the test.
    """
    content = VALID_ENTRY.replace("2026-01-13", date)
    path = _write(tmp_path=tmp_path, content=content)

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        load_invalid_results(path=path)


def test_missing_field_raises(tmp_path: Path) -> None:
    """An entry without a reason is rejected.

    Args:
        tmp_path:
            Temporary directory for files created by the test.
    """
    content = VALID_ENTRY.replace("  reason: Trained on the test split.\n", "")
    path = _write(tmp_path=tmp_path, content=content)

    with pytest.raises(ValueError, match="missing fields \\['reason'\\]"):
        load_invalid_results(path=path)


def test_no_entries_give_no_invalid_datasets() -> None:
    """Without entries no model has invalid datasets."""
    assert (
        invalid_datasets_by_model(
            invalid_results=[], model_datasets={"org/model": {ANGRY_TWEETS_CONFIG.name}}
        )
        == {}
    )


def test_single_entry_is_parsed(tmp_path: Path) -> None:
    """A complete entry is loaded with every field.

    Args:
        tmp_path:
            Temporary directory for files created by the test.
    """
    path = _write(tmp_path=tmp_path, content=VALID_ENTRY)

    assert load_invalid_results(path=path) == [
        InvalidResult(
            model="org/model",
            dataset=ANGRY_TWEETS_CONFIG.name,
            reason="Trained on the test split.",
            source="https://example.com/model-card",
            date=dt.date(2026, 1, 13),
        )
    ]


def test_unknown_dataset_raises(tmp_path: Path) -> None:
    """A dataset that EuroEval does not define is rejected.

    Args:
        tmp_path:
            Temporary directory for files created by the test.
    """
    content = VALID_ENTRY.replace(ANGRY_TWEETS_CONFIG.name, "no-such-dataset")
    path = _write(tmp_path=tmp_path, content=content)

    with pytest.raises(ValueError, match="no-such-dataset"):
        load_invalid_results(path=path)


def test_unknown_field_raises(tmp_path: Path) -> None:
    """An entry with a field outside the schema is rejected.

    Args:
        tmp_path:
            Temporary directory for files created by the test.
    """
    path = _write(tmp_path=tmp_path, content=VALID_ENTRY + "  score: 12.3\n")

    with pytest.raises(ValueError, match="unknown fields \\['score'\\]"):
        load_invalid_results(path=path)


def test_unknown_model_raises() -> None:
    """An entry for a model without results is rejected."""
    invalid_result = InvalidResult(
        model="org/missing",
        dataset=ANGRY_TWEETS_CONFIG.name,
        reason="Trained on the test split.",
        source="https://example.com/model-card",
        date=dt.date(2026, 1, 13),
    )

    with pytest.raises(ValueError, match="org/missing"):
        invalid_datasets_by_model(
            invalid_results=[invalid_result],
            model_datasets={"org/model": {ANGRY_TWEETS_CONFIG.name}},
        )
