"""Public-interface tests for preparing MultiBFCL datasets without Hub writes."""

import json

import pytest
from datasets import Dataset

from euroeval.data_models import BenchmarkConfig
from euroeval.dataset_configs.danish import MULTI_BFCL_DA_CONFIG
from euroeval.metrics.tool_calling import tool_calling_accuracy
from scripts.dataset_creation import create_multibfcl


@pytest.fixture
def source_dataset(source_row: dict[str, str]) -> Dataset:
    """Provide 2,501 distinguishable source rows for split identity checks.

    Args:
        source_row:
            Template for each source record.

    Returns:
        Source records with a unique question in each row.
    """
    return Dataset.from_list(
        [
            source_row
            | {
                "question": json.dumps(
                    [[{"role": "user", "content": f"Find København {i}"}]]
                )
            }
            for i in range(2501)
        ]
    )


@pytest.fixture
def source_row() -> dict[str, str]:
    """Provide one representative JSON-encoded source record.

    Returns:
        A source record with a JSON-encoded question, schema and reference.
    """
    return {
        "question": json.dumps([[{"role": "user", "content": "Find København"}]]),
        "function": json.dumps(
            [{"name": "lookup", "parameters": {"required": ["term"]}}]
        ),
        "ground_truth": json.dumps([{"lookup": {"term": ["København"]}}]),
    }


def test_build_dataset_keeps_source_test_and_scores(
    monkeypatch: pytest.MonkeyPatch,
    source_dataset: Dataset,
    benchmark_config: BenchmarkConfig,
) -> None:
    """A converted record retains its schema, prompt and scorable reference."""

    def fake_load_dataset(*, path: str, name: str, split: str) -> Dataset:
        """Return a test split while checking the requested source subset."""
        assert (path, name, split) == ("syvai/multi-bfcl", "da", "test")
        return source_dataset

    monkeypatch.setattr(create_multibfcl, "load_dataset", fake_load_dataset)
    dataset, repo_id = create_multibfcl.build_dataset(language_code="da")
    assert repo_id == "EuroEval/multi-bfcl-da-mini"
    assert set(dataset) == {"val", "test"}
    assert len(dataset["test"]) == 2048
    assert len(dataset["val"]) == 256
    assert dataset["val"].column_names == dataset["test"].column_names
    assert set(dataset["val"]["text"]).isdisjoint(dataset["test"]["text"])
    shuffled = source_dataset.shuffle(seed=42)
    expected_questions = [
        json.loads(row["question"])[0][0]["content"] for row in shuffled
    ]
    assert [text.rsplit("Question: ", 1)[-1] for text in dataset["test"]["text"]] == (
        expected_questions[:2048]
    )
    assert [text.rsplit("Question: ", 1)[-1] for text in dataset["val"]["text"]] == (
        expected_questions[2048:2304]
    )
    assert len(expected_questions[2304:]) == 197
    record = dataset["test"][0]
    assert set(record) == {"text", "function", "target_text"}
    assert "Find København" in record["text"]
    assert (
        dataset["test"]["text"]
        == create_multibfcl.build_dataset(language_code="da")[0]["test"]["text"]
    )
    assert len(set(dataset["test"]["text"])) == 2048
    assert json.loads(record["function"])[0]["name"] == "lookup"
    assert json.loads(record["target_text"]) == [{"lookup": {"term": ["København"]}}]
    assert (
        tool_calling_accuracy(
            predictions=[
                json.dumps(
                    {
                        "tool_calls": [
                            {"function": "lookup", "arguments": {"term": "København"}}
                        ]
                    }
                )
            ],
            references=[record["target_text"]],
            dataset=dataset["test"].select([0]),
            dataset_config=MULTI_BFCL_DA_CONFIG,
            benchmark_config=benchmark_config,
        )
        == 1.0
    )


def test_build_dataset_preserves_system_and_user_messages(
    monkeypatch: pytest.MonkeyPatch, source_row: dict[str, str]
) -> None:
    """A multi-message source question retains its system instructions."""
    source_row["question"] = json.dumps(
        [
            [
                {"role": "system", "content": "Use the supplied context."},
                {"role": "user", "content": "Find København."},
            ]
        ]
    )

    def fake_load_dataset(*, path: str, name: str, split: str) -> Dataset:
        """Return enough test examples for conversion."""
        assert (path, name, split) == ("syvai/multi-bfcl", "da", "test")
        return Dataset.from_list([source_row] * 2304)

    monkeypatch.setattr(create_multibfcl, "load_dataset", fake_load_dataset)
    dataset, _ = create_multibfcl.build_dataset(language_code="da")
    assert dataset["test"][0]["text"].endswith(
        "System: Use the supplied context.\nQuestion: Find København."
    )


@pytest.mark.parametrize("size", [1, 2303])
def test_build_dataset_rejects_insufficient_source_rows(
    monkeypatch: pytest.MonkeyPatch, source_row: dict[str, str], size: int
) -> None:
    """A short source cannot silently produce overlapping or undersized splits."""

    def fake_load_dataset(*, path: str, name: str, split: str) -> Dataset:
        """Return fewer than the required number of source rows.

        Returns:
            An undersized source dataset.
        """
        return Dataset.from_list([source_row] * size)

    monkeypatch.setattr(create_multibfcl, "load_dataset", fake_load_dataset)
    with pytest.raises(
        ValueError, match=f"requires 2304 distinct source rows; found {size}"
    ):
        create_multibfcl.build_dataset(language_code="da")


def test_build_dataset_rejects_unknown_language() -> None:
    """Only EuroEval's non-English language subsets are published."""
    with pytest.raises(ValueError, match="Unsupported MultiBFCL language"):
        create_multibfcl.build_dataset(language_code="en")


def test_build_dataset_uses_portuguese_source(
    monkeypatch: pytest.MonkeyPatch, source_row: dict[str, str]
) -> None:
    """Portuguese source uses pt-pt while the EuroEval identity uses pt."""

    def fake_load_dataset(*, path: str, name: str, split: str) -> Dataset:
        """Confirm the selected source and return enough test examples.

        Returns:
            2,304 source test examples.
        """
        assert (path, name, split) == ("syvai/multi-bfcl", "pt-pt", "test")
        return Dataset.from_list([source_row] * 2304)

    monkeypatch.setattr(create_multibfcl, "load_dataset", fake_load_dataset)
    dataset, repo_id = create_multibfcl.build_dataset(language_code="pt")
    assert repo_id == "EuroEval/multi-bfcl-pt-mini"
    assert len(dataset["test"]) == 2048
    assert len(dataset["val"]) == 256


def test_main_publishes_every_language_privately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish every generated language dataset once to its private repository."""
    published: list[tuple[str, bool]] = []

    class DummyDataset:
        def __getitem__(self, split: str) -> list[None]:
            return [None]

        def push_to_hub(self, *, repo_id: str, private: bool) -> None:
            published.append((repo_id, private))

    def fake_build_dataset(language_code: str) -> tuple[DummyDataset, str]:
        return DummyDataset(), f"EuroEval/multi-bfcl-{language_code}-mini"

    monkeypatch.setattr(create_multibfcl, "build_dataset", fake_build_dataset)

    create_multibfcl.main()

    expected = [
        (f"EuroEval/multi-bfcl-{language_code}-mini", True)
        for language_code in create_multibfcl.LANGUAGE_SOURCES
    ]
    assert len(expected) == 31
    assert published == expected
