"""Public-interface tests for preparing MultiBFCL datasets without Hub writes."""

import json

import pytest
from datasets import Dataset

from euroeval.data_models import BenchmarkConfig
from euroeval.dataset_configs.danish import MULTI_BFCL_DA_CONFIG
from euroeval.metrics.tool_calling import tool_calling_accuracy
from scripts.dataset_creation import create_multibfcl


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
    source_row: dict[str, str],
    benchmark_config: BenchmarkConfig,
) -> None:
    """A converted record retains its schema, prompt and scorable reference."""

    def fake_load_dataset(*, path: str, name: str, split: str) -> Dataset:
        """Return a test split while checking the requested source subset."""
        assert (path, name, split) == ("syvai/multi-bfcl", "da", "test")
        return Dataset.from_list(
            [
                source_row
                | {
                    "question": json.dumps(
                        [[{"role": "user", "content": f"Find København {i}"}]]
                    )
                }
                for i in range(2050)
            ]
        )

    monkeypatch.setattr(create_multibfcl, "load_dataset", fake_load_dataset)
    dataset, repo_id = create_multibfcl.build_dataset(language_code="da")
    assert repo_id == "EuroEval/multi-bfcl-da-mini"
    assert list(dataset) == ["test"]
    assert len(dataset["test"]) == 2048
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


def test_build_dataset_rejects_unknown_language() -> None:
    """Only EuroEval's non-English language subsets are published."""
    with pytest.raises(ValueError, match="Unsupported MultiBFCL language"):
        create_multibfcl.build_dataset(language_code="en")


def test_build_dataset_uses_portuguese_source_and_unsuffixed_small_repo(
    monkeypatch: pytest.MonkeyPatch, source_row: dict[str, str]
) -> None:
    """Portuguese source uses pt-pt while the EuroEval identity uses pt."""

    def fake_load_dataset(*, path: str, name: str, split: str) -> Dataset:
        """Confirm the selected source and return a single test example.

        Returns:
            One source test example.
        """
        assert (path, name, split) == ("syvai/multi-bfcl", "pt-pt", "test")
        return Dataset.from_list([source_row])

    monkeypatch.setattr(create_multibfcl, "load_dataset", fake_load_dataset)
    dataset, repo_id = create_multibfcl.build_dataset(language_code="pt")
    assert repo_id == "EuroEval/multi-bfcl-pt"
    assert len(dataset["test"]) == 1
