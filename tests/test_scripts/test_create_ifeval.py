"""Tests for the IFEval dataset creation script."""

import pytest

from src.scripts.dataset_creation import create_ifeval


def test_cli_accepts_multiple_language_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI parser accepts repeated language options.

    Args:
        monkeypatch:
            Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(
        "sys.argv", ["create_ifeval.py", "--language", "nb", "--language", "nn"]
    )

    assert create_ifeval._parse_args().language == ["nb", "nn"]


def test_main_rejects_unsupported_language() -> None:
    """Unsupported language codes fail before loading any dataset."""
    with pytest.raises(ValueError, match="Unsupported language.*xx"):
        create_ifeval.main(languages=["xx"])


def test_main_selects_requested_languages_without_deleting_repositories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only selected languages are uploaded, privately, without repository deletion.

    Args:
        monkeypatch:
            Pytest monkeypatch fixture.
    """
    loaded_sources: list[str] = []
    datasets: list[MockDataset] = []

    def load_dataset(repo_id: str, name: str | None = None) -> MockDataset:
        loaded_sources.append(repo_id)
        dataset = MockDataset()
        datasets.append(dataset)
        return dataset

    monkeypatch.setattr(create_ifeval, "load_dataset", load_dataset)
    monkeypatch.setattr(
        create_ifeval, "DatasetDict", lambda splits: {"test": splits["test"]}
    )

    create_ifeval.main(languages=["nb", "nn"])

    assert loaded_sources == ["danish-foundation-models/multi-ifeval"] * 2
    assert [dataset.map_called for dataset in datasets] == [True, True]
    assert [dataset.selected_columns for dataset in datasets] == [
        ["text", "target_text"],
        ["text", "target_text"],
    ]
    assert [dataset.push_arguments for dataset in datasets] == [
        {"repo_id": "EuroEval/ifeval-nb", "private": True},
        {"repo_id": "EuroEval/ifeval-nn", "private": True},
    ]


class MockDataset:
    """Provide the dataset operations used by the creation script."""

    column_names = {"test": ["prompt", "instruction_id_list", "kwargs"]}

    def __init__(self) -> None:
        """Initialise recorded dataset operations."""
        self.map_called = False
        self.selected_columns: list[str] | None = None
        self.push_arguments: dict[str, str | bool] | None = None

    def __len__(self) -> int:
        """Return the number of dataset splits."""
        return 1

    def keys(self) -> list[str]:
        """Return the dataset split names."""
        return ["test"]

    def map(self, function: object) -> "MockDataset":
        """Record a mapping operation and return this dataset.

        Args:
            function:
                Mapping function (unused by this mock).

        Returns:
            This dataset.
        """
        self.map_called = True
        return self

    def push_to_hub(self, *, repo_id: str, private: bool) -> None:
        """Record Hub publication arguments.

        Args:
            repo_id:
                Target Hub repository ID.
            private:
                Whether to create the repository privately.
        """
        self.push_arguments = {"repo_id": repo_id, "private": private}

    def select_columns(self, columns: list[str]) -> "MockDataset":
        """Record the selected columns and return this dataset.

        Args:
            columns:
                Column names to retain.

        Returns:
            This dataset.
        """
        self.selected_columns = columns
        return self
