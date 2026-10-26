"""Tests for shared dataset-creation upload helpers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from src.scripts.dataset_creation import dataset_utils


def test_upload_private_checks_visibility_before_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An existing public repository is rejected before writing dataset bytes."""
    api = Mock()
    api.repo_info.return_value = SimpleNamespace(private=False)
    dataset = Mock()
    monkeypatch.setattr(dataset_utils, "HfApi", Mock(return_value=api))

    with pytest.raises(RuntimeError, match="not private"):
        dataset_utils.upload_private(dataset=dataset, repository_id="EuroEval/data")

    api.repo_info.assert_called_once_with(repo_id="EuroEval/data", repo_type="dataset")
    dataset.push_to_hub.assert_not_called()


def test_upload_private_creates_missing_repository_and_verifies_postcondition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing repository is allowed and visibility is checked after upload."""
    api = Mock()
    api.repo_info.side_effect = [
        RepositoryNotFoundError(response=Mock(), message="missing"),
        SimpleNamespace(private=True),
    ]
    dataset = Mock()
    monkeypatch.setattr(dataset_utils, "HfApi", Mock(return_value=api))

    dataset_utils.upload_private(dataset=dataset, repository_id="EuroEval/data")

    dataset.push_to_hub.assert_called_once_with(repo_id="EuroEval/data", private=True)
    assert api.repo_info.call_count == 2


def test_upload_private_rejects_public_repository_after_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The postcondition still detects a repository made public during upload."""
    api = Mock()
    api.repo_info.side_effect = [
        SimpleNamespace(private=True),
        SimpleNamespace(private=False),
    ]
    dataset = Mock()
    monkeypatch.setattr(dataset_utils, "HfApi", Mock(return_value=api))

    with pytest.raises(RuntimeError, match="not private"):
        dataset_utils.upload_private(dataset=dataset, repository_id="EuroEval/data")

    dataset.push_to_hub.assert_called_once_with(repo_id="EuroEval/data", private=True)
