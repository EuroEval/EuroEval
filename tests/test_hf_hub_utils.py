"""Tests for the `hf_hub_utils` module."""

from unittest.mock import MagicMock

from huggingface_hub.errors import OfflineModeIsEnabled

from euroeval.hf_hub_utils import _repo_exists


class TestRepoExists:
    """Tests for the `_repo_exists` function."""

    def test_returns_true_when_repo_exists(self) -> None:
        """Test that ``_repo_exists`` returns True when the repo exists."""
        hf_api = MagicMock()
        hf_api.repo_exists.return_value = True
        assert _repo_exists(hf_api=hf_api, dataset_id="some/dataset-exists") is True

    def test_returns_false_when_repo_missing(self) -> None:
        """Test that ``_repo_exists`` returns False when the repo is missing."""
        hf_api = MagicMock()
        hf_api.repo_exists.return_value = False
        assert _repo_exists(hf_api=hf_api, dataset_id="some/dataset-missing") is False

    def test_returns_false_when_offline_mode_enabled(self) -> None:
        """Returns False rather than raising when ``HF_HUB_OFFLINE=1`` is set.

        Regression test for the crash reported in issue #1657, where running
        EuroEval with a valid local custom dataset under ``HF_HUB_OFFLINE=1``
        would propagate ``OfflineModeIsEnabled`` out of the Hub-existence check
        instead of letting the caller fall back to the local config.
        """
        hf_api = MagicMock()
        hf_api.repo_exists.side_effect = OfflineModeIsEnabled(
            "Cannot reach https://huggingface.co/api/datasets/some/dataset-offline: "
            "offline mode is enabled."
        )
        assert _repo_exists(hf_api=hf_api, dataset_id="some/dataset-offline") is False
