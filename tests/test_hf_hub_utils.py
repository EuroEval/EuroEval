"""Tests for the `hf_hub_utils` module."""

from unittest.mock import MagicMock

import pytest
from huggingface_hub.errors import OfflineModeIsEnabled

from euroeval.hf_hub_utils import _repo_exists


class TestRepoExists:
    """Tests for the `_repo_exists` function."""

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

    @pytest.mark.parametrize(
        ("repo_exists", "expected"),
        [(False, False), (True, True)],
        ids=["missing repository", "existing repository"],
    )
    def test_returns_repo_exists_status(
        self, repo_exists: bool, expected: bool
    ) -> None:
        """Return the Hub API's status for available and missing repositories."""
        hf_api = MagicMock()
        hf_api.repo_exists.return_value = repo_exists
        dataset_id = "some/dataset-exists" if repo_exists else "some/dataset-missing"
        assert _repo_exists(hf_api=hf_api, dataset_id=dataset_id) is expected
