"""Tests for the `safetensors_utils` module."""

from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import NotASafetensorsRepoError

from euroeval.safetensors_utils import get_num_params_from_safetensors_metadata


class TestGetNumParamsFromSafetensorsMetadata:
    """Tests for the `get_num_params_from_safetensors_metadata` function."""

    @patch("euroeval.safetensors_utils.get_hf_token")
    @patch("euroeval.safetensors_utils.get_safetensors_metadata")
    def test_single_parameter_count_entry(
        self, mock_get_metadata: MagicMock, mock_get_hf_token: MagicMock
    ) -> None:
        """Test getting number of parameters with a single entry."""
        # Mock the token retrieval
        mock_get_hf_token.return_value = False

        # Mock the metadata with a single parameter count entry
        mock_metadata = MagicMock()
        mock_metadata.parameter_count = {"BF16": 1000000}
        mock_get_metadata.return_value = mock_metadata

        result = get_num_params_from_safetensors_metadata(
            model_id="test/model", revision="main", api_key=None
        )

        assert result == 1000000
        mock_get_hf_token.assert_called_once_with(api_key=None)
        mock_get_metadata.assert_called_once_with(
            repo_id="test/model", revision="main", token=False
        )

    @patch("euroeval.safetensors_utils.get_hf_token")
    @patch("euroeval.safetensors_utils.get_safetensors_metadata")
    def test_multiple_parameter_count_entries(
        self, mock_get_metadata: MagicMock, mock_get_hf_token: MagicMock
    ) -> None:
        """Test getting number of parameters with multiple entries."""
        # Mock the token retrieval
        mock_get_hf_token.return_value = False

        # Mock the metadata with multiple parameter count entries
        mock_metadata = MagicMock()
        mock_metadata.parameter_count = {
            "BF16": 1000000,
            "FP32": 1500000,
            "I32": 500000,
        }
        mock_get_metadata.return_value = mock_metadata

        result = get_num_params_from_safetensors_metadata(
            model_id="test/large-model", revision="main", api_key=None
        )

        # Should return the maximum value
        assert result == 1500000

    @patch("euroeval.safetensors_utils.get_hf_token")
    @patch("euroeval.safetensors_utils.get_safetensors_metadata")
    def test_not_a_safetensors_repo(
        self, mock_get_metadata: MagicMock, mock_get_hf_token: MagicMock
    ) -> None:
        """Test handling of models not in safetensors format."""
        # Mock the token retrieval
        mock_get_hf_token.return_value = False

        # Mock the function to raise NotASafetensorsRepoError
        mock_get_metadata.side_effect = NotASafetensorsRepoError("Not safetensors")

        result = get_num_params_from_safetensors_metadata(
            model_id="test/non-safetensors-model", revision="main", api_key=None
        )

        assert result is None

    @patch("euroeval.safetensors_utils.get_hf_token")
    @patch("euroeval.safetensors_utils.get_safetensors_metadata")
    def test_empty_parameter_count(
        self, mock_get_metadata: MagicMock, mock_get_hf_token: MagicMock
    ) -> None:
        """Test handling of models with empty parameter count dictionary."""
        # Mock the token retrieval
        mock_get_hf_token.return_value = False

        # Mock the metadata with an empty parameter count dictionary
        mock_metadata = MagicMock()
        mock_metadata.parameter_count = {}
        mock_get_metadata.return_value = mock_metadata

        result = get_num_params_from_safetensors_metadata(
            model_id="test/broken-model", revision="main", api_key=None
        )

        assert result is None

    @patch("euroeval.safetensors_utils.get_hf_token")
    @patch("euroeval.safetensors_utils.get_safetensors_metadata")
    def test_with_api_key(
        self, mock_get_metadata: MagicMock, mock_get_hf_token: MagicMock
    ) -> None:
        """Test that API key is properly passed through."""
        # Mock the token retrieval
        mock_get_hf_token.return_value = "test_token"

        # Mock the metadata
        mock_metadata = MagicMock()
        mock_metadata.parameter_count = {"FP16": 500000}
        mock_get_metadata.return_value = mock_metadata

        result = get_num_params_from_safetensors_metadata(
            model_id="test/private-model", revision="main", api_key="test_api_key"
        )

        assert result == 500000
        mock_get_hf_token.assert_called_once_with(api_key="test_api_key")
        mock_get_metadata.assert_called_once_with(
            repo_id="test/private-model", revision="main", token="test_token"
        )

    @patch("euroeval.safetensors_utils.get_hf_token")
    @patch("euroeval.safetensors_utils.get_safetensors_metadata")
    def test_with_different_revision(
        self, mock_get_metadata: MagicMock, mock_get_hf_token: MagicMock
    ) -> None:
        """Test that revision parameter is properly passed through."""
        # Mock the token retrieval
        mock_get_hf_token.return_value = False

        # Mock the metadata
        mock_metadata = MagicMock()
        mock_metadata.parameter_count = {"BF16": 750000}
        mock_get_metadata.return_value = mock_metadata

        result = get_num_params_from_safetensors_metadata(
            model_id="test/model", revision="v2.0", api_key=None
        )

        assert result == 750000
        mock_get_hf_token.assert_called_once_with(api_key=None)
        mock_get_metadata.assert_called_once_with(
            repo_id="test/model", revision="v2.0", token=False
        )
