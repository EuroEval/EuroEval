"""Tests for the `split_utils` module."""

from unittest.mock import MagicMock, patch

from euroeval.split_utils import find_split, get_repo_split_names, get_repo_splits


class TestFindSplit:
    """Tests for the `find_split` function."""

    def test_find_split_with_exact_match(self) -> None:
        """Test finding split with exact keyword match."""
        splits = ["train", "validation", "test"]
        result = find_split(splits=splits, keyword="train")
        assert result == "train"

    def test_find_split_with_partial_match_returns_shortest(self) -> None:
        """Test that shortest matching split is returned."""
        splits = ["training", "train", "train_data", "validation"]
        result = find_split(splits=splits, keyword="train")
        assert result == "train"

    def test_find_split_no_match_returns_none(self) -> None:
        """Test that None is returned when no split matches keyword."""
        splits = ["training", "validation", "testing"]
        result = find_split(splits=splits, keyword="eval")
        assert result is None

    def test_find_split_case_insensitive(self) -> None:
        """Test that keyword matching is case insensitive."""
        splits = ["Train", "TRAINING", "Validation", "TEST"]
        result = find_split(splits=splits, keyword="train")
        assert result == "Train"

    def test_find_split_no_candidates(self) -> None:
        """Test with empty splits list."""
        splits: list[str] = []
        result = find_split(splits=splits, keyword="train")
        assert result is None

    def test_find_split_multiple_matches_same_length(self) -> None:
        """Test when multiple splits have the same length."""
        splits = ["train_a", "train_b", "validation"]
        # Should return first one in sorted order (alphabetically)
        result = find_split(splits=splits, keyword="train")
        assert result == "train_a"


class TestGetRepoSplitNames:
    """Tests for the `get_repo_split_names` function."""

    @patch("euroeval.split_utils.HfApi")
    def test_get_repo_split_names_mocked(self, mock_hf_api_class: MagicMock) -> None:
        """Test extraction of split names from mocked HfApi response."""
        # Mock the HfApi instance
        mock_api_instance = MagicMock()
        mock_hf_api_class.return_value = mock_api_instance

        # Mock the split objects (each split is a dict-like object with "name" key)
        mock_split1 = MagicMock()
        mock_split1.__getitem__.return_value = "train"
        mock_split2 = MagicMock()
        mock_split2.__getitem__.return_value = "test"

        mock_card_data = MagicMock()
        mock_card_data.dataset_info = {"splits": [mock_split1, mock_split2]}

        mock_dataset_info = MagicMock()
        mock_dataset_info.card_data = mock_card_data
        mock_api_instance.dataset_info.return_value = mock_dataset_info

        result = get_repo_split_names(
            hf_api=mock_api_instance, dataset_id="test/dataset"
        )

        assert result == ["train", "test"]
        mock_api_instance.dataset_info.assert_called_once_with(repo_id="test/dataset")


class TestGetRepoSplits:
    """Tests for the `get_repo_splits` function."""

    def test_get_repo_splits_all_splits_present(self) -> None:
        """Test returning all three splits when all are present."""
        with patch("euroeval.split_utils.get_repo_split_names") as mock_get_names:
            mock_get_names.return_value = ["train", "validation", "test"]

            result = get_repo_splits(hf_api=MagicMock(), dataset_id="test/dataset")

            assert result == ("train", "validation", "test")

    def test_get_repo_splits_missing_splits(self) -> None:
        """Test that None is returned for missing splits."""
        with patch("euroeval.split_utils.get_repo_split_names") as mock_get_names:
            mock_get_names.return_value = ["train", "test"]

            result = get_repo_splits(hf_api=MagicMock(), dataset_id="test/dataset")

            assert result == ("train", None, "test")

    def test_get_repo_splits_val_as_train_fallback(self) -> None:
        """Test that validation split can be used as training when train is missing."""
        with patch("euroeval.split_utils.get_repo_split_names") as mock_get_names:
            # Using "validation" as keyword should find "validation" split
            mock_get_names.return_value = ["validation", "test"]

            result = get_repo_splits(hf_api=MagicMock(), dataset_id="test/dataset")

            assert result == (None, "validation", "test")

    def test_get_repo_splits_no_splits(self) -> None:
        """Test when dataset has no splits defined."""
        with patch("euroeval.split_utils.get_repo_split_names") as mock_get_names:
            mock_get_names.return_value = []

            result = get_repo_splits(hf_api=MagicMock(), dataset_id="test/dataset")

            assert result == (None, None, None)

    def test_get_repo_splits_partial_matches(self) -> None:
        """Test with partial keyword matches."""
        with patch("euroeval.split_utils.get_repo_split_names") as mock_get_names:
            mock_get_names.return_value = ["training_set", "val", "testing"]

            result = get_repo_splits(hf_api=MagicMock(), dataset_id="test/dataset")

            assert result == ("training_set", "val", "testing")
