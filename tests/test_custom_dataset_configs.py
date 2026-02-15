"""Tests for the `custom_dataset_configs` module."""

import logging
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, Mock, patch

import pytest

from euroeval.custom_dataset_configs import (
    load_custom_datasets_module,
    try_get_dataset_config_from_repo,
)
from euroeval.data_models import DatasetConfig
from euroeval.languages import DANISH
from euroeval.tasks import SENT


@pytest.fixture
def mock_dataset_config() -> DatasetConfig:
    """Create a mock dataset config for testing."""
    return DatasetConfig(
        name="test-dataset",
        pretty_name="Test Dataset",
        source="test_source",
        task=SENT,
        languages=[DANISH],
    )


def test_load_custom_datasets_module_nonexistent_file(tmp_path: Path) -> None:
    """Test loading a module from a nonexistent file."""
    nonexistent_file = tmp_path / "nonexistent.py"
    result = load_custom_datasets_module(nonexistent_file)
    assert result is None


def test_load_custom_datasets_module_valid_file(tmp_path: Path) -> None:
    """Test loading a valid custom datasets module."""
    # Create a valid Python module
    module_file = tmp_path / "custom_datasets.py"
    module_file.write_text("TEST_VALUE = 42\n")

    result = load_custom_datasets_module(module_file)
    assert result is not None
    assert isinstance(result, ModuleType)
    assert hasattr(result, "TEST_VALUE")
    assert result.TEST_VALUE == 42


def test_load_custom_datasets_module_invalid_python(tmp_path: Path) -> None:
    """Test loading a module with invalid Python syntax."""
    # Create an invalid Python module
    module_file = tmp_path / "invalid.py"
    module_file.write_text("this is not valid python ][")

    with pytest.raises(SyntaxError):
        load_custom_datasets_module(module_file)


@patch("euroeval.custom_dataset_configs.HfApi")
@patch("euroeval.custom_dataset_configs.get_hf_token")
def test_try_get_dataset_config_repo_not_exists(
    mock_get_token: Mock, mock_hf_api: Mock, tmp_path: Path
) -> None:
    """Test trying to get config from a non-existent repository."""
    mock_get_token.return_value = "fake_token"
    mock_api_instance = MagicMock()
    mock_api_instance.repo_exists.return_value = False
    mock_hf_api.return_value = mock_api_instance

    result = try_get_dataset_config_from_repo(
        dataset_id="nonexistent/dataset", api_key="fake_key", cache_dir=tmp_path
    )

    assert result is None
    mock_api_instance.repo_exists.assert_called_once_with(
        repo_id="nonexistent/dataset", repo_type="dataset"
    )


@patch("euroeval.custom_dataset_configs.HfApi")
@patch("euroeval.custom_dataset_configs.get_hf_token")
@patch("euroeval.custom_dataset_configs.log_once")
def test_try_get_dataset_config_no_euroeval_config(
    mock_log: Mock, mock_get_token: Mock, mock_hf_api: Mock, tmp_path: Path
) -> None:
    """Test trying to get config from a repo without euroeval_config.py."""
    mock_get_token.return_value = "fake_token"
    mock_api_instance = MagicMock()
    mock_api_instance.repo_exists.return_value = True
    mock_api_instance.list_repo_files.return_value = ["README.md", "data.csv"]
    mock_hf_api.return_value = mock_api_instance

    result = try_get_dataset_config_from_repo(
        dataset_id="test/dataset", api_key="fake_key", cache_dir=tmp_path
    )

    assert result is None
    mock_log.assert_called_once()
    assert "does not have a euroeval_config.py file" in mock_log.call_args[0][0]
    assert mock_log.call_args[1]["level"] == logging.WARNING


@patch("euroeval.custom_dataset_configs.HfApi")
@patch("euroeval.custom_dataset_configs.get_hf_token")
@patch("euroeval.custom_dataset_configs.load_custom_datasets_module")
def test_try_get_dataset_config_success(
    mock_load_module: Mock,
    mock_get_token: Mock,
    mock_hf_api: Mock,
    tmp_path: Path,
    mock_dataset_config: DatasetConfig,
) -> None:
    """Test successfully getting a dataset config from a repo."""
    mock_get_token.return_value = "fake_token"

    # Mock HF API instance
    mock_api_instance = MagicMock()
    mock_api_instance.repo_exists.return_value = True
    mock_api_instance.list_repo_files.return_value = ["euroeval_config.py", "README.md"]

    # Mock dataset info
    mock_split_info = {"name": "test", "splits": [{"name": "train"}, {"name": "test"}]}
    mock_api_instance.dataset_info.return_value.card_data.dataset_info = mock_split_info

    mock_hf_api.return_value = mock_api_instance

    # Mock module loading - create a real module-like object
    mock_module = type("MockModule", (), {})()
    mock_module.dataset_config = mock_dataset_config
    mock_module.other_var = 42
    mock_load_module.return_value = mock_module

    result = try_get_dataset_config_from_repo(
        dataset_id="test/dataset", api_key="fake_key", cache_dir=tmp_path
    )

    assert result is not None
    assert isinstance(result, DatasetConfig)
    assert result.name == "test/dataset"
    assert result.source == "test/dataset"


@patch("euroeval.custom_dataset_configs.HfApi")
@patch("euroeval.custom_dataset_configs.get_hf_token")
@patch("euroeval.custom_dataset_configs.load_custom_datasets_module")
@patch("euroeval.custom_dataset_configs.log_once")
def test_try_get_dataset_config_no_test_split(
    mock_log: Mock,
    mock_load_module: Mock,
    mock_get_token: Mock,
    mock_hf_api: Mock,
    tmp_path: Path,
    mock_dataset_config: DatasetConfig,
) -> None:
    """Test trying to get config from a repo without a test split."""
    mock_get_token.return_value = "fake_token"

    # Mock HF API instance
    mock_api_instance = MagicMock()
    mock_api_instance.repo_exists.return_value = True
    mock_api_instance.list_repo_files.return_value = ["euroeval_config.py"]

    # Mock dataset info without test split
    mock_split_info = {"name": "test", "splits": [{"name": "train"}]}
    mock_api_instance.dataset_info.return_value.card_data.dataset_info = mock_split_info

    mock_hf_api.return_value = mock_api_instance

    # Mock module loading - create a real module-like object
    mock_module = type("MockModule", (), {})()
    mock_module.dataset_config = mock_dataset_config
    mock_load_module.return_value = mock_module

    result = try_get_dataset_config_from_repo(
        dataset_id="test/dataset", api_key="fake_key", cache_dir=tmp_path
    )

    assert result is None
    # Check that error was logged about missing test split
    log_calls = [call[0][0] for call in mock_log.call_args_list]
    assert any("does not have a test split" in call for call in log_calls)


@patch("euroeval.custom_dataset_configs.HfApi")
@patch("euroeval.custom_dataset_configs.get_hf_token")
@patch("euroeval.custom_dataset_configs.load_custom_datasets_module")
@patch("euroeval.custom_dataset_configs.log_once")
def test_try_get_dataset_config_multiple_configs(
    mock_log: Mock,
    mock_load_module: Mock,
    mock_get_token: Mock,
    mock_hf_api: Mock,
    tmp_path: Path,
    mock_dataset_config: DatasetConfig,
) -> None:
    """Test trying to get config from a repo with multiple dataset configs."""
    mock_get_token.return_value = "fake_token"

    # Mock HF API instance
    mock_api_instance = MagicMock()
    mock_api_instance.repo_exists.return_value = True
    mock_api_instance.list_repo_files.return_value = ["euroeval_config.py"]
    mock_hf_api.return_value = mock_api_instance

    # Mock module loading with multiple configs - create a real module-like object
    mock_module = type("MockModule", (), {})()
    config2 = DatasetConfig(
        name="test2",
        pretty_name="Test2",
        source="source2",
        task=SENT,
        languages=[DANISH],
    )
    mock_module.config1 = mock_dataset_config
    mock_module.config2 = config2
    mock_module.other = "value"
    mock_load_module.return_value = mock_module

    result = try_get_dataset_config_from_repo(
        dataset_id="test/dataset", api_key="fake_key", cache_dir=tmp_path
    )

    assert result is None
    # Check that warning was logged about multiple configs
    log_calls = [call[0][0] for call in mock_log.call_args_list]
    assert any("has multiple dataset configurations" in call for call in log_calls)
