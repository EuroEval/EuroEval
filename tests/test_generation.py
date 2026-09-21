"""Tests for the `generation` module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from euroeval.enums import TaskGroup
from euroeval.generation import generate


class TestBPCacheNamespace:
    """Tests that BPC runs use a separate on-disk cache from MCF runs."""

    @pytest.mark.parametrize(
        ("use_bits_per_character", "expected_cache_name"),
        [
            (False, "fake-ds-model-outputs-test.json"),
            (True, "fake-ds-bpc-model-outputs-test.json"),
        ],
        ids=["MCF legacy cache name", "BPC cache name"],
    )
    def test_cache_name_namespaces_bpc_runs(
        self,
        dataset_config_mock: MagicMock,
        model_config_mock: MagicMock,
        use_bits_per_character: bool,
        expected_cache_name: str,
    ) -> None:
        """BPC is namespaced while MCF keeps its legacy cache name."""
        with patch("euroeval.generation.ModelCache") as mock_cache:
            generate(
                model=MagicMock(),
                datasets=[],
                model_config=model_config_mock,
                dataset_config=dataset_config_mock,
                benchmark_config=_make_benchmark_config(use_bits_per_character),
            )

        assert mock_cache.call_args.kwargs["cache_name"] == expected_cache_name


def _make_benchmark_config(use_bits_per_character: bool) -> MagicMock:
    """Build a minimal BenchmarkConfig stand-in.

    Args:
        use_bits_per_character: Whether to use BPC scoring to flag on the config.

    Returns:
        A MagicMock with `scoring_method`, `debug`, and `progress_bar` set.
    """
    bc = MagicMock()
    bc.use_bits_per_character = use_bits_per_character
    bc.debug = False
    bc.progress_bar = False
    return bc


@pytest.fixture
def dataset_config_mock() -> MagicMock:
    """A minimal DatasetConfig stand-in with the fields `generate` reads.

    Returns:
        A MagicMock with `name`, `max_generated_tokens`, and `task.task_group` set.
    """
    cfg = MagicMock()
    cfg.name = "fake-ds"
    cfg.max_generated_tokens = 1
    cfg.task.task_group = TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
    return cfg


@pytest.fixture
def model_config_mock(tmp_path: Path) -> MagicMock:
    """A minimal ModelConfig stand-in pointing at a temporary cache dir.

    Args:
        tmp_path: pytest-supplied per-test temporary directory.

    Returns:
        A MagicMock with `model_id` and `model_cache_dir` set.
    """
    cfg = MagicMock()
    cfg.model_id = "fake-model"
    cfg.model_cache_dir = str(tmp_path)
    return cfg
