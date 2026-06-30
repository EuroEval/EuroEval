"""Tests for the `generation` module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from euroeval.enums import TaskGroup
from euroeval.generation import generate


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


class TestBPCacheNamespace:
    """Tests that BPC runs use a separate on-disk cache from MCF runs.

    Without the `-bpc` suffix, switching `--use-bits-per-character` between True/False
    for the same model+dataset would clobber each other's caches and silently return
    wrong results.
    """

    def test_bpc_cache_name_includes_suffix(
        self, dataset_config_mock: MagicMock, model_config_mock: MagicMock
    ) -> None:
        """In BPC mode the cache filename includes a `-bpc` segment."""
        bc = _make_benchmark_config(use_bits_per_character=True)
        with patch("euroeval.generation.ModelCache") as MockCache:
            generate(
                model=MagicMock(),
                datasets=[],
                model_config=model_config_mock,
                dataset_config=dataset_config_mock,
                benchmark_config=bc,
            )
            cache_name = MockCache.call_args.kwargs["cache_name"]
        assert "fake-ds-bpc-model-outputs" in cache_name

    def test_mcf_cache_name_omits_suffix(
        self, dataset_config_mock: MagicMock, model_config_mock: MagicMock
    ) -> None:
        """In MCF mode the cache filename matches the legacy unsuffixed pattern.

        This is the bit-identity acceptance criterion for MCF runs: their cache
        path must not change when BPC support is added.
        """
        bc = _make_benchmark_config(use_bits_per_character=False)
        with patch("euroeval.generation.ModelCache") as MockCache:
            generate(
                model=MagicMock(),
                datasets=[],
                model_config=model_config_mock,
                dataset_config=dataset_config_mock,
                benchmark_config=bc,
            )
            cache_name = MockCache.call_args.kwargs["cache_name"]
        assert "fake-ds-model-outputs" in cache_name
        assert "-bpc-" not in cache_name

    def test_bpc_and_mcf_filenames_differ(
        self, dataset_config_mock: MagicMock, model_config_mock: MagicMock
    ) -> None:
        """The two evaluation types resolve to different on-disk cache files."""
        with patch("euroeval.generation.ModelCache") as MockCache:
            generate(
                model=MagicMock(),
                datasets=[],
                model_config=model_config_mock,
                dataset_config=dataset_config_mock,
                benchmark_config=_make_benchmark_config(False),
            )
            mcf_name = MockCache.call_args.kwargs["cache_name"]
            generate(
                model=MagicMock(),
                datasets=[],
                model_config=model_config_mock,
                dataset_config=dataset_config_mock,
                benchmark_config=_make_benchmark_config(True),
            )
            bpc_name = MockCache.call_args.kwargs["cache_name"]
        assert mcf_name != bpc_name
