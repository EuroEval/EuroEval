"""Tests for the `generation` module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from euroeval.enums import EvaluationType
from euroeval.generation import generate


@pytest.fixture
def dataset_config_mock() -> MagicMock:
    """A minimal DatasetConfig stand-in with the fields `generate` reads.

    Returns:
        A MagicMock with `name` and `max_generated_tokens` set.
    """
    cfg = MagicMock()
    cfg.name = "fake-ds"
    cfg.max_generated_tokens = 1
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


def _make_benchmark_config(evaluation_type: EvaluationType) -> MagicMock:
    """Build a minimal BenchmarkConfig stand-in.

    Args:
        evaluation_type: Which evaluation formulation to flag on the config.

    Returns:
        A MagicMock with `evaluation_type`, `debug`, and `progress_bar` set.
    """
    bc = MagicMock()
    bc.evaluation_type = evaluation_type
    bc.debug = False
    bc.progress_bar = False
    return bc


class TestCFCacheNamespace:
    """Tests that CF runs use a separate on-disk cache from MCF runs.

    Without the `-cf` suffix, switching `--evaluation-type` between mcf/cf for
    the same model+dataset would clobber each other's caches and silently
    return wrong results (the `cf_scores` payload would be missing from the
    cached MCF entries, or vice versa).
    """

    def test_cf_cache_name_includes_suffix(
        self, dataset_config_mock: MagicMock, model_config_mock: MagicMock
    ) -> None:
        """In CF mode the cache filename includes a `-cf` segment."""
        bc = _make_benchmark_config(evaluation_type=EvaluationType.CF)
        with patch("euroeval.generation.ModelCache") as MockCache:
            generate(
                model=MagicMock(),
                datasets=[],
                model_config=model_config_mock,
                dataset_config=dataset_config_mock,
                benchmark_config=bc,
            )
            cache_name = MockCache.call_args.kwargs["cache_name"]
        assert "fake-ds-cf-model-outputs" in cache_name

    def test_mcf_cache_name_omits_suffix(
        self, dataset_config_mock: MagicMock, model_config_mock: MagicMock
    ) -> None:
        """In MCF mode the cache filename matches the legacy unsuffixed pattern.

        This is the bit-identity acceptance criterion for MCF runs: their cache
        path must not change when CF support is added.
        """
        bc = _make_benchmark_config(evaluation_type=EvaluationType.MCF)
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
        assert "-cf-" not in cache_name

    def test_cf_and_mcf_filenames_differ(
        self, dataset_config_mock: MagicMock, model_config_mock: MagicMock
    ) -> None:
        """The two evaluation types resolve to different on-disk cache files."""
        with patch("euroeval.generation.ModelCache") as MockCache:
            generate(
                model=MagicMock(),
                datasets=[],
                model_config=model_config_mock,
                dataset_config=dataset_config_mock,
                benchmark_config=_make_benchmark_config(EvaluationType.MCF),
            )
            mcf_name = MockCache.call_args.kwargs["cache_name"]
            generate(
                model=MagicMock(),
                datasets=[],
                model_config=model_config_mock,
                dataset_config=dataset_config_mock,
                benchmark_config=_make_benchmark_config(EvaluationType.CF),
            )
            cf_name = MockCache.call_args.kwargs["cache_name"]
        assert mcf_name != cf_name
