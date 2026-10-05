"""Tests for the `generation` module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from euroeval import generation
from euroeval.benchmark_modules.typesafe import TypesafeSystemOneModel
from euroeval.benchmark_modules.zero_shot_classifier import ZeroShotClassifierModel
from euroeval.data_models import GenerativeModelOutput
from euroeval.enums import BatchingPreference, TaskGroup
from euroeval.generation import generate, generate_single_iteration


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


@pytest.mark.parametrize("progress_bar", [True, False])
def test_single_sample_generation_preserves_predictions_and_progress(
    dataset_config: MagicMock, benchmark_config: MagicMock, progress_bar: bool
) -> None:
    """Single-sample generation retains predictions and honours progress settings."""
    assert (
        ZeroShotClassifierModel.batching_preference == BatchingPreference.SINGLE_SAMPLE
    )
    assert (
        TypesafeSystemOneModel.batching_preference == BatchingPreference.SINGLE_SAMPLE
    )

    dataset_config.prompt_label_mapping = {}
    dataset_config.bootstrap_samples = False
    benchmark_config.progress_bar = progress_bar
    benchmark_config.use_bits_per_character = False
    benchmark_config.debug = False

    model = MagicMock()
    model.batching_preference = BatchingPreference.SINGLE_SAMPLE
    model.generate.side_effect = lambda inputs: GenerativeModelOutput(
        sequences=["positive"] * len(inputs["text"])
    )
    model.extract_labels_from_generation.return_value = ["positive"]
    cache = MagicMock()
    progress_flags: list[bool] = []

    def track_progress(iterable: object, disable: bool) -> object:
        progress_flags.append(not disable)
        return iterable

    with patch.object(generation, "get_pbar", side_effect=track_progress):
        generate_single_iteration(
            dataset=Dataset.from_dict(
                {"text": ["first", "second"], "label": ["positive", "positive"]}
            ),
            model=model,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            cache=cache,
        )

    assert progress_flags == [progress_bar]
    generated_texts = [
        call.kwargs["inputs"]["text"] for call in model.generate.call_args_list
    ]
    assert generated_texts == [["first"], ["second"]]
    assert model.extract_labels_from_generation.call_count == 2
    assert all(
        call.kwargs["model_output"].sequences == ["positive"]
        for call in model.extract_labels_from_generation.call_args_list
    )
