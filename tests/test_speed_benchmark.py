"""Tests for the `speed_benchmark` module."""

import collections.abc as c
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import torch
from tqdm.auto import tqdm

from euroeval.benchmark_modules.base import BenchmarkModule
from euroeval.benchmark_modules.hf import HuggingFaceEncoderModel
from euroeval.data_models import BenchmarkConfig, HFModelInfo
from euroeval.dataset_configs import SPEED_CONFIG
from euroeval.model_config import get_model_config
from euroeval.speed_benchmark import benchmark_speed


@pytest.fixture(scope="module")
def model(
    encoder_model_id: str, benchmark_config: BenchmarkConfig
) -> Generator[BenchmarkModule, None, None]:
    """Yields a model."""
    with patch("euroeval.benchmark_modules.hf.get_model_repo_info") as mock_repo_info:
        # Fixed arguments for HFModelInfo
        mock_repo_info.return_value = HFModelInfo(
            pipeline_tag="fill-mask", tags=["da"], adapter_base_model_id=None
        )

        with patch(
            "euroeval.benchmark_modules.hf.load_model_and_tokeniser"
        ) as mock_loader:
            mock_pt_model = MagicMock()
            mock_pt_model.config.vocab_size = 1000
            mock_pt_model.config.num_params = 1000
            # Set max length attributes to specific integers to avoid comparison errors
            mock_pt_model.config.max_position_embeddings = 512
            mock_pt_model.config.max_sequence_length = 512
            mock_pt_model.config.model_max_length = 512
            mock_pt_model.config.n_positions = 512

            mock_pt_model.device = torch.device("cpu")
            mock_pt_model.side_effect = lambda *args, **kwargs: MagicMock()

            mock_tokeniser = MagicMock()
            mock_tokeniser.vocab_size = 1000
            mock_tokeniser.model_max_length = 512
            mock_tokeniser.max_model_input_sizes = {}
            mock_tokeniser.return_value = {"input_ids": torch.tensor([[1, 2]])}

            mock_loader.return_value = (mock_pt_model, mock_tokeniser)

            yield HuggingFaceEncoderModel(
                model_config=get_model_config(
                    model_id=encoder_model_id, benchmark_config=benchmark_config
                ),
                dataset_config=SPEED_CONFIG,
                benchmark_config=benchmark_config,
            )


class TestBenchmarkSpeed:
    """Tests for the `benchmark_speed` function."""

    @pytest.fixture(scope="class")
    def itr(self) -> Generator[tqdm, None, None]:
        """Yields an iterator with a progress bar."""
        yield tqdm(range(2))

    @pytest.fixture(scope="class")
    def scores(
        self, model: BenchmarkModule, benchmark_config: BenchmarkConfig
    ) -> Generator[c.Sequence[dict[str, float]], None, None]:
        """Yields the benchmark speed scores."""
        # Patch AutoTokenizer to avoid finding 'gpt2'
        with patch(
            "euroeval.speed_benchmark.AutoTokenizer.from_pretrained"
        ) as mock_gpt2:
            mock_gpt2_instance = MagicMock()
            # Return a list of length 10 when tokenizing
            mock_gpt2_instance.return_value = {"input_ids": [[1] * 10]}
            mock_gpt2.return_value = mock_gpt2_instance

            # Also patch pyinfer because it runs inference and might need real model
            # behavior. benchmark_speed calls pyinfer.InferenceReport(...).run()
            # The .run() method returns a dict with "Infer(p/sec)"
            with patch(
                "euroeval.speed_benchmark.pyinfer.InferenceReport"
            ) as mock_report:
                mock_report_instance = MagicMock()
                # .run() returns dict with speed
                mock_report_instance.run.return_value = {"Infer(p/sec)": 10.0}
                mock_report.return_value = mock_report_instance

                yield benchmark_speed(model=model, benchmark_config=benchmark_config)

    def test_scores_is_list(self, scores: list[dict[str, float]]) -> None:
        """Tests that the scores is a list."""
        assert isinstance(scores, list)

    def test_scores_contain_dicts(self, scores: list[dict[str, float]]) -> None:
        """Tests that the scores contain dicts."""
        assert all(isinstance(x, dict) for x in scores)

    def test_scores_dicts_keys(self, scores: list[dict[str, float]]) -> None:
        """Tests that the scores dicts have the correct keys."""
        assert all(set(x.keys()) == {"test_speed", "test_speed_short"} for x in scores)

    def test_scores_dicts_values_dtypes(self, scores: list[dict[str, float]]) -> None:
        """Tests that the scores dicts have the correct values dtypes."""
        assert all(
            all(isinstance(value, float) for value in x.values()) for x in scores
        )
