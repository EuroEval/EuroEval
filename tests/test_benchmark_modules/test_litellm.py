"""Unit tests for the `litellm` module."""

from unittest.mock import MagicMock

import pytest

from euroeval.benchmark_modules.litellm import LiteLLMModel
from euroeval.enums import EvaluationType
from euroeval.exceptions import InvalidBenchmark


class TestCFGating:
    """Tests that the LiteLLM backend rejects Cloze Formulation evaluation.

    Per-token logprobs for forced completions are not reliably exposed across
    LiteLLM-supported providers, so CF must surface a clear `InvalidBenchmark`
    rather than silently degrading. The guard sits at the top of `generate`.
    """

    def _make_model(self, evaluation_type: EvaluationType) -> LiteLLMModel:
        """Build a minimally-initialised LiteLLMModel.

        Args:
            evaluation_type: Which evaluation formulation to flag on the config.

        Returns:
            A `LiteLLMModel` with only `benchmark_config` set; only `generate`
            should be called on it.
        """
        bc = MagicMock()
        bc.evaluation_type = evaluation_type
        model = object.__new__(LiteLLMModel)
        model.benchmark_config = bc
        return model

    def test_generate_raises_on_cf(self) -> None:
        """`LiteLLMModel.generate` raises immediately when CF is requested."""
        model = self._make_model(evaluation_type=EvaluationType.CF)
        with pytest.raises(InvalidBenchmark, match="Cloze Formulation"):
            model.generate(inputs={"text": ["prompt"]})

    def test_generate_error_message_points_to_vllm(self) -> None:
        """The rejection message tells the user vLLM is the supported backend."""
        model = self._make_model(evaluation_type=EvaluationType.CF)
        with pytest.raises(InvalidBenchmark, match="vLLM backend"):
            model.generate(inputs={"text": ["prompt"]})
