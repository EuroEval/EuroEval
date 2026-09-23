"""Abstract base class for zero-shot classifier adapters."""

import typing as t
from abc import ABC, abstractmethod

from ..exceptions import NeedsExtraInstalled

if t.TYPE_CHECKING:
    from ..data_models import BenchmarkConfig, ModelConfig


class ZeroShotClassifierAdapter(ABC):
    """Abstract adapter wrapping a non-generative zero-shot "decision" model.

    A zero-shot classifier adapter loads a model through its own package (rather
    than through `transformers`, vLLM or LiteLLM) and answers typed classification
    questions with calibrated per-label probabilities. Adapters are registered in
    `euroeval.zero_shot_adapters.ADAPTERS` and are tried in order by
    `ZeroShotClassifierModel`.

    Attributes:
        name:
            A short, human-readable name of the adapter, used in logging.
        variants:
            The list of variants (checkpoint selectors) accepted through the
            `model_id#param` syntax. An empty list means no parameter is allowed.
            Not to be confused with `BenchmarkModule.allowed_params`, an unrelated
            per-module tokeniser-regex mapping used by finetunable modules.
        max_length:
            The maximum input length (in tokens) supported by the underlying model.
    """

    name: str
    variants: list[str]
    max_length: int

    @abstractmethod
    def __init__(
        self, model_config: "ModelConfig", benchmark_config: "BenchmarkConfig"
    ) -> None:
        """Load the model.

        Args:
            model_config:
                The model configuration.
            benchmark_config:
                The benchmark configuration.
        """
        ...

    @abstractmethod
    def classify(
        self, texts: list[str], candidate_labels: list[str], instructions: str
    ) -> list[dict[str, float]]:
        """Classify each text against the candidate labels.

        Args:
            texts:
                The texts to classify.
            candidate_labels:
                The candidate labels to classify each text into.
            instructions:
                The instructions describing the classification task, derived from
                the dataset's prompt/instruction template.

        Returns:
            A list, with one dictionary per text, mapping each candidate label to
            its predicted probability. The probabilities for a single text need not
            sum to exactly 1, but should be non-negative.
        """
        ...

    @classmethod
    @abstractmethod
    def matches(
        cls, model_id: str, benchmark_config: "BenchmarkConfig"
    ) -> "bool | NeedsExtraInstalled":
        """Check whether the given model ID is handled by this adapter.

        Args:
            model_id:
                The model ID to check, without revision or parameter suffixes.
            benchmark_config:
                The benchmark configuration.

        Returns:
            Whether the model ID is handled by this adapter, or a
            `NeedsExtraInstalled` error if the adapter's required package is not
            installed and this cannot be determined without it.
        """
        ...

    @classmethod
    @abstractmethod
    def num_params(cls, model_id: str, param: str | None) -> int:
        """Get the number of parameters of the given model variant.

        Args:
            model_id:
                The model ID, without revision or parameter suffixes.
            param:
                The parameter (variant) of the model, or None if no parameter was
                specified.

        Returns:
            The number of parameters in the model, or -1 if it is unknown.
        """
        ...
