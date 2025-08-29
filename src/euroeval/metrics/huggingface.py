"""All the Hugging Face metrics used in EuroEval."""

import collections.abc as c
import logging
import typing as t

import evaluate
import numpy as np

from ..utils import HiddenPrints
from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from evaluate import EvaluationModule

    from ..data_models import BenchmarkConfig, DatasetConfig

logger: logging.Logger = logging.getLogger("euroeval")


class HuggingFaceMetric(Metric):
    """A metric which is implemented in the `evaluate` package.

    Attributes:
        name:
            The name of the metric in snake_case.
        pretty_name:
            The pretty name of the metric, used for display purposes.
        huggingface_id:
            The Hugging Face ID of the metric.
        results_key:
            The name of the key used to extract the metric scores from the results
            dictionary.
        compute_kwargs:
            Keyword arguments to pass to the metric's compute function. Defaults to
            an empty dictionary.
    """

    def __init__(
        self,
        name: str,
        pretty_name: str,
        huggingface_id: str,
        results_key: str,
        compute_kwargs: dict[str, t.Any] | None = None,
        postprocessing_fn: t.Callable[[float], tuple[float, str]] | None = None,
    ) -> None:
        """Initialise the Hugging Face metric.

        Args:
            name:
                The name of the metric in snake_case.
            pretty_name:
                The pretty name of the metric, used for display purposes.
            huggingface_id:
                The Hugging Face ID of the metric.
            results_key:
                The name of the key used to extract the metric scores from the results
                dictionary.
            compute_kwargs:
                Keyword arguments to pass to the metric's compute function. Defaults to
                an empty dictionary.
            postprocessing_fn:
                A function to apply to the metric scores after they are computed, taking
                the score to the postprocessed score along with its string
                representation. Defaults to x -> (100 * x, f"{x:.2%}").
        """
        super().__init__(
            name=name, pretty_name=pretty_name, postprocessing_fn=postprocessing_fn
        )
        self.huggingface_id = huggingface_id
        self.results_key = results_key
        self.compute_kwargs: dict[str, t.Any] = (
            dict() if compute_kwargs is None else compute_kwargs
        )
        self.metric: "EvaluationModule | None" = None

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate the metric score.

        Args:
            predictions:
                The model predictions.
            references:
                The ground truth references.
            dataset:
                The dataset used for evaluation. This is only used in case any
                additional metadata is used to compute the metrics.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.

        Returns:
            The calculated metric score, or None if the score should be ignored.
        """
        if self.metric is None:
            self.metric = evaluate.load(path=self.huggingface_id)

        with HiddenPrints():
            results = self.metric.compute(
                predictions=predictions, references=references, **self.compute_kwargs
            )

        # The metric returns None if we are running on multi-GPU and the current
        # process is not the main process
        if results is None:
            return None

        # Convert the results to a float score
        score = results[self.results_key]
        if isinstance(score, list):
            score = sum(score) / len(score)
        if isinstance(score, np.floating):
            score = float(score)

        return score


mcc_metric = HuggingFaceMetric(
    name="mcc",
    pretty_name="Matthew's Correlation Coefficient",
    huggingface_id="matthews_correlation",
    results_key="matthews_correlation",
)

macro_f1_metric = HuggingFaceMetric(
    name="macro_f1",
    pretty_name="Macro-average F1-score",
    huggingface_id="f1",
    results_key="f1",
    compute_kwargs=dict(average="macro"),
)

micro_f1_metric = HuggingFaceMetric(
    name="micro_f1",
    pretty_name="Micro-average F1-score with MISC tags",
    huggingface_id="seqeval",
    results_key="overall_f1",
)

micro_f1_no_misc_metric = HuggingFaceMetric(
    name="micro_f1_no_misc",
    pretty_name="Micro-average F1-score without MISC tags",
    huggingface_id="seqeval",
    results_key="overall_f1",
)

f1_metric = HuggingFaceMetric(
    name="f1",
    pretty_name="F1-score",
    huggingface_id="squad_v2",
    results_key="f1",
    postprocessing_fn=lambda x: (x, f"{x:.2f}%"),
)

em_metric = HuggingFaceMetric(
    name="em",
    pretty_name="Exact Match",
    huggingface_id="squad_v2",
    results_key="exact",
    postprocessing_fn=lambda x: (x, f"{x:.2f}%"),
)

bert_score_metric = HuggingFaceMetric(
    name="bertscore",
    pretty_name="BERTScore",
    huggingface_id="bertscore",
    results_key="f1",
    compute_kwargs=dict(
        model_type="microsoft/mdeberta-v3-base", device="auto", batch_size=1
    ),
)

rouge_l_metric = HuggingFaceMetric(
    name="rouge_l", pretty_name="ROUGE-L", huggingface_id="rouge", results_key="rougeL"
)

accuracy_metric = HuggingFaceMetric(
    name="accuracy",
    pretty_name="Accuracy",
    huggingface_id="accuracy",
    results_key="accuracy",
)
