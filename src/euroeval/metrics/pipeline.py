"""Metrics based on a scikit-learn Pipeline."""

import collections.abc as c
import logging
import typing as t
from pathlib import Path

import cloudpickle
import huggingface_hub as hf_hub
import numpy as np
from scipy.special import expit as sigmoid

from ..exceptions import InvalidBenchmark
from ..utils import unscramble
from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from sklearn.pipeline import Pipeline

    from ..data_models import BenchmarkConfig, DatasetConfig

logger: logging.Logger = logging.getLogger("euroeval")


T = t.TypeVar("T", bound=int | float | str | bool)


class PreprocessingFunction(t.Protocol):
    """A protocol for a preprocessing function."""

    def __call__(
        self, predictions: c.Sequence[int], dataset: "Dataset"
    ) -> c.Sequence[int]:
        """Preprocess the model predictions before they are passed to the pipeline.

        Args:
            predictions:
                The model predictions.
            dataset:
                The dataset used for evaluation. This is only used in case any
                additional metadata is used to compute the metrics.

        Returns:
            The preprocessed model predictions.
        """
        ...


class PipelineMetric(Metric):
    """Load a scikit-learn pipeline and use it to get scores from the predictions."""

    def __init__(
        self,
        name: str,
        pretty_name: str,
        pipeline_repo: str,
        pipeline_scoring_function: c.Callable[["Pipeline", c.Sequence], float],
        pipeline_file_name: str = "pipeline.pkl",
        preprocessing_fn: PreprocessingFunction | None = None,
        postprocessing_fn: c.Callable[[float], tuple[float, str]] | None = None,
    ) -> None:
        """Initialise the pipeline transform metric.

        Args:
            name:
                The name of the metric in snake_case.
            pretty_name:
                The pretty name of the metric, used for display purposes.
            pipeline_repo:
                The Hugging Face repository ID of the scikit-learn pipeline to load.
            pipeline_scoring_method:
                The method to use for scoring the predictions with the pipeline. Takes
                a 1D sequence of predictions and returns a float score.
            pipeline_file_name (optional):
                The name of the file to download from the Hugging Face repository.
                Defaults to "pipeline.joblib".
            preprocessing_fn (optional):
                A function to apply to the predictions before they are passed to the
                pipeline. This is useful for preprocessing the predictions to match
                the expected input format of the pipeline. Defaults to a no-op function
                that returns the input unchanged.
            postprocessing_fn (optional):
                A function to apply to the metric scores after they are computed,
                taking the score to the postprocessed score along with its string
                representation. Defaults to x -> (100 * x, f"{x:.2%}").
        """
        super().__init__(
            name=name, pretty_name=pretty_name, postprocessing_fn=postprocessing_fn
        )
        self.pipeline_repo = pipeline_repo
        self.pipeline_file_name = pipeline_file_name
        self.pipeline_scoring_function = pipeline_scoring_function
        self.pipeline: "Pipeline | None" = None
        self.preprocessing_fn = preprocessing_fn

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate the metric score using the scikit-learn pipeline.

        Args:
            predictions:
                The model predictions.
            references:
                Not used, but required for consistency with the Metric interface.
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
        if self.pipeline is None:
            self.pipeline = self._download_pipeline()
        if self.preprocessing_fn is not None:
            predictions = self.preprocessing_fn(
                predictions=predictions, dataset=dataset
            )
        return self.pipeline_scoring_function(self.pipeline, predictions)

    def _download_pipeline(self) -> "Pipeline":
        """Download the scikit-learn pipeline from the given URL.

        Returns:
            The downloaded scikit-learn pipeline.

        Raises:
            InvalidBenchmark:
                If the loading of the pipeline fails for any reason.
        """
        logger.debug(f"Loading pipeline from {self.pipeline_repo}...")
        folder_path = hf_hub.HfApi(
            token=unscramble("HjccJFhIozVymqXDVqTUTXKvYhZMTbfIjMxG_")
        ).snapshot_download(repo_id=self.pipeline_repo, repo_type="model")
        model_path = Path(folder_path, self.pipeline_file_name)
        try:
            with model_path.open(mode="rb") as f:
                pipeline = cloudpickle.load(f)
        except Exception as e:
            raise InvalidBenchmark(
                f"Failed to load pipeline from {self.pipeline_repo!r}: {e}"
            ) from e
        logger.debug(f"Successfully loaded pipeline: {pipeline}")
        return pipeline


### European Values Metric ###


def european_values_preprocessing_fn(
    predictions: c.Sequence[int], dataset: "Dataset"
) -> c.Sequence[int]:
    """Preprocess the model predictions for the European Values metric.

    Args:
        predictions:
            The model predictions, a sequence of integers representing the predicted
            choices for each question.
        dataset:
            The dataset used for evaluation. This is only used in case any additional
            metadata is used to compute the metrics.

    Returns:
        The preprocessed model predictions, a sequence of integers representing the
        final predicted choices for each question after any necessary aggregation and
        mapping.

    Raises:
        AssertionError:
            If the number of predictions is not a multiple of 53, which is required
            for the European Values metric.
    """
    num_questions = 53
    num_phrasings_per_question = 5

    # Convert the predictions to integers
    integer_predictions = []
    for prediction, idx_to_choice in zip(predictions, dataset["idx_to_choice"]):
        idx_to_choice = {
            int(idx): int(choice)
            for idx, choice in idx_to_choice.items()
            if choice is not None
        }
        integer_prediction = idx_to_choice[prediction]
        integer_predictions.append(integer_prediction)

    assert len(predictions) % num_questions == 0, (
        f"The number of predictions ({len(predictions)}) is not a multiple of "
        f"{num_questions}, which is required for the European Values metric."
    )

    # When we are using the situational version of the dataset, there are 5 phrasings
    # for each question, so we need to aggregate the predictions by question, which we
    # do using majority voting.
    using_situational = len(predictions) == num_questions * num_phrasings_per_question
    if using_situational:
        # Reshape the predictions to a 2D array with `num_phrasings_per_question` rows
        # (one for each phrasing) and `num_questions` columns (one for each question).
        # The five phrasings for each question appear right after each other, e.g.,
        # (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, ...)
        # Shape: (num_questions, num_phrasings_per_question)
        arr = np.array(
            [
                integer_predictions[i : i + num_phrasings_per_question]
                for i in range(0, len(predictions), num_phrasings_per_question)
            ]
        )

        # Double check that we reshaped the predictions correctly
        for idx, pred in enumerate(predictions):
            assert arr[idx // 5, idx % 5] == pred, (
                f"Reshaped predictions do not match the original predictions at index "
                f"{idx}: {arr[idx // 5, idx % 5]} != {pred}."
            )

        # Use majority voting to get the final prediction for each question
        # Shape: (53,)
        arr = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=arr)

        # Convert the array to a list
        integer_predictions = arr.tolist()

    # Some of the questions are categorical and we're only interested in whether the
    # model chooses a specific choice or not. This mapping takes the question index
    # to the choice value that we're interested in.
    question_choices = {
        0: 1,
        1: 5,
        3: 3,
        6: 1,
        15: 4,
        20: 2,
        47: 8,
        48: 7,
        49: 4,
        51: 4,
        52: 4,
    }

    # Map the predictions to the choices we're interested in
    integer_predictions = list(integer_predictions)
    for question_idx, choice in question_choices.items():
        integer_predictions[question_idx] = (
            1 if integer_predictions[question_idx] == choice else 0
        )

    return integer_predictions


def european_values_scoring_function(
    pipeline: "Pipeline", predictions: c.Sequence[int]
) -> float:
    """Scoring function for the European Values metric."""
    normalised_predictions = pipeline[0].transform([predictions])
    log_likelihoods = pipeline[1].transform(normalised_predictions)[0]
    score = sigmoid(pipeline[2].alpha_ * (log_likelihoods - pipeline[2].center_))
    return score.item()


european_values_metric = PipelineMetric(
    name="european_values",
    pretty_name="European Values",
    pipeline_repo="EuroEval/european-values-pipeline",
    pipeline_scoring_function=european_values_scoring_function,
    preprocessing_fn=european_values_preprocessing_fn,
)
