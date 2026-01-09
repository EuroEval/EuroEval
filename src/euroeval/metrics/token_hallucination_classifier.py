"""Inference speed metric."""

import collections.abc as c
import logging
import typing as t
from collections import defaultdict

from datasets import Dataset

from .base import Metric

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig


def detect_hallucinations(
    dataset: Dataset,
    predictions: c.Sequence,
    model: str = "KRLabsOrg/tinylettuce-ettin-17m-en",
) -> dict[str, list]:
    """Load tinylettuce model and detect hallucinations.

    Args:
        dataset: Hallucination dataset, generated with e.g. lettuce.
        model: Path to model.

    Returns:
        A dictionary with the predicted answers and ground truth hallucinated parts.
    """
    # detector = HallucinationDetector(
    #     method="transformer", model_path=model, device_map="auto", torch_dtype="auto"
    # )

    predict_answers = []
    all_hallucinated_parts = []
    for context, question, answer in zip(
        dataset["context"], dataset["question"], predictions
    ):
        # Use the detector to predict if the answer is hallucinated
        try:
            answer["prediction_text"]

            # predict_answer = detector.predict(
            #     context=context, question=question, answer=answer
            # )
            print(question)
            predict_answer = []
        except Exception as e:
            logger.error(f"Error during hallucination detection: {e}. Skipping...")
            continue
        predict_answers.append(predict_answer)

    if "hallucinated_parts" in dataset.column_names:
        for hallucinated_part in dataset["hallucinated_parts"]:
            all_hallucinated_parts.append(hallucinated_part)

    data_dict: dict[str, list] = defaultdict(list)
    data_dict["predict_answers"] = predict_answers
    data_dict["ground_truth"] = all_hallucinated_parts

    return data_dict


class Token_Hallucination_Metric(Metric):
    """Hallucination metric."""

    def __init__(self, name: str, pretty_name: str) -> None:
        """Initialise the token hallucination metric.

        Args:
            name:
                The name of the metric in snake_case.
            pretty_name:
                The pretty name of the metric, used for display purposes.
        """
        super().__init__(
            name=name,
            pretty_name=pretty_name,
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:,.0f}"),
        )

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Not used with the hallucination metric, but required for consistency."""
        detect_hallucinations(
            dataset=dataset,
            predictions=predictions,
            model="alexandrainst/mmBERT-small-multi-wiki-qa-synthetic-hallucinations-da",
        )
        raise NotImplementedError


hallucination_metric = Token_Hallucination_Metric(
    name="hallucination_token", pretty_name="Tokens per second"
)
