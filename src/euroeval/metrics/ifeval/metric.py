"""IFEval instruction-following metric."""

import collections.abc as c
import logging
import typing as t

import nltk

from ..base import Metric
from .constraints import ALL_CONSTRAINTS

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig

logger = logging.getLogger(__name__)


class IFEvalInstructionAccuracy(Metric):
    """Metric for instruction-level accuracy using IFEval methodology."""

    def __init__(self) -> None:
        """Initialise the metric."""
        self.downloaded_nltk = False
        super().__init__(
            name="inst_level_strict_acc",
            pretty_name="Instruction-Level Strict Accuracy",
            postprocessing_fn=None,
        )

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate instruction-level accuracy.

        Args:
            predictions:
                The model's predictions.
            references:
                The reference data.
            dataset:
                The dataset.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.

        Returns:
            The instruction-level accuracy.
        """
        if not self.downloaded_nltk:
            nltk.download("punkt_tab", quiet=True)
            self.downloaded_nltk = True

        all_results: list[bool] = []
        for pred, ref in zip(predictions, references):
            response = str(pred)

            if not response.strip():
                results = [False] * len(
                    [
                        instruction_id
                        for instruction_id in ref["instruction_id_list"]
                        if instruction_id in ALL_CONSTRAINTS
                    ]
                )
                all_results.extend(results)
                continue

            results: list[bool] = list()
            for instruction_id, kwargs in zip(
                ref["instruction_id_list"], ref["kwargs"]
            ):
                if instruction_id not in ALL_CONSTRAINTS:
                    logger.debug(f"Skipping unsupported instruction: {instruction_id}")
                    continue

                constraint_function = ALL_CONSTRAINTS[instruction_id]
                is_following = constraint_function(response, **kwargs)
                results.append(is_following)

            all_results.extend(results)
        return sum(all_results) / len(all_results) if all_results else 0.0


instruction_accuracy = IFEvalInstructionAccuracy()
