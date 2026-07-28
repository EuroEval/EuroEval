"""IFEval instruction-following metric."""

import collections.abc as c
import logging
import typing as t
from pathlib import Path

import nltk

from ...logging_utils import log_once
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
        self._nltk_data_dir: Path | None = None
        super().__init__(
            name="instruction_accuracy",
            pretty_name="Instruction Accuracy",
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
            self._setup_nltk(Path(benchmark_config.cache_dir))
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
                    log_once(
                        f"Skipping unsupported instruction: {instruction_id}",
                        level=logging.WARNING,
                    )
                    continue

                constraint_function = ALL_CONSTRAINTS[instruction_id]
                is_following = constraint_function(response, **kwargs)
                results.append(is_following)

            all_results.extend(results)
        return sum(all_results) / len(all_results) if all_results else 0.0


    def _setup_nltk(self, cache_dir: Path) -> None:
        """Set up NLTK to use the cache directory and suppress logging.

        Args:
            cache_dir:
                The cache directory to use for NLTK data.
        """
        self._nltk_data_dir = cache_dir / "nltk_data"
        self._nltk_data_dir.mkdir(parents=True, exist_ok=True)

        # Set the NLTK search path to include only our cache directory
        # This ensures NLTK data is stored inside .euroeval_cache
        nltk.data.path.insert(0, str(self._nltk_data_dir))

        # Suppress NLTK download logging using no_terminal_output
        # This captures both stdout and stderr at the OS level
        # NLTK prints to both streams even with quiet=True
        from ...logging_utils import no_terminal_output

        with no_terminal_output():
            # Download required NLTK packages
            nltk.download("punkt_tab", download_dir=str(self._nltk_data_dir), quiet=True)
            nltk.download("wordnet", download_dir=str(self._nltk_data_dir), quiet=True)
            nltk.download("omw-1.4", download_dir=str(self._nltk_data_dir), quiet=True)


instruction_accuracy = IFEvalInstructionAccuracy()
