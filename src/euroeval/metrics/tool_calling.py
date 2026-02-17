"""Tool calling metric."""

import collections.abc as c
import json
import typing as t

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from euroeval.data_models import BenchmarkConfig, DatasetConfig

from euroeval.metrics.base import Metric


def _evaluate_function_toolcall_response(
    pred_calls_str: str, ref_calls_str: str, descriptions: list[dict]
) -> bool:
    try:
        pred_calls = json.loads(pred_calls_str)
        assert isinstance(pred_calls, list)
    except json.JSONDecodeError:
        return False
    ref_calls = json.loads(ref_calls_str)
    if len(pred_calls) != len(ref_calls):
        return False
    for pred_call, ref_call, description in zip(pred_calls, ref_calls, descriptions):
        if not isinstance(pred_call, dict):
            return False
        pred_name = pred_call.get("function", None)
        pred_args = pred_call.get("arguments", {})
        if not pred_name:
            return False
        if len(ref_call) != 1:
            return False
        ref_name: str
        ref_args: dict
        ref_name, ref_args = list(ref_call.items())[0]
        if pred_call.get("function", None) != ref_name:
            return False
        parameters = description.get("parameters", None)
        required_args = (
            parameters.get("required", None) if isinstance(parameters, dict) else None
        )
        for key, values in ref_args.items():
            if required_args and key not in required_args:
                continue
            if key not in pred_args or pred_args[key] not in values:
                return False
    return True


class ToolCallingMetric(Metric):
    """Metric for tool calling."""

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate tool calling metric.

        Returns:
            The Score.
        """
        function_descriptions = [json.loads(f) for f in dataset["function"]]
        results = []
        for x in zip(predictions, references, function_descriptions):
            results.append(_evaluate_function_toolcall_response(*x))
        if not results:
            return None
        else:
            return sum(results) / len(results)


tool_calling_metric = ToolCallingMetric(
    name="tool_calling_accuracy", pretty_name="Accuracy (Tool Calling)"
)
