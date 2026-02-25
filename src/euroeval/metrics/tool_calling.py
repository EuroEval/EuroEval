"""Tool calling metric."""

import collections.abc as c
import json
import typing as t

from euroeval.logging_utils import log_once

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig

from ..metrics.base import Metric


class ToolCallingAccuracy(Metric):
    """Metric for tool calling."""

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate tool calling accuracy.

        Args:
            predictions:
                Predicted "labels" - meaning tool calls in this context.
            references:
                Ground truth data - NB: format is different from predictions,
                since ground truth contains lists of possible outputs rather than
                a single 'truth'.
            dataset:
                Dataset - used for tool information like required arguments.
            dataset_config:
                Part of interface - not used here.
            benchmark_config:
                Part of interface - not used here.

        Returns:
            The score (accuracy).
            Returns None if any of predictions, references or dataset["function"]
            sequences are empty - meaning a score could not be calculated.
        """
        function_descriptions = [json.loads(f) for f in dataset["function"]]
        results = []
        for x in zip(predictions, references, function_descriptions):
            results.append(_evaluate_function_toolcall_response(*x))
        if not results:
            return None
        else:
            return sum(results) / len(results)


tool_calling_metric = ToolCallingAccuracy(
    name="tool_calling_accuracy", pretty_name="Tool Calling Accuracy"
)


def _evaluate_function_toolcall_response(
    pred_calls_str: str, ref_calls_str: str, descriptions: list[dict]
) -> bool:
    """Logic to evaluate tool call response against reference (ground truth).

    Args:
        pred_calls_str:
            Predicted function calls as json string
        ref_calls_str:
            Referenced function calls as json string
        descriptions:
            Function descriptions (in dataset and given as input to models)

    Returns:
        True: success, False: failure
    """
    # try deserialize prediction
    try:
        pred_calls_dict = json.loads(pred_calls_str)
        assert isinstance(pred_calls_dict, dict)
        assert "tool_calls" in pred_calls_dict
        pred_calls = pred_calls_dict["tool_calls"]
    except (json.JSONDecodeError, AssertionError):
        return False

    ref_calls = json.loads(ref_calls_str)

    # number of predicted function calls should equal the reference
    if len(pred_calls) != len(ref_calls):
        return False

    for pred_call, ref_call, description in zip(pred_calls, ref_calls, descriptions):
        # each predicted function call should be a dict
        if not isinstance(pred_call, dict):
            return False

        # get predicted function name
        if "function" not in pred_call:
            log_once(
                "Tool call prediction did not contain required keyword 'function'."
            )
            return False
        else:
            pred_name: str = pred_call["function"]

        # get predicted arguments
        if "arguments" not in pred_call:
            log_once(
                "Tool call prediction did not contain required keyword 'arguments'."
            )
            return False
        else:
            pred_args: dict = pred_call["arguments"]

        ref_name: str
        ref_args: dict
        # reference calls are packed into an extra list by BFCL default for some reason
        ref_name, ref_args = list(ref_call.items())[0]

        # did we predict the right function to call?
        if pred_name != ref_name:
            return False

        # get requires arguments from function descriptions
        parameters = description.get("parameters", None)
        required_args = (
            parameters.get("required", None) if isinstance(parameters, dict) else None
        )

        for key, values in ref_args.items():
            # we only care about required arguments
            if required_args and key not in required_args:
                continue
            # every predicted argument should be in the list of expected values
            if key not in pred_args or pred_args[key] not in values:
                return False
    return True
