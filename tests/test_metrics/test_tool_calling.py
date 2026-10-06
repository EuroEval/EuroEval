"""Public-interface tests for tool-calling accuracy."""

import json

from datasets import Dataset

from euroeval.data_models import BenchmarkConfig, DatasetConfig
from euroeval.metrics.tool_calling import tool_calling_accuracy


def test_tool_calling_accuracy_uses_called_function_schema(
    benchmark_config: BenchmarkConfig, dataset_config: DatasetConfig
) -> None:
    """Required arguments come from the schema matching the called function."""
    dataset = Dataset.from_list(
        [
            {
                "function": json.dumps(
                    [
                        {"name": "decoy", "parameters": {"required": ["other"]}},
                        {"name": "target", "parameters": {"required": ["arg"]}},
                    ]
                )
            }
        ]
    )

    score = tool_calling_accuracy(
        predictions=[
            json.dumps(
                {"tool_calls": [{"function": "target", "arguments": {"arg": "wrong"}}]}
            )
        ],
        references=[json.dumps([{"target": {"arg": ["expected"]}}])],
        dataset=dataset,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
    )

    assert score == 0.0
