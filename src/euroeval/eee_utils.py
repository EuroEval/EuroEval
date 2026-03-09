"""Utility functions for the Every Eval Ever (EEE) output format.

This module provides standalone functions for serialising and deserialising
`BenchmarkResult` objects to and from the Every Eval Ever JSON schema v0.2.1
(https://github.com/evaleval/every_eval_ever/blob/main/eval.schema.json).
"""

import collections.abc as c
import datetime
import json
import math
import time
import typing as t

from .constants import EEE_SCHEMA_VERSION

if t.TYPE_CHECKING:
    from .data_models import BenchmarkResult


def parse_optional_bool(value: str) -> bool | None:
    """Parse a string-encoded optional boolean value.

    Args:
        value:
            The string to parse.  `"null"` maps to `None`; any other value is
            compared case-insensitively to `"true"`.

    Returns:
        `None` if value is `"null"`, otherwise a boolean.
    """
    if value == "null":
        return None
    return value.lower() == "true"


def parse_optional_str(value: str) -> str | None:
    """Parse a string-encoded optional string value.

    Args:
        value:
            The string to parse.  `"null"` maps to `None`.

    Returns:
        `None` if value is `"null"`, otherwise the original string.
    """
    return None if value == "null" else value


def benchmark_result_to_eee_dict(result: "BenchmarkResult") -> dict:
    """Convert a BenchmarkResult to the Every Eval Ever (EEE) format.

    Produces a dictionary conforming to the Every Eval Ever JSON schema v0.2.1
    (https://github.com/evaleval/every_eval_ever/blob/main/eval.schema.json).
    The resulting dict can be written directly to
    `euroeval_benchmark_results.jsonl` and later reconstructed without loss via
    `benchmark_result_from_eee_dict`.

    The mapping is as follows:

    * Top-level fields: `schema_version`, `evaluation_id`,
      `evaluation_timestamp`, `retrieved_timestamp`, `source_metadata`.
    * `model_info`: model `id`/`name` plus EuroEval-specific details
      (`num_model_parameters`, `max_sequence_length`, `vocabulary_size`,
      `merge`, `generative`, `generative_type`) in `additional_details`.
    * `eval_library`: `name="euroeval"`, library version, and evaluation
      context (languages, task, shot config, library versions, raw per-iteration
      scores) in `additional_details`.
    * `evaluation_results`: one entry per metric.  The 95 % confidence interval
      half-width stored in the `_se` keys is exposed as a `confidence_interval`
      with `confidence_level: 0.95`.  Speed metrics (`test_speed`,
      `test_speed_short`) do not include `score_type`, `min_score`, or
      `max_score` because tokens-per-second has no fixed upper bound.

    Args:
        result:
            The benchmark result to convert.

    Returns:
        A dictionary matching the EEE JSON schema v0.2.1.
    """
    current_time = time.time()
    retrieved_timestamp = str(int(current_time))
    evaluation_timestamp = datetime.datetime.fromtimestamp(
        current_time, tz=datetime.timezone.utc
    ).isoformat()
    evaluation_id = f"{result.dataset}/{result.model}/{retrieved_timestamp}"

    # Parse metric results from the results dict
    total_results: dict[str, float] = {}
    raw_results: list = []
    if isinstance(result.results, dict):
        total_val = result.results.get("total", {})
        if isinstance(total_val, dict):
            total_results = total_val
        raw_val = result.results.get("raw", [])
        if isinstance(raw_val, (list, tuple)):
            raw_results = list(raw_val)

    num_samples = len(raw_results)

    # Collect main metric names (exclude SE keys and num_failed_instances)
    metric_names = [
        k
        for k in total_results
        if not k.endswith("_se") and k != "num_failed_instances"
    ]

    num_failed = total_results.get("num_failed_instances", 0.0)

    evaluation_results = []
    for metric_name in metric_names:
        score = total_results[metric_name]
        # The _se values stored in total_dict are 95% CI half-widths
        # (1.96 * SE, scaled by postprocessing_fn). Use confidence_interval.
        ci_half_width = total_results.get(f"{metric_name}_se", float("nan"))

        score_details: dict = {
            "score": score,
            "details": {"num_failed_instances": str(num_failed)},
        }

        uncertainty: dict = {}
        if not math.isnan(ci_half_width):
            uncertainty["confidence_interval"] = {
                "lower": score - ci_half_width,
                "upper": score + ci_half_width,
                "confidence_level": 0.95,
                "method": "bootstrap",
            }
        if num_samples > 0:
            uncertainty["num_samples"] = num_samples
        if uncertainty:
            score_details["uncertainty"] = uncertainty

        # Only add score_type/min_score/max_score for non-speed metrics since
        # speed is measured in tokens/second with no upper bound
        is_speed_metric = "speed" in metric_name
        metric_config: dict = {"lower_is_better": False}
        if not is_speed_metric:
            metric_config["score_type"] = "continuous"
            metric_config["min_score"] = 0
            metric_config["max_score"] = 100

        evaluation_results.append(
            {
                "evaluation_name": metric_name,
                "source_data": {
                    "dataset_name": result.dataset,
                    "source_type": "hf_dataset",
                },
                "metric_config": metric_config,
                "score_details": score_details,
            }
        )

    # Determine inference engine info
    inference_engine: dict = {}
    if result.litellm_version:
        inference_engine = {"name": "litellm", "version": result.litellm_version}
    elif result.vllm_version:
        inference_engine = {"name": "vllm", "version": result.vllm_version}
    elif result.transformers_version:
        inference_engine = {
            "name": "transformers",
            "version": result.transformers_version,
        }

    model_info: dict = {
        "name": result.model,
        "id": result.model,
        "additional_details": {
            "num_model_parameters": str(result.num_model_parameters),
            "max_sequence_length": str(result.max_sequence_length),
            "vocabulary_size": str(result.vocabulary_size),
            "merge": str(result.merge).lower(),
            "generative": str(result.generative).lower(),
            "generative_type": (
                result.generative_type if result.generative_type is not None else "null"
            ),
        },
    }
    if inference_engine:
        model_info["inference_engine"] = inference_engine

    eval_lib_additional_details = {
        "dataset": result.dataset,
        "languages": json.dumps(list(result.languages), ensure_ascii=False),
        "task": result.task,
        "few_shot": (
            str(result.few_shot).lower() if result.few_shot is not None else "null"
        ),
        "validation_split": (
            str(result.validation_split).lower()
            if result.validation_split is not None
            else "null"
        ),
        "transformers_version": result.transformers_version or "null",
        "torch_version": result.torch_version or "null",
        "vllm_version": result.vllm_version or "null",
        "xgrammar_version": result.xgrammar_version or "null",
        "litellm_version": result.litellm_version or "null",
        "raw_results": json.dumps(raw_results, ensure_ascii=False),
    }

    return {
        "schema_version": EEE_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "evaluation_timestamp": evaluation_timestamp,
        "retrieved_timestamp": retrieved_timestamp,
        "source_metadata": {
            "source_name": "EuroEval",
            "source_type": "evaluation_run",
            "source_organization_name": "EuroEval",
            "source_organization_url": "https://euroeval.com",
            "evaluator_relationship": "third_party",
        },
        "model_info": model_info,
        "eval_library": {
            "name": "euroeval",
            "version": result.euroeval_version or "unknown",
            "additional_details": eval_lib_additional_details,
        },
        "evaluation_results": evaluation_results,
    }


def benchmark_result_from_eee_dict(config: dict) -> "BenchmarkResult":
    """Create a BenchmarkResult from an Every Eval Ever format dictionary.

    Reconstructs a full `BenchmarkResult` from a dictionary conforming to the
    Every Eval Ever (EEE) JSON schema v0.2.1.  This function is the inverse of
    `benchmark_result_to_eee_dict` and enables lossless round-trips.

    Args:
        config:
            A dictionary conforming to the EEE JSON schema v0.2.1, as produced
            by `benchmark_result_to_eee_dict`.

    Returns:
        The reconstructed benchmark result.
    """
    from .data_models import BenchmarkResult

    model_info = config.get("model_info", {})
    eval_library = config.get("eval_library", {})
    evaluation_results: c.Sequence[dict] = config.get("evaluation_results", [])

    model = model_info.get("id", "")
    model_additional = model_info.get("additional_details", {})
    eval_lib_additional = eval_library.get("additional_details", {})

    # Extract dataset name from evaluation_results or additional_details
    if evaluation_results:
        dataset = (
            evaluation_results[0].get("source_data", {}).get("dataset_name", "")
        )
    else:
        dataset = eval_lib_additional.get("dataset", "")

    # Reconstruct raw results from stored JSON
    raw_results_json = eval_lib_additional.get("raw_results", "[]")
    try:
        raw_results = json.loads(raw_results_json)
    except json.JSONDecodeError:
        raw_results = []

    # Reconstruct total dict from evaluation_results
    total_dict: dict[str, float] = {}
    for eval_result in evaluation_results:
        metric_name = eval_result.get("evaluation_name", "")
        score_details = eval_result.get("score_details", {})
        total_dict[metric_name] = score_details.get("score", 0.0)

        uncertainty = score_details.get("uncertainty", {})
        ci_info = uncertainty.get("confidence_interval", {})
        if "lower" in ci_info and "upper" in ci_info:
            ci_half_width = (ci_info["upper"] - ci_info["lower"]) / 2
            total_dict[f"{metric_name}_se"] = ci_half_width

        details = score_details.get("details", {})
        if (
            "num_failed_instances" in details
            and "num_failed_instances" not in total_dict
        ):
            total_dict["num_failed_instances"] = float(
                details["num_failed_instances"]
            )

    results = {"raw": raw_results, "total": total_dict}

    languages_json = eval_lib_additional.get("languages", "[]")
    try:
        languages: list[str] = json.loads(languages_json)
    except json.JSONDecodeError:
        languages = []

    return BenchmarkResult(
        dataset=dataset,
        task=eval_lib_additional.get("task", ""),
        languages=languages,
        model=model,
        results=results,
        num_model_parameters=int(
            model_additional.get("num_model_parameters", "0") or "0"
        ),
        max_sequence_length=int(
            model_additional.get("max_sequence_length", "0") or "0"
        ),
        vocabulary_size=int(model_additional.get("vocabulary_size", "0") or "0"),
        merge=model_additional.get("merge", "false") == "true",
        generative=model_additional.get("generative", "false") == "true",
        generative_type=parse_optional_str(
            model_additional.get("generative_type", "null")
        ),
        few_shot=parse_optional_bool(eval_lib_additional.get("few_shot", "null")),
        validation_split=parse_optional_bool(
            eval_lib_additional.get("validation_split", "null")
        ),
        euroeval_version=parse_optional_str(
            eval_library.get("version", "null")
            if eval_library.get("version") != "unknown"
            else "null"
        ),
        transformers_version=parse_optional_str(
            eval_lib_additional.get("transformers_version", "null")
        ),
        torch_version=parse_optional_str(
            eval_lib_additional.get("torch_version", "null")
        ),
        vllm_version=parse_optional_str(
            eval_lib_additional.get("vllm_version", "null")
        ),
        xgrammar_version=parse_optional_str(
            eval_lib_additional.get("xgrammar_version", "null")
        ),
        litellm_version=parse_optional_str(
            eval_lib_additional.get("litellm_version", "null")
        ),
    )
