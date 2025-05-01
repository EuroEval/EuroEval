"""Utility functions related to the sequence-classification task group."""

import logging
import re
import typing as t

import evaluate
import Levenshtein
import numpy as np
from evaluate import EvaluationModule

from ..data_models import BenchmarkConfig, GenerativeModelOutput
from ..exceptions import InvalidBenchmark
from ..utils import log_once, raise_if_model_output_contains_nan_values

if t.TYPE_CHECKING:
    from transformers.trainer_utils import EvalPrediction

    from ..data_models import DatasetConfig
    from ..types import Labels, Predictions


logger = logging.getLogger("euroeval")


def compute_metrics(
    model_outputs_and_labels: "tuple[Predictions, Labels] | EvalPrediction",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
) -> dict[str, float]:
    """Compute the metrics needed for evaluation.

    Args:
        model_outputs_and_labels:
            The first sequence contains the model outputs and the second sequence
            contains the true labels.
        dataset_config:
            The configuration of the dataset.
        benchmark_config:
            The configuration of the benchmark.

    Returns:
        A dictionary with the names of the metrics as keys and the metric values as
        values.
    """
    model_outputs, labels = model_outputs_and_labels
    label2id = {label: idx for idx, label in dataset_config.id2label.items()}

    # If the model outputs is a pair, then the first element corresponds to the model
    # predictions
    if isinstance(model_outputs, tuple) and len(model_outputs) == 2:
        model_outputs = model_outputs[0]

    metrics = {
        metric_cfg.name: (
            evaluate.load(
                path=metric_cfg.huggingface_id, cache_dir=benchmark_config.cache_dir
            )
            if metric_cfg.huggingface_id != ""
            else None
        )
        for metric_cfg in dataset_config.task.metrics
    }

    model_output_dtype = np.asarray(model_outputs).dtype
    if model_output_dtype in [np.float16, np.float32, np.float64]:
        predictions = np.asarray(model_outputs).argmax(axis=-1)
    else:
        predictions = model_outputs

    assert not isinstance(model_outputs, tuple)
    raise_if_model_output_contains_nan_values(model_output=model_outputs)

    prompt_label_to_label_mapping = {
        prompt_label: label
        for label, prompt_label in dataset_config.prompt_label_mapping.items()
    }
    predictions = [
        (
            label2id[prompt_label_to_label_mapping[pred.lower()]]
            if isinstance(pred, str)
            else pred
        )
        for pred in predictions
    ]

    label_ids = [
        label2id[label.lower()] if isinstance(label, str) else label for label in labels
    ]

    results: dict[str, float] = dict()
    for cfg in dataset_config.task.metrics:
        metric = metrics[cfg.name]
        assert isinstance(metric, EvaluationModule)
        score_dict: dict[str, float] | None = metric.compute(
            predictions=predictions, references=label_ids, **cfg.compute_kwargs
        )

        # The metric returns None if we are running on multi-GPU and the current
        # process is not the main process
        if score_dict is not None:
            scores = score_dict[cfg.results_key]
            if isinstance(scores, list):
                scores = sum(scores) / len(scores)
            results[cfg.name] = scores

    return results


def extract_labels_from_generation(
    input_batch: dict[str, list],
    model_output: GenerativeModelOutput,
    dataset_config: "DatasetConfig",
    first_label_token_mapping: dict[str, str] | bool,
) -> list[str]:
    """Extract the predicted labels from the generated output.

    Args:
        input_batch:
            The input batch, where the keys are the feature names and the values
            are lists with the feature values.
        model_output:
            The raw generated output of the model.
        dataset_config:
            The configuration of the dataset.
        first_label_token_mapping:
            A mapping from labels to the first token in each label, or alternatively a
            Boolean value indicating whether the model should output scores (if the
            mapping is outputted then the model will always output scores).

    Returns:
        The predicted labels.
    """
    if model_output.scores is not None:
        if first_label_token_mapping is False:
            raise InvalidBenchmark(
                "The model outputted logprobs, but the first label token mapping is "
                "not provided. This means that the model should not output logprobs."
            )
        labels = get_closest_logprobs_labels(
            generation_logprobs=model_output.scores,
            dataset_config=dataset_config,
            first_label_token_mapping=first_label_token_mapping,
        )
        if labels is not None:
            return labels
    return get_closest_word_edit_labels(
        generated_sequences=model_output.sequences, dataset_config=dataset_config
    )


def get_closest_logprobs_labels(
    generation_logprobs: list[list[list[tuple[str, float]]]],
    dataset_config: "DatasetConfig",
    first_label_token_mapping: dict[str, str] | t.Literal[True],
) -> list[str] | None:
    """Get the labels with the highest predicted logprob value.

    In case a candidate label is split into multiple tokens, we only use the first
    token to compute the logprob value. E.g., if the candidate label "positive" is
    tokenised as ["pos", "itive"], we only use the logprob value of "pos" to
    represent the logprob value of the entire label.

    Args:
        generation_logprobs:
            The logprobs of the generated tokens, for all samples in the batch. Of shape
            (batch_size, num_tokens, num_logprobs).
        dataset_config:
            The configuration of the dataset.
        first_label_token_mapping:
            A mapping from labels to the first token in each label, or alternatively a
            `True` value indicating that the model should output logprobs.

    Returns:
        The predicted labels, or None if labels could not be extracted.

    Raises:
        InvalidBenchmark:
            If no candidate label can be found for any of the generated labels.
    """
    english_labels = list(dataset_config.id2label.values())
    english2local = dataset_config.prompt_label_mapping
    candidate_labels = [english2local[lbl].lower() for lbl in english_labels]

    output_labels: list[str] = list()
    for sample in generation_logprobs:
        for logprob_list in sample:
            generated_labels = [
                re.sub(
                    pattern=r"^[^a-zæøåüöä]+|[^a-zæøåüöä]+$",
                    repl="",
                    string=label.lower(),
                )
                for label, _ in logprob_list
            ]
            generated_labels = [label for label in generated_labels if label != ""]

            # We want to use the first generated label which contains a unique candidate
            # label, as the output label
            output_label: str | None = None
            for generated_label in generated_labels:
                # Get the candidate labels. If we have a first label token mapping, we
                # use it to get the candidate labels. Otherwise, we check if any of the
                # labels start with the generated label.
                if isinstance(first_label_token_mapping, dict):
                    if any(
                        candidate_label not in first_label_token_mapping
                        for candidate_label in candidate_labels
                    ):
                        raise InvalidBenchmark(
                            "There is a label not present in the first label token "
                            "mapping - this should never happen! Please report this "
                            "issue to the EuroEval team at "
                            "github.com/EuroEval/EuroEval/issues."
                        )

                    candidate_output_labels = {
                        candidate_label
                        for candidate_label in candidate_labels
                        if generated_label == first_label_token_mapping[candidate_label]
                    }
                else:
                    candidate_output_labels = {
                        candidate_label
                        for candidate_label in candidate_labels
                        if candidate_label.startswith(generated_label)
                    }

                # If we can uniquely determine the output label, we break the loop.
                if len(candidate_output_labels) == 1:
                    output_label = candidate_output_labels.pop()
                    break

                # If we have multiple candidate labels, we cannot uniquely determine the
                # output label, so we abandon extracting the labels using logprobs and
                # fall back to using word edit distance.
                elif len(candidate_output_labels) > 1:
                    log_once(
                        "Multiple candidate labels found for the generated label "
                        f"{generated_label!r}: {candidate_output_labels}. This means "
                        "that using logprobs to extract the labels is not reliable, "
                        "and we will instead fall back to extracting the labels "
                        "using word edit distance.",
                        level=logging.DEBUG,
                    )
                    return None

                # If no candidate label is found, we first check if any of the labels
                # start with the generated label. This could be the case if the labels
                # in the first token mapping is inaccurate or incomplete, for instance
                # if 'pos' is in the first label token mapping, but the model outputted
                # 'posit'. If this is the case then we cannot trust the first label
                # token mapping, and we fall back to using word edit distance.
                # Otherwise, the generated label is just bad, and we skip to the next
                # generated label.
                elif len(candidate_output_labels) == 0:
                    candidate_output_labels_starting_with_generated_label = [
                        candidate_label
                        for candidate_label in candidate_labels
                        if candidate_label.startswith(generated_label)
                    ]
                    if candidate_output_labels_starting_with_generated_label:
                        log_once(
                            f"No candidate label found for the generated label "
                            f"{generated_label!r}. This means that using logprobs to "
                            "extract the labels is not reliable, and we will instead "
                            "fall back to extracting the labels using word edit "
                            "distance.",
                            level=logging.DEBUG,
                        )
                        return None

            # If we did not find any candidate label for any of the generated labels, we
            # assume that something is wrong with the model output, and we fall back to
            # using word edit distance to extract the labels
            else:
                log_once(
                    f"No candidate label found for any of the generated labels "
                    f"{generated_labels}. This means that using logprobs to extract "
                    "the labels is not reliable, and we will instead fall back to "
                    "extracting the labels using word edit distance.",
                    level=logging.DEBUG,
                )
                return None

            if output_label is not None:
                output_labels.append(output_label)
                break
        else:
            if len(sample) == 0:
                log_once(
                    "The model outputted an empty string, so no candidate labels could "
                    f"be determined. Using {candidate_labels[0]!r} as the output "
                    "label.",
                    level=logging.DEBUG,
                )
            else:
                log_once(
                    "Could not find a candidate label for any of the generated "
                    f"labels in the sample {sample}. Using {candidate_labels[0]!r} "
                    "as the output label.",
                    level=logging.DEBUG,
                )
            output_labels.append(candidate_labels[0])

    assert len(output_labels) == len(generation_logprobs)
    return output_labels


def get_closest_word_edit_labels(
    generated_sequences: list[str], dataset_config: "DatasetConfig"
) -> list[str]:
    """Get the labels with the smallest edit distance to the predicted labels.

    Args:
        generated_sequences:
            The generated sequences from the model.
        dataset_config:
            The configuration of the dataset.

    Returns:
        The candidate labels with the smallest edit distance to the predicted labels.
    """
    candidate_labels = [
        dataset_config.prompt_label_mapping[lbl]
        for lbl in dataset_config.id2label.values()
    ]
    new_predicted_labels: list[str] = list()
    for predicted_label in generated_sequences:
        edit_distances = [
            Levenshtein.distance(s1=predicted_label.lower(), s2=candidate_label.lower())
            for candidate_label in candidate_labels
        ]
        closest_label = candidate_labels[np.argmin(edit_distances).item()]
        new_predicted_labels.append(closest_label)
    return new_predicted_labels
