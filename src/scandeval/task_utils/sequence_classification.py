"""Utility functions related to the sequence-classification supertask."""

import logging
import re
import typing as t

import evaluate
import Levenshtein
import numpy as np
from evaluate import EvaluationModule

from ..data_models import BenchmarkConfig, GenerativeModelOutput
from ..utils import raise_if_model_output_contains_nan_values

if t.TYPE_CHECKING:
    from ..data_models import DatasetConfig
    from ..types import Labels, Predictions


logger = logging.getLogger("scandeval")


def compute_metrics(
    model_outputs_and_labels: tuple["Predictions", "Labels"],
    id2label: dict[int, str],
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
) -> dict[str, float]:
    """Compute the metrics needed for evaluation.

    Args:
        model_outputs_and_labels:
            The first sequence contains the model outputs and the second sequence
            contains the true labels.
        id2label:
            Conversion of indices to labels.
        dataset_config:
            The configuration of the dataset.
        benchmark_config:
            The configuration of the benchmark.

    Returns:
        A dictionary with the names of the metrics as keys and the metric values as
        values.
    """
    model_outputs, labels = model_outputs_and_labels
    label2id = {label: idx for idx, label in id2label.items()}
    raise_if_model_output_contains_nan_values(model_output=model_outputs)

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

    Returns:
        The predicted labels.
    """
    if model_output.scores is not None:
        return get_closest_logprobs_labels(
            generation_logprobs=model_output.scores, dataset_config=dataset_config
        )
    else:
        return get_closest_word_edit_labels(
            generated_sequences=model_output.sequences, dataset_config=dataset_config
        )


def get_closest_logprobs_labels(
    generation_logprobs: list[list[list[tuple[str, float]]]],
    dataset_config: "DatasetConfig",
) -> list[str]:
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

    Returns:
        The predicted labels.

    Raises:
        InvalidBenchmark:
            If no candidate label can be found for any of the generated labels.
    """
    english_labels = list(dataset_config.id2label.values())
    english2local = dataset_config.prompt_label_mapping
    candidate_labels = [
        english2local[lbl].lower() for lbl in english_labels
    ] + english_labels

    output_labels: list[str] = list()
    given_warning: bool = False
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

            # We want to use the first generated label which starts with a candidate
            # label, as the output label
            output_label: str | None = None
            for generated_label in generated_labels:
                candidate_output_labels = [
                    candidate_label
                    for candidate_label in candidate_labels
                    if candidate_label.startswith(generated_label)
                ]
                if candidate_output_labels:
                    output_label = candidate_output_labels[0]
                    break

            if output_label is not None:
                output_label = english2local.get(output_label, output_label)
                output_labels.append(output_label)
                break
        else:
            if not given_warning:
                logger.debug(
                    "Could not find a candidate label for any of the generated "
                    f"labels in the sample {sample}. Using {candidate_labels[0]!r} "
                    "as the output label."
                )
                given_warning = True
            output_labels.append(candidate_labels[0])
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
