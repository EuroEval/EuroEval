"""Utility functions related to the sequence-classification task group."""

import collections.abc as c
import logging
import re
import typing as t

import Levenshtein
import numpy as np

from ..enums import TaskGroup
from ..exceptions import InvalidBenchmark
from ..utils import (
    extract_multiple_choice_labels,
    log_once,
    raise_if_model_output_contains_nan_values,
)

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from transformers.trainer_utils import EvalPrediction

    from ..data_models import (
        BenchmarkConfig,
        DatasetConfig,
        GenerativeModelOutput,
        ModelConfig,
    )
    from ..types import Labels, Predictions


def compute_metrics(
    model_outputs_and_labels: "tuple[Predictions, Labels] | EvalPrediction",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
    dataset: "Dataset",
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
        dataset:
            The dataset used for evaluation. This is only used in case any additional
            metadata is used to compute the metrics.

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
    for metric in dataset_config.task.metrics:
        score: float | None = metric(
            predictions=predictions,
            references=label_ids,
            dataset=dataset,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )

        # The metric returns None if we are running on multi-GPU and the current
        # process is not the main process
        if score is not None:
            results[metric.name] = score

    return results


def extract_labels_from_generation(
    input_batch: dict[str, list],
    model_output: "GenerativeModelOutput",
    dataset_config: "DatasetConfig",
    model_config: "ModelConfig",
    first_label_token_mapping: dict[str, str] | bool,
) -> c.Sequence[str]:
    """Extract the predicted labels from the generated output.

    Args:
        input_batch:
            The input batch, where the keys are the feature names and the values
            are lists with the feature values.
        model_output:
            The raw generated output of the model.
        dataset_config:
            The configuration of the dataset.
        model_config:
            The configuration of the model.
        first_label_token_mapping:
            A mapping from labels to the first token in each label, or alternatively a
            Boolean value indicating whether the model should output scores (if the
            mapping is outputted then the model will always output scores).

    Returns:
        The predicted labels.

    Raises:
        InvalidBenchmark:
            If the task requires log probabilities, but the model did not output them,
            or if the model outputted log probabilities but the first label token
            mapping is not provided.
    """
    # Get the candidate labels, which are the labels that the model can predict
    default_labels = [
        dataset_config.prompt_label_mapping[lbl]
        for lbl in dataset_config.id2label.values()
    ]
    if dataset_config.task.task_group == TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION:
        sample_candidate_labels = [
            extract_multiple_choice_labels(
                prompt=prompt, candidate_labels=default_labels
            )
            for prompt in input_batch["prompt"]
        ]
    else:
        sample_candidate_labels = [default_labels] * len(input_batch["prompt"])

    if model_output.scores is not None:
        if first_label_token_mapping is False:
            raise InvalidBenchmark(
                "The model outputted logprobs, but the first label token mapping is "
                "not provided, which is not supported."
            )
        labels = get_closest_logprobs_labels(
            generation_logprobs=model_output.scores,
            first_label_token_mapping=first_label_token_mapping,
            candidate_labels=sample_candidate_labels,
        )
        if labels is not None:
            return labels
        elif dataset_config.task.requires_logprobs:
            raise InvalidBenchmark(
                "This task requires the model to output logprobs, and this model "
                "does not seem to be able to do that. Skipping the evaluation."
            )

    new_predicted_labels: list[str] = list()
    num_predictions_being_very_off = 0
    for idx, predicted_label in enumerate(model_output.sequences):
        # If the prediction includes a boxed answer, use that instead of the full
        # generation
        if (m := re.search(r"boxed\{(.*?)\}", predicted_label)) is not None:
            predicted_label = m.group(1)

        # We set the word edit distance weights such that we heavily penalise insertions
        # and substitutions, so that we don't just insert the correct label, but that we
        # want the model to have included the correct label in its output.
        insertion_weight = 1000
        deletion_weight = 1
        substitution_weight = 1000

        # Compute the word edit distances between the predicted label and all candidate
        # labels
        edit_distances = [
            Levenshtein.distance(
                s1=predicted_label.lower(),
                s2=candidate_label.lower(),
                weights=(insertion_weight, deletion_weight, substitution_weight),
            )
            for candidate_label in sample_candidate_labels[idx]
        ]

        best_candidate_label = sample_candidate_labels[idx][
            np.argmin(edit_distances).item()
        ]

        # If no candidate labels were found, we either pick the label with the smallest
        # word edit distance to the predicted label (if invalid model outputs are
        # allowed), or we raise an error
        if min(edit_distances) >= 1000:
            num_predictions_being_very_off += 1

        new_predicted_labels.append(best_candidate_label)

    if num_predictions_being_very_off > 0:
        if dataset_config.allow_invalid_model_outputs:
            log_msg = (
                "No candidate labels found for the predicted label in "
                f"{num_predictions_being_very_off:,}/{len(model_output.sequences):,} "
                f"of the samples with the model {model_config.model_id!r}. This "
                "likely means that the model were completely off in these cases, "
                "but since invalid model outputs are allowed for this task, we used "
                "the closest candidate labels as the output labels."
            )
            level = logging.DEBUG
            if num_predictions_being_very_off / len(model_output.sequences) > 0.5:
                log_msg += (
                    " Since this happened for most of the model's predictions, please "
                    "report this issue to the EuroEval team at "
                    "github.com/EuroEval/EuroEval/issues."
                )
                level = logging.WARNING
            log_once(log_msg, level=level)
        else:
            raise InvalidBenchmark(
                "No candidate labels found for the predicted label in "
                f"{num_predictions_being_very_off:,}/{len(model_output.sequences):,} "
                "of the samples. This likely means that the model were completely "
                "off in these cases. Since this task does not allow invalid model "
                "outputs, we have to abort the evaluation. Please re-run the "
                "evaluation with the `--debug` flag (or `debug=True` if you're using "
                "the `Benchmarker` API) to see the precise model outputs."
            )

    return new_predicted_labels


def get_closest_logprobs_labels(
    generation_logprobs: c.Sequence[c.Sequence[c.Sequence[tuple[str, float]]]],
    first_label_token_mapping: dict[str, str] | t.Literal[True],
    candidate_labels: c.Sequence[c.Sequence[str]],
) -> c.Sequence[str] | None:
    """Get the labels with the highest predicted logprob value.

    In case a candidate label is split into multiple tokens, we only use the first
    token to compute the logprob value. E.g., if the candidate label "positive" is
    tokenised as ["pos", "itive"], we only use the logprob value of "pos" to
    represent the logprob value of the entire label.

    Args:
        generation_logprobs:
            The logprobs of the generated tokens, for all samples in the batch. Of shape
            (batch_size, num_tokens, num_logprobs).
        first_label_token_mapping:
            A mapping from labels to the first token in each label, or alternatively a
            `True` value indicating that the model should output logprobs.
        candidate_labels:
            The candidate labels for each sample in the batch.

    Returns:
        The predicted labels, or None if labels could not be extracted.

    Raises:
        InvalidBenchmark:
            If no candidate label can be found for any of the generated labels.
    """
    output_labels: list[str] = list()
    for idx, sample in enumerate(generation_logprobs):
        for logprob_list in sample:
            generated_labels = [
                re.sub(pattern=r"^[^a-zæøåüöä0-9]+$", repl="", string=label.lower())
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
                        for candidate_label in candidate_labels[idx]
                    ):
                        raise InvalidBenchmark(
                            "There is a label not present in the first label token "
                            "mapping - this should never happen! Please report this "
                            "issue to the EuroEval team at "
                            "github.com/EuroEval/EuroEval/issues."
                        )

                    candidate_output_labels = {
                        candidate_label
                        for candidate_label in candidate_labels[idx]
                        if generated_label == first_label_token_mapping[candidate_label]
                    }
                else:
                    candidate_output_labels = {
                        candidate_label
                        for candidate_label in candidate_labels[idx]
                        if candidate_label.startswith(generated_label.strip())
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
                        for candidate_label in candidate_labels[idx]
                        if candidate_label.startswith(generated_label)
                    ]
                    if candidate_output_labels_starting_with_generated_label:
                        log_once(
                            f"No candidate label found for the generated label "
                            f"{generated_label!r}, but there are candidate labels "
                            f"starting with it: "
                            f"{candidate_output_labels_starting_with_generated_label}. "
                            "This means that the first label token mapping is not "
                            "reliable, and we will instead fall back to extracting "
                            "the labels using word edit distance.",
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
                    "be determined. This means that using logprobs to extract the "
                    "labels is not reliable, and we will instead fall back to "
                    "extracting the labels using word edit distance.",
                    level=logging.DEBUG,
                )
            else:
                log_once(
                    "No candidate label found for any of the generated labels, which "
                    "means that using logprobs to extract the labels is not reliable, "
                    "and we will instead fall back to extracting the labels using "
                    "word edit distance.",
                    level=logging.DEBUG,
                )
            return None

    assert len(output_labels) == len(generation_logprobs)
    return output_labels
