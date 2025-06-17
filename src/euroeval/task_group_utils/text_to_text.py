"""Utility functions related to the text-to-text task group."""

import logging
import typing as t

import evaluate
import numpy as np
from evaluate import EvaluationModule
from lingua import IsoCode639_1, LanguageDetectorBuilder

from ..benchmark_config_factory import get_correct_language_codes
from ..constants import METRIC_ATTRIBUTES_TAKING_UP_MEMORY, MIN_CONFIDENCE_SCORE
from ..data_models import BenchmarkConfig, DatasetConfig, GenerativeModelOutput
from ..exceptions import InvalidBenchmark
from ..utils import HiddenPrints, raise_if_model_output_contains_nan_values

if t.TYPE_CHECKING:
    from transformers.trainer_utils import EvalPrediction

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

    # If the model outputs is a pair, then the first element corresponds to the model
    # predictions
    if isinstance(model_outputs, tuple) and len(model_outputs) == 2:
        model_outputs = model_outputs[0]

    assert not isinstance(model_outputs, tuple)
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
    output_is_prob = model_output_dtype in [np.float16, np.float32, np.float64]
    if output_is_prob:
        predictions = np.asarray(model_outputs).argmax(axis=-1)
    else:
        predictions = model_outputs

    # Set up languages and language detector with lingua
    target_langs = get_correct_language_codes(
        language_codes=[lang.code for lang in dataset_config.languages]
    )
    lingua_langs = [IsoCode639_1.from_str(lang_code) for lang_code in target_langs]

    # We use the detector with all latin languages, since we do not want to be in
    # single language mode for datasets that only have a single language.
    # This is important for datasets that have 'multiple' languages, such as Norwegian.
    # See: https://github.com/pemistahl/lingua-py#116-single-language-mode
    detector = (
        LanguageDetectorBuilder.from_all_languages_with_latin_script()
        .with_preloaded_language_models()
        .build()
    )

    results: dict[str, float] = dict()
    for cfg in dataset_config.task.metrics:
        metric = metrics[cfg.name]
        assert isinstance(metric, EvaluationModule)

        # Some metrics can be computed on hardware accelerators. In this case we
        # start by setting the device to the same device as the model
        if cfg.compute_kwargs.get("device", None) == "auto":
            cfg.compute_kwargs["device"] = benchmark_config.device.type

        while True:
            try:
                with HiddenPrints():
                    score_dict: dict[str, float] | None = metric.compute(
                        predictions=predictions, references=labels, **cfg.compute_kwargs
                    )
                break
            except Exception as e:
                oom_error = [
                    "CUDA out of memory",
                    "CUDA error",
                    "MPS backend out of memory",
                ]
                if not any(error in str(e) for error in oom_error):
                    raise InvalidBenchmark(str(e))

                if cfg.compute_kwargs.get("device", "cpu") != "cpu":
                    cfg.compute_kwargs["device"] = "cpu"
                    logger.debug(
                        "Out of memory error occurred during the computation of "
                        f"the metric {cfg.pretty_name}. Moving the computation to "
                        "the CPU."
                    )
                else:
                    raise InvalidBenchmark(str(e))
            finally:
                for attribute in METRIC_ATTRIBUTES_TAKING_UP_MEMORY:
                    if hasattr(metric, attribute):
                        logger.debug(
                            f"Deleting the {attribute!r} attribute of the metric "
                            f"{cfg.pretty_name} to free up memory."
                        )
                        delattr(metric, attribute)

        # The metric returns None if we are running on multi-GPU and the current
        # process is not the main process
        if score_dict is not None:
            scores = score_dict[cfg.results_key]
            confidence_values = detector.compute_language_confidence_values_in_parallel(
                predictions
            )

            confidence_mask = []
            # collect mask for samples with correct language
            for confidences in confidence_values:
                sample_confidence = sum(
                    conf.value
                    for conf in confidences
                    if conf.language.iso_code_639_1 in lingua_langs
                )
                if sample_confidence >= MIN_CONFIDENCE_SCORE:
                    confidence_mask.append(1)
                else:
                    confidence_mask.append(0)

            # if scores is not an aggregate, then we multiply the sample scores with
            # the mask
            # otherwise we multiply the aggregate score with the proportion of samples
            # that are in the correct language
            if isinstance(scores, list):
                scores_np = np.array(scores)
                confidence_mask_np = np.array(confidence_mask)
                corrected_scores = scores_np * confidence_mask_np

                final_score = np.mean(corrected_scores).item()
            else:
                proportion_correct = np.mean(confidence_mask).item()
                corrected_score = scores * proportion_correct

                final_score = corrected_score

            results[cfg.name] = final_score

    return results


def extract_labels_from_generation(
    input_batch: dict[str, list], model_output: "GenerativeModelOutput"
) -> list[t.Any]:
    """Extract the predicted labels from the generated output.

    Args:
        input_batch:
            The input batch, where the keys are the feature names and the values
            are lists with the feature values.
        model_output:
            The raw generated output of the model.

    Returns:
        The predicted labels.
    """
    return model_output.sequences
