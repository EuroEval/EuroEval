"""Functions related to text generation of models."""

import collections.abc as c
import logging
import sys
import typing as t
from pathlib import Path

from datasets import Dataset
from tqdm.auto import tqdm

from .enums import BatchingPreference, TaskGroup
from .exceptions import InvalidBenchmark, InvalidModel
from .logging_utils import get_pbar, log, log_once
from .metrics.bpc import bpc_metric
from .model_cache import (
    ModelCache,
    load_cached_model_outputs,
    split_dataset_into_cached_and_non_cached,
)
from .utils import clear_memory

if t.TYPE_CHECKING:
    from datasets import DatasetDict

    from .benchmark_modules import BenchmarkModule
    from .data_models import (
        BenchmarkConfig,
        DatasetConfig,
        GenerativeModelOutput,
        ModelConfig,
    )
    from .types import FailedInstance, IterationScores


def _labels_match(predicted: object, gold: object) -> bool:
    """Check whether a predicted label (or label sequence) matches the gold one.

    The comparison is case-insensitive, and for token-classification label
    sequences it is an exact element-wise match. For sequence classification this
    mirrors the task metric exactly. For token classification the metric is
    span-based F1 rather than element-wise equality, but failed instances there
    only arise when JSON parsing fails entirely (so the prediction defaults to all
    "o"); exact-match then correctly treats an all-"o" prediction against an
    all-"o" gold as not a failure, which is the intended behaviour.

    Args:
        predicted:
            The model's predicted label, or its list of per-token labels.
        gold:
            The gold label, or its list of per-token labels.

    Returns:
        Whether the prediction matches the gold label.
    """
    if isinstance(predicted, str) and isinstance(gold, str):
        return predicted.lower() == gold.lower()
    if isinstance(predicted, list) and isinstance(gold, list):
        predicted_norm = [
            label.lower() if isinstance(label, str) else label for label in predicted
        ]
        gold_norm = [
            label.lower() if isinstance(label, str) else label for label in gold
        ]
        return predicted_norm == gold_norm
    return predicted == gold


def generate(
    model: "BenchmarkModule",
    datasets: c.Sequence["DatasetDict"],
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
) -> c.Sequence["IterationScores"]:
    """Evaluate a model on a dataset through generation.

    Args:
        model:
            The model to evaluate.
        datasets:
            The datasets to evaluate on.
        model_config:
            The configuration of the model.
        benchmark_config:
            The configuration of the benchmark.
        dataset_config:
            The configuration of the dataset.

    Returns:
        A list of dictionaries containing the test scores.
    """
    # Set up the name of the model output cache. If we are testing then we save the
    # model outputs to a different cache and ensure that that cache is deleted before
    # the next test, to ensure that the tests are independent of each other
    if benchmark_config.debug:
        model_cache_dir = Path.cwd()
    else:
        model_cache_dir = Path(model_config.model_cache_dir)
    # Namespace the cache by BPC flag so different scoring methods do not collide.
    # MCF runs use the legacy unsuffixed cache path for backward compatibility.
    cache_suffix = "-bpc" if benchmark_config.use_bits_per_character else ""
    if hasattr(sys, "_called_from_test"):
        cache_name = f"{dataset_config.name}{cache_suffix}-model-outputs-test.json"
        (model_cache_dir / cache_name).unlink(missing_ok=True)
    elif benchmark_config.debug:
        cache_name = (
            f"{model_config.model_id}-{dataset_config.name}{cache_suffix}"
            "-model-outputs.json"
        )
    else:
        cache_name = f"{dataset_config.name}{cache_suffix}-model-outputs.json"

    cache = ModelCache(
        model_cache_dir=model_cache_dir,
        cache_name=cache_name,
        max_generated_tokens=dataset_config.max_generated_tokens,
        progress_bar=benchmark_config.progress_bar,
        store_metadata=benchmark_config.debug,
        indent_json_when_saving=benchmark_config.debug,
    )

    scores: list["IterationScores"] = list()
    iteration_errors: list["InvalidBenchmark"] = list()
    for idx in get_pbar(
        iterable=range(len(datasets)),
        desc="Benchmarking",
        disable=not benchmark_config.progress_bar,
    ):
        # We tolerate individual iterations failing (e.g. the model refusing to answer
        # in a way that produces no valid label), as long as at least one iteration
        # succeeds. This way we can still report the scores of the successful
        # iterations rather than discarding the entire evaluation.
        try:
            test_scores = generate_single_iteration(
                model=model,
                dataset=datasets[idx]["test"],
                cache=cache,
                dataset_config=dataset_config,
                benchmark_config=benchmark_config,
            )
        except InvalidBenchmark as e:
            log(
                f"Iteration {idx} failed and will be skipped: {e}",
                level=logging.WARNING,
            )
            iteration_errors.append(e)
            clear_memory()
            continue
        log(f"Test scores for iteration {idx}: {test_scores}", level=logging.DEBUG)
        scores.append(test_scores)
        clear_memory()

    # If every iteration that actually ran failed then there is nothing to report, so
    # we raise the first encountered error to abort the evaluation. If no iterations
    # ran at all (e.g. `num_iterations` is zero) there are no errors to raise, and we
    # simply return the empty list of scores, as before.
    if not scores and iteration_errors:
        raise iteration_errors[0]
    if iteration_errors:
        log(
            f"{len(iteration_errors):,} of {len(datasets):,} iterations failed and "
            f"were skipped; reporting the scores of the {len(scores):,} successful "
            "iterations.",
            level=logging.WARNING,
        )

    if not benchmark_config.debug:
        cache.remove()

    return scores


def generate_single_iteration(
    dataset: "Dataset",
    model: "BenchmarkModule",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
    cache: ModelCache,
) -> "IterationScores":
    """Evaluate a model on a dataset in a single iteration through generation.

    Args:
        dataset:
            The dataset to evaluate on.
        model:
            The model to evaluate.
        dataset_config:
            The configuration of the dataset.
        benchmark_config:
            The configuration of the benchmark.
        cache:
            The model output cache.

    Returns:
        A list of dictionaries containing the scores for each metric.

    Raises:
        InvalidModel:
            If the model's batching preference is not supported.
    """
    cache.load()

    # Split up the dataset into a cached and non-cached part, unless we are not
    # bootstrapping the samples. In that case, we just use the dataset as is.
    if dataset_config.bootstrap_samples:
        cached_dataset, non_cached_dataset = split_dataset_into_cached_and_non_cached(
            dataset=dataset, cache=cache
        )
    else:
        cached_dataset = Dataset.from_dict({})
        non_cached_dataset = dataset

    all_preds: list[str] = list()
    all_predicted_labels: list[object] = list()
    all_bpc_scores: list[float] = list()
    failed_instances: list["FailedInstance"] = list()

    if len(non_cached_dataset) > 0:
        itr: t.Iterable
        match model.batching_preference:
            case BatchingPreference.SINGLE_SAMPLE:
                itr = get_pbar(
                    iterable=non_cached_dataset,
                    disable=not benchmark_config.progress_bar,
                )
            case BatchingPreference.ALL_AT_ONCE:
                itr = [non_cached_dataset[:]]
            case _:
                raise InvalidModel(
                    f"The batching preference {model.batching_preference!r} is "
                    "currently not supported."
                )
                # NOTE: The code below can be used if we want to support batching for
                # generative models. But in that case, we have to deal with the naming
                # of the batch size variable, since it is currently
                # `finetuning_batch_size`, as it is only used during finetuning of
                # encoder models.
                # num_batches = len(non_cached_dataset) // benchmark_config.batch_size
                # if len(non_cached_dataset) % benchmark_config.batch_size != 0:
                #     num_batches += 1
                # itr = get_pbar(
                #     iterable=mit.batched(
                #         iterable=non_cached_dataset, n=benchmark_config.batch_size
                #     ),
                #     total=len(non_cached_dataset) // benchmark_config.batch_size,
                # )

        # Generate the completions for the non-cached examples
        for batch in itr:
            assert isinstance(batch, dict), (
                f"Expected a dictionary but got {type(batch)}."
            )

            single_sample_batch = (
                "text" in batch and isinstance(batch["text"], str)
            ) or ("messages" in batch and isinstance(batch["messages"][0], dict))
            if single_sample_batch:
                batch = {key: [value] for key, value in batch.items()}

            # Use score() for BPC, generate() for standard generation
            if benchmark_config.use_bits_per_character:
                model_output = model.score(inputs=batch)  # type: ignore[attr-defined]
            else:
                model_output = model.generate(inputs=batch)

            # In BPC mode we score via prompt_logprobs only, so the accuracy
            # label-extraction pipeline (and its metrics) is skipped entirely.
            if not benchmark_config.use_bits_per_character:
                # Extracted labels are the labels extracted from the generation - these
                # are in the language of the dataset. The predicted labels is the result
                # of mapping the extracted labels to the original labels in the dataset
                # (which are typically English).
                extracted_labels = model.extract_labels_from_generation(
                    input_batch=batch, model_output=model_output
                )
                if pred2extracted := dataset_config.prompt_label_mapping:
                    extracted_to_predicted = {
                        extracted: predicted
                        for predicted, extracted in pred2extracted.items()
                    }
                    model_output.predicted_labels = [
                        extracted_to_predicted.get(label, label).lower()
                        if isinstance(label, str)
                        else [
                            extracted_to_predicted.get(lbl, lbl).lower()
                            for lbl in label
                        ]
                        for label in extracted_labels
                    ]
                else:
                    model_output.predicted_labels = extracted_labels
                # Re-key the batch-local failed-instance indices to global indices
                # (into `all_preds`/`ground_truth`) so we can later check, per
                # failed sample, whether the fallback label was actually wrong.
                offset = len(all_preds)
                for failed_instance in model_output.failed_instances:
                    failed_instance["sample_index"] += offset
                all_preds.extend(extracted_labels)
                all_predicted_labels.extend(model_output.predicted_labels or [])

            failed_instances += model_output.failed_instances

            # Extended logging if we are running in debug mode
            if benchmark_config.debug:
                debug_log(
                    batch=batch,
                    model_output=model_output,
                    dataset_config=dataset_config,
                )

            cache.add_to_cache(model_inputs=batch, model_output=model_output)

            # Collect BPC scores for BPC evaluation
            if benchmark_config.use_bits_per_character and model_output.bpc_scores:
                all_bpc_scores.extend(model_output.bpc_scores)

            # If we are debugging then we save the cache often, but since this makes
            # evaluation slower, we do not do this by default
            if benchmark_config.debug:
                cache.save()

        if isinstance(itr, tqdm):
            itr.close()

        # Store the cache to disk
        cache.save()

    # Fetch the cached predictions for the cached examples
    if len(cached_dataset) > 0:
        model_output = load_cached_model_outputs(
            cached_dataset=cached_dataset, cache=cache
        )
        # As above, the accuracy label-extraction pipeline is skipped in BPC mode.
        if not benchmark_config.use_bits_per_character:
            extracted_labels = model.extract_labels_from_generation(
                input_batch=cached_dataset[:], model_output=model_output
            )
            if model_output.predicted_labels is None:
                if pred2extracted := dataset_config.prompt_label_mapping:
                    extracted_to_predicted = {
                        extracted: predicted
                        for predicted, extracted in pred2extracted.items()
                    }
                    model_output.predicted_labels = [
                        extracted_to_predicted.get(label, label).lower()
                        if isinstance(label, str)
                        else [
                            extracted_to_predicted.get(lbl, lbl).lower()
                            for lbl in label
                        ]
                        for label in extracted_labels
                    ]
                else:
                    model_output.predicted_labels = extracted_labels
            offset = len(all_preds)
            for failed_instance in model_output.failed_instances:
                failed_instance["sample_index"] += offset
            all_preds.extend(extracted_labels)
            all_predicted_labels.extend(model_output.predicted_labels or [])

        failed_instances += model_output.failed_instances

        # Collect BPC scores from cached outputs
        if benchmark_config.use_bits_per_character and model_output.bpc_scores:
            all_bpc_scores.extend(model_output.bpc_scores)

    if "label" in non_cached_dataset.column_names:
        non_cached_labels = non_cached_dataset["label"]
        if not isinstance(non_cached_labels, list):
            non_cached_labels = list(non_cached_labels)
        cached_labels = cached_dataset["label"]
        if not isinstance(cached_labels, list):
            cached_labels = list(cached_labels)
        ground_truth = [
            label.lower() if isinstance(label, str) else label
            for label in non_cached_labels + cached_labels
        ]
    elif "labels" in non_cached_dataset.column_names:
        non_cached_labels = non_cached_dataset["labels"]
        if not isinstance(non_cached_labels, list):
            non_cached_labels = list(non_cached_labels)
        cached_labels = cached_dataset["labels"]
        if not isinstance(cached_labels, list):
            cached_labels = list(cached_labels)
        ground_truth = [
            [label.lower() if isinstance(label, str) else label for label in label_list]
            for label_list in non_cached_labels + cached_labels
        ]
    elif "target_text" in non_cached_dataset.column_names:
        non_cached_labels = non_cached_dataset["target_text"]
        if not isinstance(non_cached_labels, list):
            non_cached_labels = list(non_cached_labels)
        cached_labels = cached_dataset["target_text"]
        if not isinstance(cached_labels, list):
            cached_labels = list(cached_labels)
        ground_truth = non_cached_labels + cached_labels
    else:
        log_once(
            "No labels found in the dataset. We assume that this is intentional, and "
            "will not supply any ground truth labels for evaluation.",
            level=logging.DEBUG,
        )
        ground_truth = []

    # For BPC evaluation, only compute BPC metric (ignore accuracy metrics)
    if benchmark_config.use_bits_per_character:
        if all_bpc_scores:
            bpc_score = bpc_metric(
                predictions=all_bpc_scores,
                references=[],
                dataset=dataset,
                dataset_config=dataset_config,
                benchmark_config=benchmark_config,
            )
            # Key the score under the metric's name so it aggregates correctly in
            # `log_scores`/`aggregate_scores`.
            return {bpc_metric.name: bpc_score, "failed_instances": failed_instances}  # ty: ignore[invalid-return-type]
        else:
            log_once(
                "BPC evaluation requested but no BPC scores were computed. "
                "Assigning infinite BPC (worst score).",
                level=logging.WARNING,
            )
            return {bpc_metric.name: float("inf"), "failed_instances": failed_instances}
    else:
        # A sample is only a genuine scoring failure if the fallback label that
        # was assigned (after no clean label could be parsed) is *also* wrong.
        # Fallbacks that happen to land on the correct label are not failures, so
        # we drop them here to keep `num_failed_instances` meaningful.
        failed_instances = [
            failed_instance
            for failed_instance in failed_instances
            if (idx := failed_instance["sample_index"]) < len(ground_truth)
            and idx < len(all_predicted_labels)
            and not _labels_match(all_predicted_labels[idx], ground_truth[idx])
        ]
        metrics_scores = model.compute_metrics(
            model_outputs_and_labels=(all_preds, ground_truth),
            dataset=dataset,
            benchmark_config=benchmark_config,
        )
        return {**metrics_scores, "failed_instances": failed_instances}


def debug_log(
    batch: dict[str, t.Any],
    model_output: "GenerativeModelOutput",
    dataset_config: "DatasetConfig",
) -> None:
    """Log inputs and outputs for debugging purposes.

    Args:
        batch:
            The batch of examples to evaluate on.
        model_output:
            The output of the model.
        dataset_config:
            The configuration of the dataset.

    Raises:
        InvalidBenchmark:
            If the dataset is not passed to the metric.
    """
    assert model_output.predicted_labels is not None, (
        "The predicted labels should not be None if the model output is not None."
    )

    match dataset_config.task.task_group:
        case TaskGroup.TOKEN_CLASSIFICATION:
            log_msgs = [""]
            for tokens, predictions, labels in zip(
                batch["tokens"],
                t.cast(c.Sequence[str], model_output.predicted_labels),
                batch["labels"],
            ):
                predictions = [tag.upper() for tag in predictions]
                sample = list(zip(tokens, predictions, labels))
                log_batches = [
                    [("Tokens: ", "Predictions: ", "Labels: ")] + sample[i : i + 10]
                    for i in range(0, len(sample), 10)
                ]
                for log_batch in log_batches:
                    lengths = [len(max(triple, key=len)) for triple in log_batch]
                    log_batch = [
                        [f"{x:<{length}}" for x in triple]
                        for triple, length in zip(log_batch, lengths)
                    ]
                    tokens = [triple[0] for triple in log_batch]
                    predictions = [triple[1] for triple in log_batch]
                    labels = [triple[2] for triple in log_batch]
                    log_msgs.append(
                        "\t".join(tokens)
                        + "\n"
                        + "\t".join(predictions)
                        + "\n"
                        + "\t".join(labels)
                    )
            log("\n\n".join(log_msgs), level=logging.DEBUG)
            return

        case (
            TaskGroup.SEQUENCE_CLASSIFICATION | TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
        ):
            if "label" in batch:
                labels = [
                    dataset_config.prompt_label_mapping.get(label, label).lower()
                    for label in batch["label"]
                ]
            else:
                labels = [None] * len(model_output.predicted_labels)

        case TaskGroup.QUESTION_ANSWERING:
            model_output.predicted_labels = [
                prediction["prediction_text"]  # ty: ignore[invalid-argument-type]
                for prediction in model_output.predicted_labels
                if isinstance(prediction, dict)
            ]
            labels = [label["answers"]["text"][0] for label in batch["label"]]

        case TaskGroup.TEXT_TO_TEXT:
            labels = batch["target_text"]

        case _:
            raise InvalidBenchmark(
                f"The task group '{dataset_config.task.task_group}' is not supported."
            )

    if "messages" in batch:
        input_texts = [messages[-1]["content"] for messages in batch["messages"]]
    else:
        input_texts = batch["text"]

    metadata_keys: c.Sequence[str] = [
        key
        for key in batch.keys()
        if key not in ["text", "messages", "label", "labels", "target_text"]
    ]

    for idx in range(len(input_texts)):
        data_to_log: dict[str, t.Any] = {
            "Input": input_texts[idx],
            "Raw output": model_output.sequences[idx],
            "Prediction": model_output.predicted_labels[idx],
        }
        if labels[idx]:
            data_to_log["Label"] = labels[idx]
        data_to_log |= {key.capitalize(): batch[key][idx] for key in metadata_keys}
        log(
            "\n".join(f"{key}: {value!r}" for key, value in data_to_log.items()),
            level=logging.DEBUG,
        )
