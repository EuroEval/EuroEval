"""Utility functions related to the question-answering task group."""

import collections.abc as c
import logging
import typing as t
from collections import defaultdict

import evaluate
import numpy as np
from evaluate import EvaluationModule
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer

from ..data_models import BenchmarkConfig, DatasetConfig, GenerativeModelOutput
from ..tokenization_utils import get_special_token_metadata
from ..utils import raise_if_model_output_contains_nan_values

if t.TYPE_CHECKING:
    import torch.nn as nn
    from datasets.arrow_dataset import Dataset
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import BatchEncoding
    from transformers.trainer_callback import TrainerCallback
    from transformers.trainer_utils import EvalPrediction
    from transformers.training_args import TrainingArguments

    from ..types import Labels, Predictions

logger = logging.getLogger("euroeval")


class QuestionAnsweringTrainer(Trainer):
    """Trainer subclass for question answering tasks."""

    def __init__(
        self,
        model: "PreTrainedModel | nn.Module",
        processing_class: "PreTrainedTokenizerBase",
        args: "TrainingArguments",
        train_dataset: "Dataset",
        eval_dataset: "Dataset",
        compute_metrics: "c.Callable[[EvalPrediction], dict[str, float]]",
        callbacks: "list[TrainerCallback]",
        data_collator: "c.Callable",
    ) -> None:
        """Initialise the trainer."""
        super().__init__(
            model=model,
            processing_class=processing_class,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            data_collator=data_collator,
        )

        # Get the CLS token id for the tokenizer
        if self.tokenizer is not None:
            assert isinstance(self.tokenizer, PreTrainedTokenizerBase)
            special_token_metadata = get_special_token_metadata(self.tokenizer)
            self.cls_token_id = special_token_metadata["cls_token_id"]

        # Set the label names
        self.label_names = ["start_positions", "end_positions"]

    def evaluate(  # type: ignore[override]
        self,
        eval_dataset: "Dataset | None" = None,
        orig_eval_dataset: "Dataset | None" = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Evaluate the model on the given dataset.

        Args:
            eval_dataset:
                The dataset to evaluate on. If None, then use the stored evaluation
                dataset.
            orig_eval_dataset:
                The original evaluation dataset, before any postprocessing. If None,
                then use the stored original evaluation dataset.
            ignore_keys:
                The keys to ignore when computing the metrics.
            metric_key_prefix:
                The prefix to use for the metric keys.

        Returns:
            The metrics computed on the evaluation dataset.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics  # type: ignore[has-type]
        self.compute_metrics = None
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        predictions = output.predictions
        assert isinstance(predictions, tuple)

        metrics = output.metrics
        assert metrics is not None

        if orig_eval_dataset is not None:
            preds_and_labels = postprocess_predictions_and_labels(
                predictions=predictions,  # type: ignore[arg-type]
                dataset=orig_eval_dataset,
                prepared_dataset=eval_dataset,
                cls_token_index=self.cls_token_id,
            )
            assert self.compute_metrics is not None
            new_metrics = self.compute_metrics(preds_and_labels)  # type: ignore[arg-type]
            metrics.update(new_metrics)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        # Only the main node log the results by default
        if self.args.should_log:
            self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,  # type: ignore[has-type]
            metrics,
        )
        return metrics


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
    if model_output_dtype in [np.float16, np.float32, np.float64]:
        predictions = np.asarray(model_outputs).argmax(axis=-1)
    else:
        predictions = model_outputs

    results: dict[str, float] = dict()
    for cfg in dataset_config.task.metrics:
        metric = metrics[cfg.name]
        assert isinstance(metric, EvaluationModule)
        score_dict: dict[str, float] | None = metric.compute(
            predictions=predictions, references=labels, **cfg.compute_kwargs
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
    raw_predictions = model_output.sequences
    predictions = [
        dict(id=id, prediction_text=predicted_answer.lower(), no_answer_probability=0.0)
        for id, predicted_answer in zip(input_batch["id"], raw_predictions)
    ]
    return predictions


def prepare_train_examples(
    examples: "BatchEncoding", tokenizer: "PreTrainedTokenizer"
) -> "BatchEncoding":
    """Prepare the features for training.

    Args:
        examples:
            The examples to prepare.
        tokenizer:
            The tokenizer to use to prepare the examples.

    Returns:
        The prepared examples.
    """
    # Some of the questions have lots of whitespace on the left, which is not useful
    # and will make the truncation of the context fail (the tokenized question will
    # take a lots of space). So we remove that left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Extract special token metadata from the tokenizer
    special_token_metadata = get_special_token_metadata(tokenizer=tokenizer)
    has_cls_token = special_token_metadata["has_cls_token"]
    has_sep_token = special_token_metadata["has_sep_token"]
    cls_token_id = special_token_metadata["cls_token_id"]
    cls_token = special_token_metadata["cls_token"]
    sep_token = special_token_metadata["sep_token"]

    # If the tokenizer is not adding special tokens, then we add them manually
    if not has_cls_token and not has_sep_token:
        examples["question"] = [
            f"{cls_token}{q}{sep_token}" for q in examples["question"]
        ]
        examples["context"] = [f"{c}{sep_token}" for c in examples["context"]]

    # Set the stride used during tokenization, when the context is long enough to be
    # split into several features. Since we are always keeping the question tokens, we
    # need to make sure that the stride does not exceed the resulting maximum context
    # length.
    max_question_tokens = max(len(tokenizer(q).input_ids) for q in examples["question"])
    num_special_tokens = int(has_cls_token) + int(has_sep_token)
    stride = tokenizer.model_max_length // 4
    max_length = tokenizer.model_max_length - stride
    stride = min(stride, max_length - max_question_tokens - num_special_tokens)
    max_length = tokenizer.model_max_length - stride

    # Tokenize our examples with truncation and padding, but keep the overflows using a
    # stride. This results in one example possible giving several features when a
    # context is long, each of those features having a context that overlaps a bit the
    # context of the previous feature.
    tokenized_examples = tokenizer(
        text=examples["question"],
        text_pair=examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we
    # need a map from a feature to its corresponding example. This key gives us just
    # that
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # The offset mappings will give us a map from token to character position in the
    # original context. This will help us compute the start_positions and
    # end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Initialise the start- and end positions of the answers
    tokenized_examples["start_positions"] = list()
    tokenized_examples["end_positions"] = list()

    for i, offsets in enumerate(offset_mapping):
        # Get the input IDs for the current example
        input_ids = tokenized_examples.input_ids[i]

        # We will label impossible answers with the index of the CLS token
        cls_index = input_ids.index(cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context
        # and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Manually ensure that the special tokens are set to None in `sequence_ids`
        for special_token in tokenizer.special_tokens_map.keys():
            if hasattr(tokenizer, f"{special_token}_id"):
                special_token_id = getattr(tokenizer, f"{special_token}_id")
                if special_token_id is not None:
                    sequence_ids = [
                        None if token_id == special_token_id else seq_id
                        for token_id, seq_id in zip(input_ids, sequence_ids)
                    ]

        # One example can give several spans, this is the index of the example
        # containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples.start_positions.append(cls_index)
            tokenized_examples.end_positions.append(cls_index)

        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is
            # labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples.start_positions.append(cls_index)
                tokenized_examples.end_positions.append(cls_index)

            # Otherwise move the token_start_index and token_end_index to the two ends
            # of the answer. Note: we could go after the last offset if the answer is
            # the last word (edge case).
            else:
                while (
                    token_start_index <= token_end_index
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                token_start_index -= 1
                tokenized_examples.start_positions.append(token_start_index)
                while (
                    token_start_index <= token_end_index
                    and offsets[token_end_index][1] >= end_char
                ):
                    token_end_index -= 1
                token_end_index += 1
                tokenized_examples.end_positions.append(token_end_index)
                assert token_end_index >= token_start_index

    return tokenized_examples


def prepare_test_examples(
    examples: "BatchEncoding", tokenizer: "PreTrainedTokenizer"
) -> "BatchEncoding":
    """Prepare test examples.

    Args:
        examples:
            Dictionary of test examples.
        tokenizer:
            The tokenizer used to preprocess the examples.

    Returns:
        The prepared test examples.
    """
    # Some of the questions have lots of whitespace on the left, which is not useful
    # and will make the truncation of the context fail (the tokenized question will
    # take a lots of space). So we remove that left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Extract special token metadata from the tokenizer
    special_token_metadata = get_special_token_metadata(tokenizer=tokenizer)
    has_cls_token = special_token_metadata["has_cls_token"]
    has_sep_token = special_token_metadata["has_sep_token"]
    cls_token = special_token_metadata["cls_token"]
    sep_token = special_token_metadata["sep_token"]

    # If the tokenizer is not adding special tokens, then we add them manually
    if not has_cls_token and not has_sep_token:
        examples["question"] = [
            f"{cls_token}{q}{sep_token}" for q in examples["question"]
        ]
        examples["context"] = [f"{c}{sep_token}" for c in examples["context"]]

    # Set the stride used during tokenization, when the context is long enough to be
    # split into several features. Since we are always keeping the question tokens, we
    # need to make sure that the stride does not exceed the resulting maximum context
    # length.
    max_question_tokens = max(len(tokenizer(q).input_ids) for q in examples["question"])
    num_special_tokens = int(has_cls_token) + int(has_sep_token)
    stride = tokenizer.model_max_length // 4
    max_length = tokenizer.model_max_length - stride
    stride = min(stride, max_length - max_question_tokens - num_special_tokens)
    max_length = tokenizer.model_max_length - stride

    # Tokenize our examples with truncation and maybe padding, but keep the overflows
    # using a stride. This results in one example possible giving several features when
    # a context is long, each of those features having a context that overlaps a bit
    # the context of the previous feature.
    tokenized_examples = tokenizer(
        text=examples["question"],
        text_pair=examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we
    # need a map from a feature to its corresponding example. This key gives us just
    # that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the id that gave us this feature and we will store the offset mappings.
    tokenized_examples["id"] = list()

    for i in range(len(tokenized_examples.input_ids)):
        # Grab the sequence corresponding to that example (to know what is the context
        # and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example
        # containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples.id.append(examples["id"][sample_index])

        # Set to (-1, -1) the offset_mapping that are not part of the context so it's
        # easy to determine if a token position is part of the context or not.
        tokenized_examples.offset_mapping[i] = [
            (o if sequence_ids[k] == context_index else (-1, -1))
            for k, o in enumerate(tokenized_examples.offset_mapping[i])
        ]

    return tokenized_examples


def postprocess_predictions_and_labels(
    predictions: tuple[np.ndarray, np.ndarray],
    dataset: "Dataset",
    prepared_dataset: "Dataset",
    cls_token_index: int,
) -> tuple[list[dict], list[dict]]:
    """Postprocess the predictions and labels, to allow easier metric computation.

    Args:
        predictions:
            A pair of (start_logits, end_logits) predictions.
        dataset:
            The dataset containing the examples.
        prepared_dataset:
            The dataset containing the prepared examples.
        cls_token_index:
            The index of the CLS token.

    Returns:
        The postprocessed predictions and labels.
    """
    all_start_logits, all_end_logits = predictions

    # Build a map from an example to its corresponding features, being the blocks of
    # text from the context that we're feeding into the model. An example can have
    # multiple features/blocks if it has a long context.
    id_to_index = {k: i for i, k in enumerate(dataset["id"])}
    features_per_example = defaultdict(list)
    for i, feature in enumerate(prepared_dataset):
        id = feature["id"]
        example_index = id_to_index[id]
        features_per_example[example_index].append(i)

    # Loop over all the examples
    prediction_list: list[dict[str, t.Any]] = list()
    labels = list()
    for example_index, example in enumerate(dataset):
        # Extract the best valid answer associated with the current example
        best_answer = find_best_answer(
            all_start_logits=all_start_logits,
            all_end_logits=all_end_logits,
            prepared_dataset=prepared_dataset,
            feature_indices=features_per_example[example_index],
            context=example["context"],
            max_answer_length=30,
            num_best_logits=20,
            min_null_score=0.0,
            cls_token_index=cls_token_index,
        )

        # Create the final prediction dictionary, to be added to the list of
        # predictions
        prediction = dict(
            id=example["id"], prediction_text=best_answer, no_answer_probability=0.0
        )

        # Add the answer to the list of predictions
        prediction_list.append(prediction)

        # Create the associated reference dictionary, to be added to the list of
        # references
        label = dict(
            id=example["id"],
            answers=dict(
                text=example["answers"]["text"],
                answer_start=example["answers"]["answer_start"],
            ),
        )

        # Add the answer and label to the list of predictions and labels, respectively
        labels.append(label)

    return prediction_list, labels


def find_best_answer(
    all_start_logits: np.ndarray,
    all_end_logits: np.ndarray,
    prepared_dataset: "Dataset",
    feature_indices: list[int],
    context: str,
    max_answer_length: int,
    num_best_logits: int,
    min_null_score: float,
    cls_token_index: int,
) -> str:
    """Find the best answer for a given example.

    Args:
        all_start_logits:
            The start logits for all the features.
        all_end_logits:
            The end logits for all the features.
        prepared_dataset:
            The dataset containing the prepared examples.
        feature_indices:
            The indices of the features associated with the current example.
        context:
            The context of the example.
        max_answer_length:
            The maximum length of the answer.
        num_best_logits:
            The number of best logits to consider.
        min_null_score:
            The minimum score an answer can have.
        cls_token_index:
            The index of the CLS token.

    Returns:
        The best answer for the example.
    """
    # Loop through all the features associated to the current example
    valid_answers = list()
    for feature_index in feature_indices:
        # Get the features associated with the current example
        features = prepared_dataset[feature_index]

        # Get the predictions of the model for this feature
        start_logits = all_start_logits[feature_index]
        end_logits = all_end_logits[feature_index]

        # Update minimum null prediction
        cls_index = features["input_ids"].index(cls_token_index)
        feature_null_score = (start_logits[cls_index] + end_logits[cls_index]).item()
        if min_null_score < feature_null_score:
            min_null_score = feature_null_score

        # Find the valid answers for the feature
        valid_answers_for_feature = find_valid_answers(
            start_logits=start_logits,
            end_logits=end_logits,
            offset_mapping=features["offset_mapping"],
            context=context,
            max_answer_length=max_answer_length,
            num_best_logits=num_best_logits,
            min_null_score=min_null_score,
        )
        valid_answers.extend(valid_answers_for_feature)

    # In the very rare edge case we have not a single non-null prediction, we create a
    # fake prediction to avoid failure
    if not valid_answers:
        return ""

    # Otherwise, we select the answer with the largest score as the best answer, and
    # return it
    best_answer_dict = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
    return best_answer_dict["text"]


def find_valid_answers(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    offset_mapping: list[tuple[int, int]],
    context: str,
    max_answer_length: int,
    num_best_logits: int,
    min_null_score: float,
) -> list[dict]:
    """Find the valid answers from the start and end indexes.

    Args:
        start_logits:
            The logits for the start of the answer.
        end_logits:
            The logits for the end of the answer.
        offset_mapping:
            The offset mapping, being a list of pairs of integers for each token index,
            containing the start and end character index in the original context.
        context:
            The context of the example.
        max_answer_length:
            The maximum length of the answer.
        num_best_logits:
            The number of best logits to consider. Note that this function will run in
            O(`num_best_logits` ^ 2) time.
        min_null_score:
            The minimum score an answer can have.

    Returns:
        A list of the valid answers, each being a dictionary with keys "text" and
        "score", the score being the sum of the start and end logits.
    """
    # Fetch the top-k predictions for the start- and end token indices
    start_indexes = np.argsort(start_logits)[-1 : -num_best_logits - 1 : -1].tolist()
    end_indexes = np.argsort(end_logits)[-1 : -num_best_logits - 1 : -1].tolist()

    # We loop over all combinations of starting and ending indexes for valid answers
    valid_answers = list()
    for start_index in start_indexes:
        for end_index in end_indexes:
            # If the starting or ending index is out-of-scope, meaning that they are
            # either out of bounds or correspond to part of the input_ids that are not
            # in the context, then we skip this index
            if (
                start_index >= len(offset_mapping)
                or end_index >= len(offset_mapping)
                or tuple(offset_mapping[start_index]) == (-1, -1)
                or tuple(offset_mapping[end_index]) == (-1, -1)
            ):
                continue

            # Do not consider answers with a length that is either negative or greater
            # than the context length
            max_val = max_answer_length + start_index - 1
            if end_index < start_index or end_index > max_val:
                continue

            # If we got to this point then the answer is valid, so we store the
            # corresponding start- and end character indices in the original context,
            # and from these extract the answer
            start_char = offset_mapping[start_index][0]
            end_char = offset_mapping[end_index][1]
            text = context[start_char:end_char]

            # Compute the score of the answer, being the sum of the start and end
            # logits. Intuitively, this indicates how likely the answer is to be
            # correct, and allows us to pick the best valid answer.
            score = start_logits[start_index] + end_logits[end_index]

            # Add the answer to the list of valid answers, if the score is greater
            # than the minimum null score
            if score > min_null_score:
                valid_answers.append(dict(score=score, text=text))

    return valid_answers
