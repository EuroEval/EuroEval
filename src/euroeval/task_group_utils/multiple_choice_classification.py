"""Utility functions related to the multiple-choice classification task group."""

import hashlib
import typing as t
from collections import defaultdict

import numpy as np
from transformers.trainer import Trainer

from ..exceptions import InvalidBenchmark
from ..string_utils import CHOICE_LETTERS
from .cloze import parse_bare_question_and_choices

if t.TYPE_CHECKING:
    from datasets import Dataset
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_base import BatchEncoding

    from ..types import Labels, Predictions


class MultipleChoiceClassificationTrainer(Trainer):
    """Trainer subclass for multiple-choice classification tasks."""

    def evaluate(  # ty: ignore[invalid-method-override]
        self,
        eval_dataset: "Dataset | None" = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Evaluate the model on the given dataset.

        Args:
            eval_dataset:
                The dataset to evaluate on. If None, then use the stored evaluation
                dataset.
            ignore_keys:
                The keys to ignore when computing the metrics.
            metric_key_prefix:
                The prefix to use for the metric keys.

        Returns:
            The metrics computed on the evaluation dataset.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)  # ty: ignore[invalid-argument-type]

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        predictions = output.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        assert isinstance(predictions, np.ndarray)

        metrics = output.metrics
        assert metrics is not None

        if metric_key_prefix == "test":
            assert eval_dataset is not None, (
                "eval_dataset must be provided when metric_key_prefix is 'test'."
            )
            preds_and_labels = postprocess_predictions_and_labels(
                predictions=predictions, dataset=eval_dataset
            )
            assert self.compute_metrics is not None
            new_metrics = self.compute_metrics(preds_and_labels)  # ty: ignore[invalid-argument-type]
            metrics.update(new_metrics)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        # Only the main node log the results by default
        if self.args.should_log:
            self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )
        return metrics


def prepare_examples(
    examples: "BatchEncoding", tokeniser: "PreTrainedTokenizer"
) -> "BatchEncoding":
    """Prepare the features.

    Args:
        examples:
            The examples to prepare.
        tokeniser:
            The tokeniser to use to prepare the examples.

    Returns:
        The prepared examples.
    """
    doc: str = examples["text"][0]

    # Recover the bare question and the individual choice texts from the formatted
    # prompt. This is the canonical parser shared with cloze/BPC scoring, so the two
    # paths cannot drift in how they split the choices block out of the prompt.
    context_and_question, choices = parse_bare_question_and_choices(doc)
    assert len(choices) > 0, "No choices found in the document."

    gold_letter = examples["label"][0]
    new_examples = tokeniser(
        text=[context_and_question] * len(choices),
        text_pair=choices,
        padding=True,
        truncation=True,
    )
    new_examples["label"] = [
        int(letter == gold_letter) for letter, _ in zip(CHOICE_LETTERS, choices)
    ]
    new_examples["id"] = [hashlib.md5(string=doc.encode()).hexdigest()] * len(choices)
    return new_examples


def postprocess_predictions_and_labels(
    predictions: np.ndarray, dataset: "Dataset"
) -> tuple["Predictions", "Labels"]:
    """Postprocess the predictions and labels.

    Args:
        predictions:
            The model predictions, of shape (num_examples, 2), corresponding to the
            False/True probabilities for each example.
        dataset:
            The dataset containing the examples.

    Returns:
        The postprocessed predictions and labels.

    Raises:
        InvalidBenchmark:
            If the predictions are not a 2D array with shape (num_examples, 2).
    """
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise InvalidBenchmark(
            "Predictions must be a 2D array with shape (num_examples, 2). Found "
            f"shape {predictions.shape}."
        )

    mapping = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}

    all_predictions: list[str] = list()
    all_labels: list[str] = list()

    pred_label_dict = defaultdict(list)
    for pred_arr, example in zip(predictions, dataset):
        pred_label_dict[example["id"]].append((pred_arr[1], example["label"]))

    # Compute the final predictions and labels
    for id_ in set(dataset["id"]):
        preds, labels = zip(*pred_label_dict[id_])

        # Some IDs appear multiple times in the dataset, since we are bootstrapping.
        # Here we separate them into their respective groups.
        assert len(labels) % sum(labels) == 0, (
            "The number of labels is not divisible by the sum of the labels."
        )
        group_size = len(labels) // sum(labels)
        preds_groups = [
            preds[i : i + group_size] for i in range(0, len(preds), group_size)
        ]
        labels_groups = [
            labels[i : i + group_size] for i in range(0, len(labels), group_size)
        ]
        for preds_group, labels_group in zip(preds_groups, labels_groups):
            prediction: str = mapping[np.argmax(preds_group).item()]
            label: str = mapping[np.argmax(labels_group).item()]
            all_predictions.append(prediction)
            all_labels.append(label)

    return all_predictions, all_labels
