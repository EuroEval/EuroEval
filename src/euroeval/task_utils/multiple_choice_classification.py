"""Utility functions related to the multiple-choice classification task group."""

import hashlib
import logging
import re
import typing as t
from collections import defaultdict

import numpy as np
from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer import Trainer

if t.TYPE_CHECKING:
    from ..types import Labels, Predictions

logger = logging.getLogger("euroeval")


class MultipleChoiceClassificationTrainer(Trainer):
    """Trainer subclass for question answering tasks."""

    def evaluate(  # type: ignore[override]
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
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        predictions = output.predictions
        assert isinstance(predictions, np.ndarray)

        metrics = output.metrics
        assert metrics is not None

        if metric_key_prefix == "test":
            preds_and_labels = postprocess_predictions_and_labels(
                predictions=predictions, dataset=eval_dataset
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
            output.metrics,
        )
        return metrics


def prepare_examples(
    examples: "BatchEncoding", tokenizer: "PreTrainedTokenizer"
) -> "BatchEncoding":
    """Prepare the features.

    Args:
        examples:
            The examples to prepare.
        tokenizer:
            The tokenizer to use to prepare the examples.

    Returns:
        The prepared examples.
    """
    doc: str = examples["text"][0]
    sections = doc.split("\n")

    choice_idxs = [
        idx
        for idx, section in enumerate(sections)
        if re.match(pattern=r"^[a-e]\. ", string=section) is not None
    ]
    choices = [sections[idx] for idx in choice_idxs]

    # Check that the choices are present, and that all of them are at the end
    assert len(choices) > 0, "No choices found in the document."
    assert all(
        choice_idx == len(sections) - i
        for i, choice_idx in enumerate(sorted(choice_idxs, reverse=True), start=1)
    ), "Choices are not at the end of the document."

    question_idx = min(choice_idxs) - 2  # -2 to remove the 'Choices:' line
    context_and_question = "\n".join(sections[: question_idx + 1]).strip()

    new_examples = tokenizer(
        text=[context_and_question] * len(choices),
        text_pair=[choice[3:] for choice in choices],
        padding=True,
        truncation=True,
    )
    new_examples["label"] = [
        int(choice.startswith(f"{letter}. ") and letter == examples["label"][0])
        for letter, choice in zip("abcde", choices)
    ]
    new_examples["id"] = [hashlib.md5(string=doc.encode()).hexdigest()] * len(choices)
    return new_examples


def postprocess_predictions_and_labels(
    predictions: np.ndarray, dataset: "Dataset"
) -> tuple["Predictions", "Labels"]:
    """Postprocess the predictions and labels.

    Args:
        predictions:
            The model predictions, of shape (num_examples, 2).
        dataset:
            The dataset containing the examples.

    Returns:
        The postprocessed predictions and labels.
    """
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
