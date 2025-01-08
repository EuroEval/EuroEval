"""Utility functions related to the multiple-choice classification task group."""

import hashlib
import logging
import typing as t
from collections import defaultdict

import numpy as np
from datasets import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer, Trainer

if t.TYPE_CHECKING:
    from ..types import Labels, Predictions

logger = logging.getLogger("scandeval")


class MultipleChoiceClassificationTrainer(Trainer):
    """Trainer subclass for question answering tasks."""

    def evaluate(
        self,
        eval_dataset: "Dataset | None" = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float] | None:
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

        if metric_key_prefix == "test":
            preds_and_labels = postprocess_predictions_and_labels(
                predictions=output.predictions, dataset=eval_dataset
            )
            output.metrics.update(self.compute_metrics(preds_and_labels))

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(output.metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(
                        key
                    )

        # Only the main node log the results by default
        if self.args.should_log:
            self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,  # type: ignore[has-type]
            output.metrics,
        )
        return output.metrics


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
    question = sections[0]
    choices = sections[2:]

    # Sanity check
    for letter, choice in zip("abcde", choices):
        assert choice.startswith(f"{letter}. ")

    new_examples = tokenizer(
        text=[question] * len(choices), text_pair=[choice[3:] for choice in choices]
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
        prediction: str = mapping[np.argmax(preds).item()]
        label: str = mapping[np.argmax(labels).item()]
        all_predictions.append(prediction)
        all_labels.append(label)

    return all_predictions, all_labels
