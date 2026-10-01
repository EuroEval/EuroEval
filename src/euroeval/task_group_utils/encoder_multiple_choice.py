"""Batching and training support for encoder multiple-choice tasks."""

import typing as t

import torch
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.utils import ModelOutput


class VariableChoiceCollator:
    """Pad tokens and choice dimensions for batches with mixed choice counts."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Initialise the collator.

        Args:
            tokenizer: The tokenizer used to pad token sequences.
        """
        self._token_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, padding="longest"
        )

    def __call__(self, features: list[dict[str, t.Any]]) -> dict[str, torch.Tensor]:
        """Collate nested question-choice features without inventing valid answers.

        Args:
            features: One tokenized feature dict per question.

        Returns:
            Padded tensors, labels, and a boolean mask distinguishing real choices.
        """
        counts = [len(feature["input_ids"]) for feature in features]
        max_choices = max(counts)
        flattened = [
            {key: value[choice_idx] for key, value in feature.items() if key != "label"}
            for feature, count in zip(features, counts)
            for choice_idx in range(count)
        ]
        token_batch = self._token_collator(flattened)
        batch_size = len(features)
        result: dict[str, torch.Tensor] = {}
        for key, value in token_batch.items():
            reshaped = value.new_zeros((batch_size, max_choices, *value.shape[1:]))
            cursor = 0
            for batch_idx, count in enumerate(counts):
                reshaped[batch_idx, :count] = value[cursor : cursor + count]
                cursor += count
            result[key] = reshaped
        result["labels"] = torch.tensor([feature["label"] for feature in features])
        result["choice_mask"] = torch.arange(max_choices).unsqueeze(0) < torch.tensor(
            counts
        ).unsqueeze(1)
        return result


class VariableChoiceTrainer(Trainer):
    """Trainer that excludes batch-padding choices from loss and predictions."""

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, t.Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ModelOutput]:
        """Compute classification loss after masking nonexistent choices.

        Returns:
            The loss, optionally paired with model outputs.
        """
        del num_items_in_batch
        inputs = dict(inputs)
        choice_mask = inputs.pop("choice_mask").bool()
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.masked_fill(
            ~choice_mask, torch.finfo(outputs.logits.dtype).min
        )
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if return_outputs:
            outputs.logits = logits
            return loss, outputs
        return loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, t.Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Return predictions with nonexistent choices assigned a minimal score."""
        choice_mask = inputs["choice_mask"].to(self.args.device).bool()
        loss, logits, labels = super().prediction_step(
            model=model,
            inputs=inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )
        if logits is not None and not prediction_loss_only:
            logits = logits.masked_fill(~choice_mask, torch.finfo(logits.dtype).min)
        return loss, logits, labels
