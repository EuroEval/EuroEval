"""Tests for encoder multiple-choice batching and prediction masking."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import Dataset
from transformers import EvalPrediction, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from euroeval.task_group_utils.encoder_multiple_choice import (
    VariableChoiceCollator,
    VariableChoiceTrainer,
)


def test_variable_choice_collator_marks_padding_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The collator batches variable choices and flags only real answer rows."""
    token_collator = MagicMock(
        return_value={
            "input_ids": torch.tensor([[10, 11], [20, 21], [30, 31]]),
            "attention_mask": torch.ones((3, 2), dtype=torch.long),
        }
    )
    monkeypatch.setattr(
        "euroeval.task_group_utils.encoder_multiple_choice.DataCollatorWithPadding",
        lambda **kwargs: token_collator,
    )
    collator = VariableChoiceCollator(tokenizer=MagicMock())

    batch = collator(
        [
            {"input_ids": [[10, 11]], "attention_mask": [[1, 1]], "label": 0},
            {
                "input_ids": [[20, 21], [30, 31]],
                "attention_mask": [[1, 1], [1, 1]],
                "label": 1,
            },
        ]
    )

    assert batch["input_ids"].shape == (2, 2, 2)
    assert batch["choice_mask"].tolist() == [[True, False], [True, True]]
    assert batch["input_ids"][0, 1].tolist() == [0, 0]
    assert batch["labels"].tolist() == [0, 1]


def test_variable_choice_trainer_evaluate_does_not_argmax_aggregation_padding(
    tmp_path: Path,
) -> None:
    """Evaluation preserves row-wise argmax across batches with different widths."""

    class EvaluationDataset(Dataset):
        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            """Return one example with its real choice width."""
            if index == 0:
                return {
                    "input_ids": torch.tensor([-150.0, -151.0]),
                    "choice_mask": torch.tensor([True, True]),
                    "labels": torch.tensor(0),
                }
            return {
                "input_ids": torch.tensor([-200.0, -300.0, -400.0]),
                "choice_mask": torch.tensor([True, True, True]),
                "labels": torch.tensor(0),
            }

        def __len__(self) -> int:
            """Return the number of evaluation examples."""
            return 2

    class LogitModel(torch.nn.Module):
        def forward(
            self, input_ids: torch.FloatTensor, labels: torch.LongTensor | None = None
        ) -> SequenceClassifierOutput:
            """Use input values directly as choice logits.

            Returns:
                A model output containing the choice logits.
            """
            del labels
            return SequenceClassifierOutput(logits=input_ids)

    captured_predictions: list[int] = []

    def capture_predictions(prediction: EvalPrediction) -> dict[str, float]:
        """Capture the IDs produced by evaluation aggregation.

        Returns:
            A fixed accuracy score for the evaluation run.
        """
        captured_predictions.extend(prediction.predictions.tolist())
        return {"accuracy": 1.0}

    trainer = VariableChoiceTrainer(
        model=LogitModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_eval_batch_size=1,
            remove_unused_columns=False,
            report_to=[],
        ),
        eval_dataset=EvaluationDataset(),
        compute_metrics=capture_predictions,
    )

    metrics = trainer.evaluate()

    assert captured_predictions == [0, 0]
    assert metrics["eval_accuracy"] == 1.0
