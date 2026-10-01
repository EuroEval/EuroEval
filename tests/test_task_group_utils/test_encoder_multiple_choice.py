"""Tests for encoder multiple-choice batching and prediction masking."""

from unittest.mock import MagicMock

import pytest
import torch

from euroeval.task_group_utils.encoder_multiple_choice import VariableChoiceCollator


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
