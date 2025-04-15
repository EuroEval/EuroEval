"""Unit tests for the `callbacks` module."""

from dataclasses import dataclass
from typing import Generator

import pytest
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from euroeval.callbacks import NeverLeaveProgressCallback


@dataclass
class FakeState(TrainerState):
    """Dummy state class for testing."""

    is_local_process_zero: bool = True


@dataclass
class FakeEvalDataloader(DataLoader):
    """Dummy evaluation dataloader class for testing."""

    dataset: Dataset = Dataset.from_dict({"value": [1, 2, 3]})

    def __len__(self) -> int:
        """Return the length of the dataloader."""
        return len(self.dataset)


class TestNeverLeaveProgressCallback:
    """Tests for the `NeverLeaveProgressCallback` class."""

    @pytest.fixture(scope="class")
    def callback(self) -> Generator[NeverLeaveProgressCallback, None, None]:
        """Yields a callback."""
        yield NeverLeaveProgressCallback()

    @pytest.fixture(scope="class")
    def state(self) -> Generator[FakeState, None, None]:
        """Yields a state."""
        yield FakeState(is_local_process_zero=True)

    @pytest.fixture(scope="class")
    def eval_dataloader(self) -> Generator[FakeEvalDataloader, None, None]:
        """Yields an evaluation dataloader."""
        yield FakeEvalDataloader()

    def test_on_train_begin_initialises_not_leaving_pbar(
        self, callback: NeverLeaveProgressCallback, state: FakeState
    ) -> None:
        """Test that the `leave` attribute on the training progress bar is False."""
        assert callback.training_bar is None
        callback.on_train_begin(
            args=TrainingArguments(), state=state, control=TrainerControl()
        )
        assert callback.training_bar is not None
        assert not callback.training_bar.leave

    def test_on_prediction_step_initialises_not_leaving_pbar(
        self,
        callback: NeverLeaveProgressCallback,
        state: FakeState,
        eval_dataloader: FakeEvalDataloader,
    ) -> None:
        """Test that the `leave` attribute on the prediction progress bar is False."""
        assert callback.prediction_bar is None
        callback.on_prediction_step(
            args=TrainingArguments(),
            state=state,
            control=TrainerControl(),
            eval_dataloader=eval_dataloader,
        )
        assert callback.prediction_bar is not None
        assert not callback.prediction_bar.leave
