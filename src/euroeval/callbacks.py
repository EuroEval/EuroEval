"""Callbacks for the Hugging Face Trainer."""

import sys
from collections.abc import Sized

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.trainer_callback import ProgressCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class NeverLeaveProgressCallback(ProgressCallback):
    """Progress callback which never leaves the progress bar."""

    def __init__(self, max_str_len: int = 100) -> None:
        """Initialise the callback."""
        super().__init__(max_str_len=max_str_len)
        self.training_bar: tqdm | None = None
        self.prediction_bar: tqdm | None = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: str,
    ) -> None:
        """Callback actions when training begins."""
        if state.is_local_process_zero:
            desc = "Finetuning model"
            self.training_bar = tqdm(
                total=None,
                leave=False,
                desc=desc,
                disable=hasattr(sys, "_called_from_test"),
            )
        self.current_step = 0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: str,
    ) -> None:
        """Callback actions when a training step ends."""
        if state.is_local_process_zero and self.training_bar is not None:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_prediction_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        eval_dataloader: DataLoader | None = None,
        **kwargs: str,
    ) -> None:
        """Callback actions when a prediction step ends."""
        if eval_dataloader is None:
            return
        correct_dtype = isinstance(eval_dataloader.dataset, Sized)
        if state.is_local_process_zero and correct_dtype:
            if self.prediction_bar is None:
                desc = "Evaluating model"
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader),
                    leave=False,
                    desc=desc,
                    disable=hasattr(sys, "_called_from_test"),
                )
            self.prediction_bar.update(1)
