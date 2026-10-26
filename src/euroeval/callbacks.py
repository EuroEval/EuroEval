"""Callbacks for the Hugging Face Trainer."""

import collections.abc as c
import sys
import typing as t

from tqdm.auto import tqdm
from transformers.trainer_callback import ProgressCallback

from .logging_utils import get_pbar

if t.TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformers.trainer_callback import TrainerControl, TrainerState
    from transformers.training_args import TrainingArguments


class NeverLeaveProgressCallback(ProgressCallback):
    """Progress callback which never leaves the progress bar."""

    def __init__(self, max_str_len: int = 100) -> None:
        """Initialise the callback.

        Args:
            max_str_len (optional):
                Maximum length of the progress bar description. Defaults to 100.
        """
        super().__init__(max_str_len=max_str_len)
        self.training_bar: tqdm | None = None
        self.prediction_bar: tqdm | None = None

    def on_prediction_step(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        eval_dataloader: "DataLoader | None" = None,
        **kwargs: str,
    ) -> None:
        """Update the progress bar after a prediction step.

        Args:
            args:
                The training arguments.
            state:
                The current trainer state.
            control:
                The trainer control object.
            eval_dataloader (optional):
                The evaluation data loader, if available. Defaults to None.
            **kwargs:
                Additional callback keyword arguments.
        """
        if eval_dataloader is None:
            return
        correct_dtype = isinstance(eval_dataloader.dataset, c.Sized)
        if state.is_local_process_zero and correct_dtype:
            if self.prediction_bar is None:
                desc = "Evaluating model"
                self.prediction_bar = get_pbar(
                    total=len(eval_dataloader),
                    desc=desc,
                    disable=hasattr(sys, "_called_from_test"),
                )
            self.prediction_bar.update(1)

    def on_step_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs: str,
    ) -> None:
        """Update the progress bar after a training step.

        Args:
            args:
                The training arguments.
            state:
                The current trainer state.
            control:
                The trainer control object.
            **kwargs:
                Additional callback keyword arguments.
        """
        if state.is_local_process_zero and self.training_bar is not None:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs: str,
    ) -> None:
        """Create the training progress bar when training begins.

        Args:
            args:
                The training arguments.
            state:
                The current trainer state.
            control:
                The trainer control object.
            **kwargs:
                Additional callback keyword arguments.
        """
        if state.is_local_process_zero:
            desc = "Finetuning model"
            self.training_bar = get_pbar(
                total=None, desc=desc, disable=hasattr(sys, "_called_from_test")
            )
        self.current_step = 0
