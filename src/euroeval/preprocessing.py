"""Preprocessing utilities for custom dataset column mapping."""

import collections.abc as c
import functools
import typing as t

from .enums import TaskGroup
from .exceptions import InvalidBenchmark

if t.TYPE_CHECKING:
    from datasets import DatasetDict


def merge_input_and_choices(
    example: dict,
    input_column: str,
    choices_column: "str | list[str]",
    choices_label: str,
) -> dict:
    """Merge input text and choices into a single text field.

    Args:
        example:
            A single dataset example with at least the ``input_column`` and the column(s)
            named by ``choices_column``.
        input_column:
            The name of the column containing the input text.
        choices_column:
            Either the name of a single column containing a list of answer-choice
            strings, or a list of column names each containing a single answer-choice
            string.
        choices_label:
            The language-specific label for the choices section (e.g. ``"Choices"``).

    Returns:
        The example with a new ``"text"`` key containing the merged input and formatted
        choices.
    """
    input_text = example[input_column].replace("\n", " ").strip()
    if isinstance(choices_column, list):
        choices = [example[col] for col in choices_column]
    else:
        choices = example[choices_column]
    options = "\n".join(
        f"{letter}. {choice.replace('\n', ' ').strip()}"
        for letter, choice in zip("abcdefghijklmnopqrstuvwxyz", choices)
    )
    example["text"] = f"{input_text}\n{choices_label}:\n{options}"
    return example


def build_preprocessing_func(
    dataset_name: str,
    task_group: "TaskGroup",
    input_column: str,
    target_column: str | None,
    choices_column: "str | list[str] | None",
    choices_label: str,
) -> "c.Callable[[DatasetDict], DatasetDict]":
    """Build a preprocessing function from column mapping arguments.

    The returned function renames or merges columns in a DatasetDict to match the
    framework's standard column names:

    - If ``input_column`` differs from ``"text"`` (without ``choices_column``), it is
      renamed to ``"text"``.
    - If ``choices_column`` is given, ``input_column`` and ``choices_column`` are merged
      into a single ``"text"`` column formatted as::

          <input_text>
          <choices_label>:
          a. <choice_0>
          b. <choice_1>
          ...

    - If ``target_column`` is given, it is renamed to the task-group standard:
      ``"labels"`` for token classification, ``"target_text"`` for text-to-text, and
      ``"label"`` for everything else.

    Args:
        dataset_name:
            The name of the dataset, used in error messages.
        task_group:
            The task group, used to determine the standard target column name.
        input_column:
            Column to rename to ``"text"``. When combined with ``choices_column``, the
            two are merged into a formatted ``"text"`` column instead. Defaults to
            ``"text"`` (no rename).
        target_column:
            Column to rename to the task-appropriate standard target column name.
        choices_column:
            Either the name of a single column containing a list of answer-choice
            strings, or a list of column names each containing a single answer-choice
            string, to merge with the input column.
        choices_label:
            The language-specific label for the choices section (e.g. ``"Choices"``).

    Returns:
        A callable that accepts a ``DatasetDict`` and returns a preprocessed
        ``DatasetDict``.

    Raises:
        InvalidBenchmark:
            When the returned callable is called, if any configured column is absent
            from all splits.
    """
    # Determine the standard target column for the task group
    if target_column is not None:
        if task_group == TaskGroup.TOKEN_CLASSIFICATION:
            std_target = "labels"
        elif task_group == TaskGroup.TEXT_TO_TEXT:
            std_target = "target_text"
        else:
            std_target = "label"
    else:
        std_target = None

    def preprocessing_func(dataset: "DatasetDict") -> "DatasetDict":
        """Apply column mapping and merging to all splits in the dataset.

        Validates that configured columns exist in all splits before processing, then
        renames or merges columns according to the configuration passed to
        :func:`build_preprocessing_func`.

        Args:
            dataset:
                The dataset to preprocess.

        Returns:
            The preprocessed dataset with columns renamed or merged as configured.

        Raises:
            InvalidBenchmark:
                If a configured input or target column is absent from all splits.
        """
        # Normalize choices_column to a list for uniform handling
        if isinstance(choices_column, list):
            choices_cols: list[str] | None = choices_column
        elif choices_column is not None:
            choices_cols = [choices_column]
        else:
            choices_cols = None

        # Validate that the configured columns exist in all splits
        if input_column != "text":
            input_found = all(
                input_column in split.column_names for split in dataset.values()
            )
            if not input_found:
                raise InvalidBenchmark(
                    f"The dataset is configured with an input column "
                    f"{input_column!r}, but this column was not found in all splits "
                    f"for the dataset {dataset_name!r}."
                )
        if choices_cols is not None:
            for col in choices_cols:
                col_found = all(
                    col in split.column_names for split in dataset.values()
                )
                if not col_found:
                    raise InvalidBenchmark(
                        f"The dataset is configured with a choices column "
                        f"{col!r}, but this column was not found in all splits "
                        f"for the dataset {dataset_name!r}."
                    )
        if target_column is not None:
            target_found = all(
                target_column in split.column_names for split in dataset.values()
            )
            if not target_found:
                raise InvalidBenchmark(
                    f"The dataset is configured with a target column "
                    f"{target_column!r}, but this column was not found in all splits "
                    f"for the dataset {dataset_name!r}."
                )

        for split_name, split in dataset.items():
            # Handle input column (optionally merging with choices)
            if choices_cols is not None:
                merge_fn = functools.partial(
                    merge_input_and_choices,
                    input_column=input_column,
                    choices_column=choices_column,
                    choices_label=choices_label,
                )
                split = split.map(merge_fn)
                cols_to_drop = [
                    col
                    for col in [input_column, *choices_cols]
                    if col in split.column_names and col != "text"
                ]
                if cols_to_drop:
                    split = split.remove_columns(cols_to_drop)
            elif input_column != "text":
                if "text" in split.column_names:
                    split = split.remove_columns(["text"])
                split = split.rename_column(input_column, "text")

            # Handle target column renaming
            if (
                std_target is not None
                and target_column is not None
                and target_column != std_target
            ):
                if std_target in split.column_names:
                    split = split.remove_columns([std_target])
                split = split.rename_column(target_column, std_target)

            dataset[split_name] = split

        return dataset

    return preprocessing_func
