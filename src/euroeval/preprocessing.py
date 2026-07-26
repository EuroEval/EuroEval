"""Preprocessing utilities for custom dataset column mapping."""

import collections.abc as c
import functools
import typing as t

from datasets import Dataset, DatasetDict

from .enums import TaskGroup
from .exceptions import InvalidBenchmark
from .string_utils import CHOICE_LETTERS

if t.TYPE_CHECKING:
    pass


def merge_input_and_choices(
    example: dict,
    input_column: str,
    choices_column: "str | list[str]",
    choices_label: str,
) -> dict:
    """Merge input text and choices into a single text field.

    Args:
        example:
            A single dataset example with at least the ``input_column`` and
            the column(s) named by ``choices_column``.
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
        choices. The original bare input text and raw choice list are also preserved
        as ``"bare_input"`` and ``"raw_choices"`` to support BPC scoring that needs
        bare questions without enumerated choices.
    """
    input_text = example[input_column].replace("\n", " ").strip()
    if isinstance(choices_column, list):
        choices = [example[col] for col in choices_column]
    else:
        choices = example[choices_column]
    cleaned_choices = [choice.replace("\n", " ").strip() for choice in choices]
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(CHOICE_LETTERS, cleaned_choices)
    )
    example["text"] = f"{input_text}\n{choices_label}:\n{options}"
    example["bare_input"] = input_text
    example["raw_choices"] = cleaned_choices
    return example


def _get_standard_target_column(
    task_group: TaskGroup, target_column: str | None
) -> str | None:
    """Determine the standard target column name for a task group.

    Args:
        task_group:
            The task group to determine the standard for.
        target_column:
            The target column name, or None if not applicable.

    Returns:
        The standard target column name, or None if target_column is None.
    """
    if target_column is None:
        return None
    if task_group == TaskGroup.TOKEN_CLASSIFICATION:
        return "labels"
    if task_group == TaskGroup.TEXT_TO_TEXT:
        return "target_text"
    return "label"


def _validate_columns(
    dataset: DatasetDict,
    dataset_name: str,
    input_column: str,
    choices_cols: list[str] | None,
    target_column: str | None,
) -> None:
    """Validate that configured columns exist in all splits.

    Args:
        dataset:
            The dataset to validate.
        dataset_name:
            The name of the dataset, used in error messages.
        input_column:
            The input column to validate.
        choices_cols:
            List of choices columns to validate, or None.
        target_column:
            The target column to validate, or None.

    Raises:
        InvalidBenchmark:
            If a configured column is absent from all splits.
    """
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
            col_found = all(col in split.column_names for split in dataset.values())
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


def _fix_mc_label_column(
    example: dict,
    choices_column: "str | list[str]",
    target_column: str,
) -> dict:
    """Ensure multiple choice labels are lowercase letters.

    Args:
        example:
            The example to fix.
        choices_column:
            Either the name of a single column containing a list of answer-choice
            strings, or a list of column names each containing a single answer-choice
            string.
        target_column:
            The target column containing the label to fix.

    Returns:
        The fixed example.
    """
    if isinstance(choices_column, list):
        choices = [example[col] for col in choices_column]
    else:
        choices = example[choices_column]
    label = example[target_column]
    if label in choices:
        example[target_column] = CHOICE_LETTERS[choices.index(label)]
    if isinstance(example[target_column], int):
        example[target_column] = CHOICE_LETTERS[label]
    if isinstance(example[target_column], str):
        example[target_column] = example[target_column].lower()
    return example


def _handle_choices_columns(
    split: Dataset,
    input_column: str,
    choices_column: "str | list[str]",
    choices_cols: list[str],
    choices_label: str,
    target_column: str | None,
) -> Dataset:
    """Handle splitting when choices columns are present.

    Args:
        split:
            The dataset split to process.
        input_column:
            The input column name.
        choices_column:
            The choices column specification.
        choices_cols:
            List of choices column names.
        choices_label:
            The label for the choices section.
        target_column:
            The target column name, or None.

    Returns:
        The processed dataset split.
    """
    if target_column is not None:
        fix_fn = functools.partial(
            _fix_mc_label_column,
            choices_column=choices_column,
            target_column=target_column,
        )
        split = split.map(fix_fn)

    merge_fn = functools.partial(
        merge_input_and_choices,
        input_column=input_column,
        choices_column=choices_column,
        choices_label=choices_label,
    )
    split = split.map(merge_fn)

    preserved = {"text", "bare_input", "raw_choices"}
    cols_to_drop = [
        col
        for col in [input_column, *choices_cols]
        if col in split.column_names and col not in preserved
    ]
    if cols_to_drop:
        split = split.remove_columns(cols_to_drop)
    return split


def _handle_target_column_rename(
    split: Dataset, std_target: str | None, target_column: str | None
) -> Dataset:
    """Handle target column renaming if needed.

    Args:
        split:
            The dataset split to process.
        std_target:
            The standard target column name, or None.
        target_column:
            The original target column name, or None.

    Returns:
        The processed dataset split.
    """
    if (
        std_target is not None
        and target_column is not None
        and target_column != std_target
    ):
        if std_target in split.column_names:
            split = split.remove_columns([std_target])
        split = split.rename_column(target_column, std_target)
    return split


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
    """
    std_target = _get_standard_target_column(task_group, target_column)

    # Normalize choices_column to a list for uniform handling
    choices_cols: list[str] | None
    if isinstance(choices_column, list):
        choices_cols = choices_column
    elif choices_column is not None:
        choices_cols = [choices_column]
    else:
        choices_cols = None

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
        """
        _validate_columns(
            dataset=dataset,
            dataset_name=dataset_name,
            input_column=input_column,
            choices_cols=choices_cols,
            target_column=target_column,
        )

        for split_name, split in dataset.items():
            if choices_cols is not None:
                assert choices_column is not None
                split = _handle_choices_columns(
                    split=split,
                    input_column=input_column,
                    choices_column=choices_column,
                    choices_cols=choices_cols,
                    choices_label=choices_label,
                    target_column=target_column,
                )
            elif input_column != "text":
                if "text" in split.column_names:
                    split = split.remove_columns(["text"])
                split = split.rename_column(input_column, "text")

            split = _handle_target_column_rename(
                split=split, std_target=std_target, target_column=target_column
            )

            dataset[split_name] = split

        return dataset

    return preprocessing_func
