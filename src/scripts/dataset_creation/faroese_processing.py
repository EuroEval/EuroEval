"""Clean and format generated Faroese multiple-choice datasets."""

import logging
import random
import typing as t

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

if __package__:
    from .constants import CHOICES_MAPPING
    from .dataset_processing import (
        format_multiple_choice_record,
        split_generated_dataset,
    )
    from .dataset_utils import upload_private
else:
    from constants import CHOICES_MAPPING
    from dataset_processing import (
        format_multiple_choice_record,
        split_generated_dataset,
    )
    from dataset_utils import upload_private

logger = logging.getLogger(__name__)
FaroeseTask = t.Literal["semantic-relations", "metaphorical-explanations"]


def create_faroese_dataset(source: str, task: FaroeseTask) -> None:
    """Load, prepare and upload Faroese data with conditional mini naming.

    Args:
        source:
            Path or URL of the original JSONL data.
        task:
            Faroese task identifying the input schema and output repository.
    """
    raw = load_dataset(path="json", data_files=source, split="train")
    assert isinstance(raw, Dataset)
    frame = raw.to_pandas()
    assert isinstance(frame, pd.DataFrame)
    dataset, truncated = prepare_faroese_dataset(frame=frame, task=task)
    repository_id = f"EuroEval/faroese-{task}"
    if truncated:
        repository_id += "-mini"
    upload_private(dataset=dataset, repository_id=repository_id)


def prepare_faroese_dataset(
    frame: pd.DataFrame, task: FaroeseTask
) -> tuple[DatasetDict, bool]:
    """Validate, clean, deduplicate, shuffle choices and split Faroese records.

    Args:
        frame:
            Original JSONL records for one Faroese task.
        task:
            Semantic relations or metaphorical explanations input schema.

    Returns:
        Prepared dataset and whether any proportional split exceeded its cap.

    Raises:
        ValueError:
            If required source columns are missing.
    """
    semantic = task == "semantic-relations"
    subject_column = "orð" if semantic else "example"
    answer_column = "andheiti" if semantic else "correct explanation"
    distractor_column = "random_outliers" if semantic else "distractor explanations"
    expected = {subject_column, answer_column, distractor_column}
    if not expected <= set(frame.columns):
        raise ValueError(f"Expected columns {expected}, got {set(frame.columns)}")
    count = 5 if semantic else 3
    letters = list("abcdef"[: count + 1])
    prompt = (
        "Hvat er andheitið hjá orðinum '{}'?"
        if semantic
        else "Hvat merkir orðafellið '{}'?"
    )
    rng = random.Random(4242)
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    skipped = 0
    for _, row in frame.iterrows():
        subject = _clean_text(value=row[subject_column])
        answer = _clean_text(value=row[answer_column])
        raw_distractors = row[distractor_column]
        if not isinstance(raw_distractors, (list, tuple, np.ndarray)):
            skipped += 1
            continue
        distractors = [
            _clean_text(
                value=value
                if semantic
                else (
                    value.get("distractor explanation")
                    if isinstance(value, dict)
                    else None
                )
            )
            for value in raw_distractors
        ]
        if (
            not subject
            or not answer
            or len(distractors) != count
            or not all(distractors)
        ):
            skipped += 1
            continue
        identity = (subject, answer, tuple(distractors))
        if identity in seen:
            continue
        seen.add(identity)
        records.append(
            format_multiple_choice_record(
                prompt=prompt.format(subject),
                correct_answer=answer,
                distractors=distractors,
                letters=letters,
                choices_label=CHOICES_MAPPING["fo"],
                rng=rng,
            )
        )
    if skipped:
        logger.warning(f"Skipped {skipped} malformed samples.")
    prepared = pd.DataFrame(records, columns=["text", "label"]).drop_duplicates()
    return split_generated_dataset(frame=prepared)


def _clean_text(value: object) -> str:
    """Normalise one textual field, rejecting null and non-text values.

    Args:
        value:
            Raw source field.

    Returns:
        Stripped single-line text, or an empty string for malformed fields.
    """
    return value.replace("\n", " ").strip() if isinstance(value, str) else ""
