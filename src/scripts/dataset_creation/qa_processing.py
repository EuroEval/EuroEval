"""Shared answer validation and split-preserving question-answering preparation."""

import typing as t

import pandas as pd
from datasets import Dataset, DatasetDict

if __package__:
    from .constants import (
        MAX_NUM_CHARS_IN_CONTEXT,
        MAX_NUM_CHARS_IN_QUESTION,
        MIN_NUM_CHARS_IN_CONTEXT,
        MIN_NUM_CHARS_IN_QUESTION,
    )
    from .dataset_processing import (
        STANDARD_LIMITS,
        dataset_dict_from_frames,
        filter_by_character_lengths,
        process_source_splits,
        sample_dataframe,
    )
else:
    from constants import (
        MAX_NUM_CHARS_IN_CONTEXT,
        MAX_NUM_CHARS_IN_QUESTION,
        MIN_NUM_CHARS_IN_CONTEXT,
        MIN_NUM_CHARS_IN_QUESTION,
    )
    from dataset_processing import (
        STANDARD_LIMITS,
        dataset_dict_from_frames,
        filter_by_character_lengths,
        process_source_splits,
        sample_dataframe,
    )


class Answer(t.TypedDict):
    """One answer in a raw SQuAD article."""

    text: str
    answer_start: int


class Answers(t.TypedDict):
    """Parallel answer texts and character offsets for a processed question."""

    text: list[str]
    answer_start: list[int]


class Question(t.TypedDict):
    """Question and candidate answers in a raw SQuAD paragraph."""

    id: str
    question: str
    answers: list[Answer]


class Paragraph(t.TypedDict):
    """SQuAD passage with one or more questions."""

    context: str
    qas: list[Question]


class Article(t.TypedDict):
    """Article containing SQuAD paragraphs."""

    paragraphs: list[Paragraph]


def prepare_squad_dataset(
    source: DatasetDict,
    source_splits: dict[str, str],
    answer_mode: t.Literal["provided", "infer", "first"] = "provided",
    limits: dict[str, int] | None = None,
) -> DatasetDict:
    """Filter, validate, partition and cap SQuAD-style source splits.

    A shared heldout source may supply val and test, but training is never pooled
    with heldout data. Partition exclusion uses stable row indices, even after
    filtering. Indices are discarded only when the output datasets are built.

    Args:
        source:
            Original source dataset dictionary.
        source_splits:
            Output train/val/test names mapped to source split names. Only val
            and test may share a source.
        answer_mode (optional):
            Validate provided offsets, infer missing offsets from context, or
            retain only the first heldout answer (Italian SQuAD). Training keeps
            all valid answers. Defaults to provided.
        limits (optional):
            Per-output caps. Defaults to 1024 train, 256 val and 2048 test.

    Returns:
        Dataset with valid answer spans and disjoint source partitions.

    Raises:
        ValueError:
            If the split mapping is incomplete or training shares heldout data.
    """
    limits = STANDARD_LIMITS if limits is None else limits
    if source_splits.keys() != STANDARD_LIMITS.keys():
        raise ValueError("SQuAD requires train, val and test mappings")
    if source_splits["train"] in (source_splits["val"], source_splits["test"]):
        raise ValueError("Training cannot share a source with heldout splits")
    frames = {}
    for name in dict.fromkeys(source_splits.values()):
        mode = (
            "provided"
            if answer_mode == "first" and name == source_splits["train"]
            else answer_mode
        )
        frame = source[name].to_pandas().reset_index(drop=True)
        if "id" in frame.columns:
            frame["id"] = frame["id"].astype(str)
            frame = frame.drop_duplicates(subset="id")
        frame = filter_by_character_lengths(
            frame=frame,
            bounds={
                "context": (MIN_NUM_CHARS_IN_CONTEXT, MAX_NUM_CHARS_IN_CONTEXT),
                "question": (MIN_NUM_CHARS_IN_QUESTION, MAX_NUM_CHARS_IN_QUESTION),
            },
        )
        frame["answers"] = pd.Series(
            [
                _validate_answers(
                    context=row["context"], answers=row["answers"], mode=mode
                )
                for _, row in frame.iterrows()
            ],
            index=frame.index,
            dtype=object,
        )
        frames[name] = frame.loc[
            frame["answers"].map(lambda value: bool(value["text"]))
        ]
    output = {name: frames[original] for name, original in source_splits.items()}
    if source_splits["val"] == source_splits["test"]:
        heldout = output["val"]
        output["val"] = sample_dataframe(frame=heldout, maximum=limits["val"])
        output["test"] = heldout.loc[~heldout.index.isin(output["val"].index)]
    capped = process_source_splits(frames=output, limits=limits)
    return dataset_dict_from_frames(
        frames=capped, columns=list(capped["train"].columns)
    )


def _validate_answers(context: str, answers: Answers, mode: str) -> Answers:
    """Retain non-empty answers whose offsets identify the context span.

    Args:
        context:
            Passage containing the answers.
        answers:
            Candidate texts and possibly missing offsets.
        mode:
            Provided offsets, inferred offsets, or first-answer-only selection.

    Returns:
        Validated parallel answer texts and offsets.
    """
    texts = list(answers["text"])
    offsets = list(answers.get("answer_start", []))
    if mode == "first":
        texts, offsets = texts[:1], offsets[:1]
    valid: Answers = {"text": [], "answer_start": []}
    for index, text in enumerate(texts):
        if not text or (mode == "first" and len(text) <= 1):
            continue
        start = (
            context.find(text)
            if mode == "infer"
            else (offsets[index] if index < len(offsets) else -1)
        )
        if start >= 0 and context[start : start + len(text)] == text:
            valid["text"].append(text)
            valid["answer_start"].append(int(start))
    return valid


def squad_articles_to_dataset(articles: list[Article]) -> Dataset:
    """Flatten every paragraph and question from raw SQuAD articles.

    Args:
        articles:
            Raw articles from one source split, never pooled across splits.

    Returns:
        Dataset of contexts, questions, IDs and parallel answer lists.
    """
    records = [
        {
            "id": str(question["id"]),
            "context": paragraph["context"],
            "question": question["question"],
            "answers": {
                "text": [answer["text"] for answer in question["answers"]],
                "answer_start": [
                    answer["answer_start"] for answer in question["answers"]
                ],
            },
        }
        for article in articles
        for paragraph in article["paragraphs"]
        for question in paragraph["qas"]
    ]
    return Dataset.from_pandas(
        pd.DataFrame(records, columns=["id", "context", "question", "answers"])
    )
