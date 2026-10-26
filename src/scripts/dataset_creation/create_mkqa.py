# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "requests==2.32.5",
# ]
# ///

"""Create the MKQA multiple choice datasets and upload them to the HF Hub.

MKQA (https://github.com/apple/ml-mkqa) has the same 10,000 open-domain questions,
taken from Natural Questions, translated into 26 languages. Each question comes with a
short answer but no context, so we turn the questions into multiple choice knowledge
questions: the correct answer plus three answers to other questions of the same answer
type.
"""

import gzip
import io
import json
import logging
import random
import re
import typing as t
import unicodedata

import requests
from constants import (
    CHOICES_MAPPING,
    MAX_NUM_CHARS_IN_OPTION,
    MAX_NUM_CHARS_IN_QUESTION,
    MIN_NUM_CHARS_IN_OPTION,
    MIN_NUM_CHARS_IN_QUESTION,
)
from dataset_utils import MAX_TEST, MAX_TRAIN, MAX_VAL, SEED, upload_private
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

SOURCE_URL = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"

# The EuroEval languages that MKQA has translations for
LANGUAGES = [
    "da",
    "de",
    "en",
    "es",
    "fi",
    "fr",
    "hu",
    "it",
    "nl",
    "no",
    "pl",
    "pt",
    "sv",
]

# Answer types that make sense as multiple choice options. Binary answers have too few
# distinct options, long answers are sentences, and unanswerable questions have no
# answer.
ANSWER_TYPES = ["entity", "date", "number", "number_with_unit", "short_phrase"]

LETTERS = ["a", "b", "c", "d"]


def main() -> None:
    """Create the MKQA multiple choice datasets and upload them to the HF Hub."""
    rows = load_rows()
    for language in LANGUAGES:
        dataset = create_dataset(rows=rows, language=language)
        upload_private(dataset=dataset, repository_id=f"EuroEval/mkqa-{language}-mini")


def _normalise(text: str) -> str:
    """Normalise a question, answer or alias, so that they can be compared.

    Args:
        text:
            The question, answer or alias.

    Returns:
        The text with Unicode compatibility forms folded, whitespace collapsed and
        case removed.
    """
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


class Sample(t.TypedDict):
    """A usable MKQA question and its answer in one language."""

    example_id: str
    question: str
    answer: str
    accepted: set[str]
    match_key: str


def _build_split(
    samples: list[Sample], language: str, max_size: int
) -> tuple[Dataset, int]:
    """Build one split, with distractors taken from the questions in the split.

    The distractors of a question are answers to other kept questions that have the
    same match key, so that the format of an option, such as its unit, does not reveal
    the correct option. Questions with fewer than three such answers are left out, and
    the next questions take their place.

    Args:
        samples:
            The usable, unique questions and answers of the split, in order.
        language:
            The MKQA language code.
        max_size:
            The maximum number of samples in the split.

    Returns:
        A dataset with the columns "text" and "label", and the number of questions that
        were left out because they had too few comparable options.
    """
    kept = samples[:max_size]
    remaining = samples[max_size:]
    num_left_out = 0
    while True:
        pool_by_key: dict[str, list[str]] = {}
        for sample in kept:
            pool_by_key.setdefault(sample["match_key"], []).append(sample["answer"])
        candidates_by_id = {
            sample["example_id"]: sorted(
                {
                    _normalise(text=answer): answer
                    for answer in pool_by_key[sample["match_key"]]
                    if _normalise(text=answer) not in sample["accepted"]
                }.values()
            )
            for sample in kept
        }
        ids_without_options = {
            example_id
            for example_id, candidates in candidates_by_id.items()
            if len(candidates) < len(LETTERS) - 1
        }
        if not ids_without_options:
            break
        num_left_out += len(ids_without_options)
        kept = [
            sample for sample in kept if sample["example_id"] not in ids_without_options
        ]
        num_refills = max_size - len(kept)
        kept += remaining[:num_refills]
        remaining = remaining[num_refills:]

    texts: list[str] = []
    labels: list[str] = []
    for sample in kept:
        rng = random.Random(f"{SEED}-{language}-{sample['example_id']}")
        distractors = rng.sample(
            candidates_by_id[sample["example_id"]], k=len(LETTERS) - 1
        )
        options = [sample["answer"], *distractors]
        rng.shuffle(options)
        lines = [f"{letter}. {option}" for letter, option in zip(LETTERS, options)]
        texts.append(
            f"{sample['question']}\n{CHOICES_MAPPING[language]}:\n" + "\n".join(lines)
        )
        labels.append(LETTERS[options.index(sample["answer"])])
    return Dataset.from_dict(mapping=dict(text=texts, label=labels)), num_left_out


def _get_match_key(answer_type: str, answer: str) -> str:
    """Get the key that decides which answers can be distractors for each other.

    Args:
        answer_type:
            The MKQA answer type.
        answer:
            The answer text.

    Returns:
        The answer type, extended with the unit of a number with a unit (and how many
        numbers it has), or with the format of a date, such as a year or a full date.
    """
    text = _normalise(text=answer)
    if answer_type == "number_with_unit":
        tokens = text.split()
        num_numbers = 0
        while num_numbers < len(tokens) and re.fullmatch(
            r"[+-]?\d[\d.,]*", tokens[num_numbers]
        ):
            num_numbers += 1
        return f"{answer_type}:{num_numbers}:{' '.join(tokens[num_numbers:])}"
    if answer_type == "date":
        digits_as_nines = re.sub(r"\d+", lambda match: "9" * len(match.group()), text)
        date_format = re.sub(r"[^\W\d_]+", "A", digits_as_nines)
        return f"{answer_type}:{date_format}"
    return answer_type


def _remove_duplicate_questions(samples: list[Sample]) -> list[Sample]:
    """Keep only the first sample for each question.

    Args:
        samples:
            The samples of a split.

    Returns:
        The samples, with the later ones with a question that an earlier one has
        removed.
    """
    seen_questions: set[str] = set()
    unique_samples: list[Sample] = []
    for sample in samples:
        question = _normalise(text=sample["question"])
        if question not in seen_questions:
            seen_questions.add(question)
            unique_samples.append(sample)
    return unique_samples


class Answer(t.TypedDict):
    """An answer to an MKQA question in one language."""

    type: str
    text: str | None
    aliases: list[str] | None


class MkqaRow(t.TypedDict):
    """An MKQA question with its translations and answers in all languages."""

    example_id: int | str
    queries: dict[str, str]
    answers: dict[str, list[Answer]]


def create_dataset(
    rows: list[MkqaRow],
    language: str,
    train_size: int = MAX_TRAIN,
    val_size: int = MAX_VAL,
    test_size: int = MAX_TEST,
) -> DatasetDict:
    """Create the multiple choice dataset for one language.

    Args:
        rows:
            The MKQA rows, with the queries and answers for all languages.
        language:
            The MKQA language code, such as "da".
        train_size (optional):
            The maximum number of training samples. Defaults to MAX_TRAIN.
        val_size (optional):
            The maximum number of validation samples. Defaults to MAX_VAL.
        test_size (optional):
            The maximum number of test samples. Defaults to MAX_TEST.

    Returns:
        A dataset dictionary with the splits "train", "val" and "test", each with the
        columns "text" and "label".
    """
    groups = _group_rows(rows=rows)
    groups.sort(key=lambda group: min(str(row["example_id"]) for row in group))
    random.Random(SEED).shuffle(groups)

    total_size = train_size + val_size + test_size
    test_end = round(len(rows) * test_size / total_size)
    val_end = test_end + round(len(rows) * val_size / total_size)
    rows_by_split: dict[str, list[MkqaRow]] = dict(test=[], val=[], train=[])
    num_rows = 0
    for group in groups:
        if num_rows < test_end:
            rows_by_split["test"].extend(group)
        elif num_rows < val_end:
            rows_by_split["val"].extend(group)
        else:
            rows_by_split["train"].extend(group)
        num_rows += len(group)
    max_size_by_split = dict(train=train_size, val=val_size, test=test_size)

    splits: dict[str, Dataset] = dict()
    for split, split_rows in rows_by_split.items():
        samples = [
            sample
            for sample in (
                _get_sample(row=row, language=language) for row in split_rows
            )
            if sample is not None
        ]
        unique_samples = _remove_duplicate_questions(samples=samples)
        dataset, num_without_options = _build_split(
            samples=unique_samples, language=language, max_size=max_size_by_split[split]
        )
        logger.info(
            f"mkqa-{language}, {split}: kept {dataset.num_rows} samples, left out "
            f"{len(samples) - len(unique_samples)} duplicate questions and "
            f"{num_without_options} questions without enough comparable options"
        )
        splits[split] = dataset
    return DatasetDict({split: dataset for split, dataset in splits.items()})


def _get_sample(row: MkqaRow, language: str) -> Sample | None:
    """Get the question and answer of a row, if they are usable.

    Args:
        row:
            An MKQA row.
        language:
            The MKQA language code.

    Returns:
        The example ID, question, answer, the key that decides which answers can be
        distractors for it, and the normalised texts of all accepted answers and their
        aliases, or None if the row cannot be used as a multiple choice question. The
        first answer is the one that is used as the correct option.
    """
    question = row["queries"][language].strip()
    answers = row["answers"][language]
    if not answers or answers[0]["type"] not in ANSWER_TYPES:
        return None
    answer = (answers[0]["text"] or "").strip()
    if not MIN_NUM_CHARS_IN_QUESTION <= len(question) <= MAX_NUM_CHARS_IN_QUESTION:
        return None
    if not MIN_NUM_CHARS_IN_OPTION <= len(answer) <= MAX_NUM_CHARS_IN_OPTION:
        return None
    return Sample(
        example_id=str(row["example_id"]),
        question=question,
        answer=answer,
        accepted={
            _normalise(text=text)
            for accepted_answer in answers
            for text in [
                accepted_answer["text"],
                *(accepted_answer.get("aliases") or []),
            ]
            if isinstance(text, str)
        },
        match_key=_get_match_key(answer_type=answers[0]["type"], answer=answer),
    )


def _group_rows(rows: list[MkqaRow]) -> list[list[MkqaRow]]:
    """Group the rows that are the same question.

    Two rows are in the same group if they have the same question in at least one
    language, directly or through other rows. MKQA has such duplicates with different
    example IDs, and keeping a group together puts all of them in the same split in all
    languages.

    Args:
        rows:
            The MKQA rows.

    Returns:
        The groups of rows, where the rows of a group are sorted by example ID.
    """
    parents = list(range(len(rows)))

    def find_root(index: int) -> int:
        """Find the root of the group that a row is in.

        Args:
            index:
                The index of the row.

        Returns:
            The index of the row that is the root of the group.
        """
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    row_with_question: dict[tuple[str, str], int] = dict()
    for index, row in enumerate(rows):
        for language, query in row["queries"].items():
            question = _normalise(text=query)
            if not question:
                continue
            other_index = row_with_question.setdefault((language, question), index)
            parents[find_root(index=index)] = find_root(index=other_index)

    groups: dict[int, list[MkqaRow]] = dict()
    for index, row in enumerate(rows):
        groups.setdefault(find_root(index=index), []).append(row)
    return [
        sorted(group, key=lambda row: str(row["example_id"]))
        for group in groups.values()
    ]


def load_rows() -> list[MkqaRow]:
    """Download the MKQA questions and answers.

    Returns:
        The MKQA rows, with the queries and answers for all languages.
    """
    response = requests.get(url=SOURCE_URL, timeout=300)
    response.raise_for_status()
    with gzip.open(io.BytesIO(response.content), mode="rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


if __name__ == "__main__":
    main()
