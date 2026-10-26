"""Public-interface tests for creating the MKQA multiple choice datasets."""

import re

import pytest
from datasets import DatasetDict

from scripts.dataset_creation import create_mkqa


class TestCreateDataset:
    """Tests for the `create_dataset` function."""

    def test_creation_is_deterministic(
        self, rows: list[create_mkqa.MkqaRow], dataset: DatasetDict
    ) -> None:
        """Creating the dataset twice gives the same samples.

        Args:
            rows:
                MKQA rows used to construct the dataset.
            dataset:
                Danish MKQA dataset fixture under test.
        """
        again = create_mkqa.create_dataset(
            rows=rows, language="da", train_size=10, val_size=5, test_size=20
        )
        assert again["test"].to_list() == dataset["test"].to_list()

    def test_date_distractors_have_the_same_format(self) -> None:
        """A date is never mixed with dates in another format."""
        years = [("1900", 1), ("1950", 2), ("1960", 3), ("1970", 4), ("1980", 5)]
        full_dates = [(f"20{day}-01-05", 10 + day) for day in range(10, 15)]
        month_year = [("juli 2004", 20)]
        rows = [
            _row(index=index, answer_type="date", answer_text=text)
            for text, index in years + full_dates + month_year
        ]
        dataset = create_mkqa.create_dataset(
            rows=rows, language="da", train_size=0, val_size=0, test_size=len(rows)
        )

        formats = {
            "year": re.compile(r"\d{4}"),
            "full date": re.compile(r"\d{4}-\d{2}-\d{2}"),
        }
        assert dataset["test"].num_rows == len(years) + len(full_dates)
        for sample in dataset["test"]:
            options = [option[3:] for option in sample["text"].split("\n")[2:]]
            kinds = {
                name
                for option in options
                for name, pattern in formats.items()
                if pattern.fullmatch(option)
            }
            assert len(kinds) == 1
            assert not any(option == "juli 2004" for option in options)

    def test_distractors_are_not_any_accepted_answer_or_alias(self) -> None:
        """No distractor matches an accepted answer or alias of the question."""
        rows = [
            _row(
                index=0,
                extra_danish_answers=[
                    create_mkqa.Answer(
                        type="entity", text="svar 1", aliases=["  SVAR   2"]
                    )
                ],
            ),
            *[_row(index=index) for index in range(1, 5)],
        ]
        dataset = create_mkqa.create_dataset(
            rows=rows, language="da", train_size=0, val_size=0, test_size=5
        )

        questions = {sample["text"].split("\n")[0] for sample in dataset["test"]}
        assert questions == {f"Spørgsmål nummer {index}?" for index in (1, 2, 3, 4)}
        for sample in dataset["test"]:
            question, _, *options = sample["text"].split("\n")
            number = question.removeprefix("Spørgsmål nummer ").removesuffix("?")
            assert f"svar {number}" in {option[3:] for option in options}
            assert len({option[3:] for option in options}) == 4

    def test_distractors_come_from_the_same_split(self, dataset: DatasetDict) -> None:
        """The options of a sample are answers to questions in the same split.

        Args:
            dataset:
                Danish MKQA dataset fixture under test.
        """
        for split in dataset.values():
            answers = {
                f"svar {sample['text'].split(chr(10))[0].split()[2].rstrip('?')}"
                for sample in split
            }
            for sample in split:
                options = {option[3:] for option in sample["text"].split("\n")[2:]}
                assert options <= answers

    def test_duplicate_questions_are_in_one_split_and_appear_once(self) -> None:
        """Two rows with the same question get the same split, and only one is kept."""
        rows = [_row(index=index) for index in range(100, 160)]
        rows += [
            _row(index=0, danish_question="Hvornår blev han valgt?"),
            _row(index=1, danish_question="hvornår  blev han valgt?"),
        ]

        danish = create_mkqa.create_dataset(
            rows=rows, language="da", train_size=1000, val_size=1000, test_size=1000
        )
        english = create_mkqa.create_dataset(
            rows=rows, language="en", train_size=1000, val_size=1000, test_size=1000
        )

        danish_questions = [
            sample["text"].split("\n")[0].casefold()
            for samples in danish.values()
            for sample in samples
        ]
        assert danish_questions.count("hvornår blev han valgt?") == 1
        english_splits = _split_by_question_number(dataset=english)
        assert english_splits["0"] == english_splits["1"]

    def test_four_distinct_options(self, dataset: DatasetDict) -> None:
        """Every sample has the options a to d, and they are all different.

        Args:
            dataset:
                Danish MKQA dataset fixture under test.
        """
        for split in dataset.values():
            for sample in split:
                options = sample["text"].split("\n")[2:]
                assert [option[:2] for option in options] == ["a.", "b.", "c.", "d."]
                assert len({option[3:] for option in options}) == 4

    def test_label_points_to_the_answer(self, dataset: DatasetDict) -> None:
        """The label is the letter of the option holding the answer.

        Args:
            dataset:
                Danish MKQA dataset fixture under test.
        """
        for split in dataset.values():
            for sample in split:
                question, header, *options = sample["text"].split("\n")
                assert header == "Svarmuligheder:"
                number = question.removeprefix("Spørgsmål nummer ").removesuffix("?")
                by_letter = {option[0]: option[3:] for option in options}
                assert by_letter[sample["label"]] == f"svar {number}"

    def test_number_with_unit_distractors_have_the_same_unit(self) -> None:
        """A number with a unit is never mixed with numbers in another unit."""
        rows = [
            *[
                _row(index=index, answer_type="number_with_unit", answer_text=text)
                for index, text in enumerate(
                    f"{number}.0 miles" for number in (10, 20, 30, 40, 50)
                )
            ],
            *[
                _row(index=index, answer_type="number_with_unit", answer_text=text)
                for index, text in enumerate(
                    (f"{number}.0 år" for number in (11, 21, 31, 41, 51)), start=10
                )
            ],
            _row(index=20, answer_type="number_with_unit", answer_text="5.0 kg"),
            _row(index=21, answer_type="number_with_unit", answer_text="3.5 6.9 miles"),
        ]
        dataset = create_mkqa.create_dataset(
            rows=rows, language="da", train_size=0, val_size=0, test_size=len(rows)
        )

        assert dataset["test"].num_rows == 10
        for sample in dataset["test"]:
            options = [option[3:] for option in sample["text"].split("\n")[2:]]
            assert len({option.split()[-1] for option in options}) == 1
            assert not any("kg" in option or "6.9" in option for option in options)

    def test_questions_linked_in_any_language_share_a_split(self) -> None:
        """Rows are grouped when they have an equal question in some language."""
        rows = [_row(index=index) for index in range(100, 160)]
        # Rows 0 and 1 have the same Danish question, and rows 1 and 2 the same English
        # question, so all three are the same question.
        rows += [
            _row(index=0, danish_question="Hvornår blev han valgt?"),
            _row(
                index=1,
                danish_question="hvornår  blev han valgt?",
                english_question="When was he picked?",
            ),
            _row(index=2, english_question="when was he PICKED?"),
        ]

        splits = {}
        for language in ("da", "en"):
            dataset = create_mkqa.create_dataset(
                rows=rows,
                language=language,
                train_size=1000,
                val_size=1000,
                test_size=1000,
            )
            splits[language] = {
                sample["text"].split("\n")[0].casefold(): split
                for split, samples in dataset.items()
                for sample in samples
            }

        assert (
            splits["da"]["hvornår blev han valgt?"]
            == splits["da"]["spørgsmål nummer 2?"]
            == splits["en"]["question number 0?"]
            == splits["en"]["when was he picked?"]
        )

    def test_same_questions_in_the_same_split_in_all_languages(
        self, rows: list[create_mkqa.MkqaRow], dataset: DatasetDict
    ) -> None:
        """A question is in the same split in Danish and English.

        Args:
            rows:
                MKQA rows used to construct datasets in both languages.
            dataset:
                Danish MKQA dataset fixture under test.
        """
        english = create_mkqa.create_dataset(
            rows=rows, language="en", train_size=10, val_size=5, test_size=20
        )
        for split in ("train", "val", "test"):
            danish_numbers = [
                sample["text"].split("\n")[0].split()[2].rstrip("?")
                for sample in dataset[split]
            ]
            english_numbers = [
                sample["text"].split("\n")[0].split()[2].rstrip("?")
                for sample in english[split]
            ]
            assert danish_numbers == english_numbers

    def test_split_does_not_depend_on_language_specific_filtering(self) -> None:
        """A question rejected in one language does not shift splits in the others."""
        rows = [
            _row(index=index, english_question="Short?" if index % 5 == 0 else None)
            for index in range(100)
        ]
        danish = create_mkqa.create_dataset(
            rows=rows, language="da", train_size=10, val_size=5, test_size=20
        )
        english = create_mkqa.create_dataset(
            rows=rows, language="en", train_size=10, val_size=5, test_size=20
        )

        danish_splits = _split_by_question_number(dataset=danish)
        english_splits = _split_by_question_number(dataset=english)

        shared = danish_splits.keys() & english_splits.keys()
        assert {danish_splits[number] for number in shared} == {"train", "val", "test"}
        for number in shared:
            assert danish_splits[number] == english_splits[number]
        assert not any(int(number) % 5 == 0 for number in english_splits)

    def test_splits_and_columns(self, dataset: DatasetDict) -> None:
        """The dataset has train, val and test splits with text and label.

        Args:
            dataset:
                Danish MKQA dataset fixture under test.
        """
        assert set(dataset.keys()) == {"train", "val", "test"}
        assert [dataset[split].num_rows for split in ("train", "val", "test")] == [
            10,
            5,
            20,
        ]
        for split in dataset.values():
            assert split.column_names == ["text", "label"]

    def test_splits_do_not_share_questions(self, dataset: DatasetDict) -> None:
        """No question is in more than one split.

        Args:
            dataset:
                Danish MKQA dataset fixture under test.
        """
        questions = [
            sample["text"].split("\n")[0]
            for split in dataset.values()
            for sample in split
        ]
        assert len(questions) == len(set(questions))

    def test_unusable_answer_types_are_left_out(self, dataset: DatasetDict) -> None:
        """Unanswerable, binary and long answers are not used.

        Args:
            dataset:
                Danish MKQA dataset fixture under test.
        """
        questions = [
            sample["text"].split("\n")[0]
            for split in dataset.values()
            for sample in split
        ]
        for index in (100, 101, 102):
            assert f"Spørgsmål nummer {index}?" not in questions


def _row(
    index: int,
    answer_type: str = "entity",
    answer_text: str | None = None,
    danish_question: str | None = None,
    english_question: str | None = None,
    extra_danish_answers: list[create_mkqa.Answer] | None = None,
) -> create_mkqa.MkqaRow:
    """Create an MKQA row with a Danish and an English question.

    Args:
        index:
            The index of the row, which makes the question and answer unique.
        answer_type (optional):
            The MKQA answer type of the answer. Defaults to "entity".
        answer_text (optional):
            The Danish answer, or None to use an answer that is unique to the row.
        danish_question (optional):
            The Danish question, or None to use a question that is unique to the row.
        english_question (optional):
            The English question, or None to use a question that is unique to the row.
        extra_danish_answers (optional):
            Further accepted Danish answers, after the first one.

    Returns:
        An MKQA row.
    """
    return create_mkqa.MkqaRow(
        example_id=index,
        queries=dict(
            da=danish_question or f"Spørgsmål nummer {index}?",
            en=english_question or f"Question number {index}?",
        ),
        answers=dict(
            da=[
                create_mkqa.Answer(
                    type=answer_type, text=answer_text or f"svar {index}", aliases=[]
                ),
                *(extra_danish_answers or []),
            ],
            en=[
                create_mkqa.Answer(type=answer_type, text=f"answer {index}", aliases=[])
            ],
        ),
    )


def _split_by_question_number(dataset: DatasetDict) -> dict[str, str]:
    """Map the number in each question to the split that the question is in.

    Args:
        dataset:
            A dataset created from rows made with `_row`.

    Returns:
        A dictionary mapping question numbers to split names.
    """
    return {
        sample["text"].split("\n")[0].split()[2].rstrip("?"): str(split)
        for split, samples in dataset.items()
        for sample in samples
    }


@pytest.fixture(scope="module")
def dataset(rows: list[create_mkqa.MkqaRow]) -> DatasetDict:
    """Create a small Danish dataset.

    Args:
        rows:
            MKQA rows.

    Returns:
        The Danish dataset with 10 / 5 / 20 samples.
    """
    return create_mkqa.create_dataset(
        rows=rows, language="da", train_size=10, val_size=5, test_size=20
    )


@pytest.fixture(scope="module")
def rows() -> list[create_mkqa.MkqaRow]:
    """Provide 60 usable rows plus rows that cannot be used.

    Returns:
        MKQA rows.
    """
    unusable = [
        _row(index=100, answer_type="unanswerable"),
        _row(index=101, answer_type="binary"),
        _row(index=102, answer_type="long_answer"),
    ]
    return [_row(index=index) for index in range(60)] + unusable
