"""Tests for scripts/create_scala.py."""

import re
from pathlib import Path

import pytest
from pytest import FixtureRequest

from scripts.create_scala import prepare_df
from scripts.load_ud_pos import load_ud_pos


@pytest.fixture(scope="module")
def scala_test_data_path(request: FixtureRequest) -> Path:
    """Path to the test data directory, relative to this test file.

    Args:
        request: The pytest fixture request.

    Returns:
        The path to the test data directory.
    """
    return Path(request.path).parent / "test_data"


@pytest.mark.parametrize(
    argnames=["filename", "wrong_words", "correct_word"],
    argvalues=(
        (
            "pl_pdb-ud-train.conllu.aux_clitic_01",
            ["postanowili", "śmy"],
            "postanowiliśmy",
        ),
        ("pl_pdb-ud-train.conllu.aux_clitic_02", ["nadział", "em"], "nadziałem"),
        ("pl_pdb-ud-train.conllu.aux_clitic_02", ["zarzucił", "em"], "zarzuciłem"),
        ("pl_pdb-ud-train.conllu.aux_clitic_03", ["chciał", "by", "m"], "chciałbym"),
        ("pl_pdb-ud-train.conllu.aux_clitic_03", ["aby", "ś"], "abyś"),
        ("de_gsd-ud-train.conllu.adp_det", ["in", "dem"], "im"),
        (
            "en_gum-ud-train.conllu.case",
            [re.compile(r"(^|\s)'($|\s)"), re.compile(r"\bGalois(\s|$)")],
            re.compile(r"\bGalois'"),
        ),
    ),
)
def test_load_ud_pos(
    filename: str,
    wrong_words: list[str | re.Pattern],
    correct_word: str | re.Pattern,
    scala_test_data_path: Path,
) -> None:
    """Test of the universal dependencies can be loaded correctly."""
    wrong_patterns = [
        wrong_word
        if isinstance(wrong_word, re.Pattern)
        else re.compile(rf"\b{wrong_word}\b", flags=re.IGNORECASE)
        for wrong_word in wrong_words
    ]
    correct_pattern = (
        correct_word
        if isinstance(correct_word, re.Pattern)
        else re.compile(rf"\b{correct_word}\b", flags=re.IGNORECASE)
    )

    dfs = load_ud_pos(
        str(scala_test_data_path / filename),
        str(scala_test_data_path / "empty.file"),
        str(scala_test_data_path / "empty.file"),
    )

    ds = prepare_df(df=dfs["train"], split="train")
    for row in ds:
        assert isinstance(row, dict), (
            f"Rows need to be dictionaries, but found {type(row)} for the row {row}."
        )
        if row["label"] == "incorrect":
            for wrong_pattern in wrong_patterns:
                assert wrong_pattern.search(row["text"]) is None, (
                    f"The wrong pattern {wrong_pattern!r} appeared in the text "
                    f"{row['text']!r}, which was not allowed."
                )
        else:
            assert correct_pattern.search(row["text"]) is not None, (
                f"The correct pattern {correct_pattern!r} did not appear in the text "
                f"{row['text']!r}, which is required."
            )
