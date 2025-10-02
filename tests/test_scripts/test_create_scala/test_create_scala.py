import re
from pathlib import Path
from re import Pattern

import pytest
from pytest import FixtureRequest

from scripts.create_scala import prepare_df
from scripts.load_ud_pos import load_ud_pos


@pytest.fixture
def tst_data_path(request: FixtureRequest) -> Path:
    return Path(request.fspath).parent / "test_data"


def _wb(s: str) -> Pattern:
    return re.compile(rf"\b{s}\b", re.I)


def _test_load_ud_pos(
    tst_data_path: Path, data_path: str, nay: list[Pattern], yay: Pattern
) -> None:
    df = load_ud_pos(
        str(tst_data_path / data_path),
        str(tst_data_path / "empty.file"),
        str(tst_data_path / "empty.file"),
    )

    ds = prepare_df(df["train"], "train")
    for row in ds:
        if row["label"] == "incorrect":
            for rx in nay:
                assert rx.search(row["text"]) is None
        else:
            assert yay.search(row["text"]) is not None


def test_load_ud_pos_pl_aux_clitic_01(tst_data_path: Path) -> None:
    _test_load_ud_pos(
        tst_data_path,
        "pl_pdb-ud-train.conllu.aux_clitic_01",
        [_wb("postanowili"), _wb("śmy")],
        _wb("postanowiliśmy"),
    )


def test_load_ud_pos_pl_aux_clitic_02(tst_data_path: Path) -> None:
    _test_load_ud_pos(
        tst_data_path,
        "pl_pdb-ud-train.conllu.aux_clitic_02",
        [_wb("nadział"), _wb("em")],
        _wb("nadziałem"),
    )


def test_load_ud_pos_pl_aux_clitic_03(tst_data_path: Path) -> None:
    _test_load_ud_pos(
        tst_data_path,
        "pl_pdb-ud-train.conllu.aux_clitic_02",
        [_wb("zarzucił"), _wb("em")],
        _wb("zarzuciłem"),
    )


def test_load_ud_pos_pl_aux_clitic_04(tst_data_path: Path) -> None:
    _test_load_ud_pos(
        tst_data_path,
        "pl_pdb-ud-train.conllu.aux_clitic_03",
        [_wb("chciał"), _wb("by"), _wb("m")],
        _wb("chciałbym"),
    )


def test_load_ud_pos_pl_aux_clitic_05(tst_data_path: Path) -> None:
    _test_load_ud_pos(
        tst_data_path,
        "pl_pdb-ud-train.conllu.aux_clitic_03",
        [_wb("aby"), _wb("ś")],
        _wb("abyś"),
    )


def test_load_ud_pos_de_adp_det(tst_data_path: Path) -> None:
    _test_load_ud_pos(
        tst_data_path,
        "de_gsd-ud-train.conllu.adp_det",
        [_wb("in"), _wb("dem")],
        _wb("im"),
    )


def test_load_ud_pos_en_case(tst_data_path: Path) -> None:
    _test_load_ud_pos(
        tst_data_path,
        "en_gum-ud-train.conllu.case",
        [re.compile(r"(^|\s)'($|\s)"), re.compile(r"\bGalois(\s|$)")],
        re.compile(r"\bGalois'"),
    )
