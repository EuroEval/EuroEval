import re
from pathlib import Path

import pytest
from pytest import FixtureRequest

from scripts.create_scala import prepare_df
from scripts.load_ud_pos import load_ud_pos


@pytest.fixture
def tst_data_path(request: FixtureRequest) -> Path:
    return Path(request.fspath).parent / "test_data"


def _test_load_ud_pos(
    tst_data_path: Path, data_path: str, nay: list[str], yay: str
) -> None:
    rx_nay = [re.compile(rf"\b{n}\b", re.I) for n in nay]
    rx_yay = re.compile(rf"\b{yay}\b", re.I)

    df = load_ud_pos(
        str(tst_data_path / data_path),
        str(tst_data_path / "empty.file"),
        str(tst_data_path / "empty.file"),
    )

    ds = prepare_df(df["train"], "train")
    for row in ds:
        if row["label"] == "incorrect":
            for rx in rx_nay:
                assert rx.search(row["text"]) is None
        else:
            assert rx_yay.search(row["text"]) is not None


def test_load_ud_pos_pl_aux_clitic(tst_data_path: Path) -> None:
    _test_load_ud_pos(
        tst_data_path,
        "pl_pdb-ud-train.conllu.aux_clitic_01",
        ["postanowili", "śmy"],
        "postanowiliśmy",
    )


def test_load_ud_pos_de_adp_det(tst_data_path: Path) -> None:
    _test_load_ud_pos(
        tst_data_path, "de_gsd-ud-train.conllu.adp_det", ["in", "dem"], "im"
    )
