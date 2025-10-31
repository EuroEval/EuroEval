# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "pandas==2.2.0",
#     "requests==2.32.3",
# ]
# ///

"""Load the part-of-speech part of a Universal Dependencies treebank."""

import logging
import re
from collections import defaultdict
from functools import partial
from typing import Callable, DefaultDict
from urllib.parse import urlparse

import pandas as pd
import requests

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("load_ud_pos")


def load_dadt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Danish Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://github.com/UniversalDependencies/UD_Danish-DDT/raw/master/"
        "da_ddt-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_ptdt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Portuguese Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://github.com/UniversalDependencies/UD_Portuguese-Bosque/raw/master/"
        "pt_bosque-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(
        train_url=train_url,
        val_url=val_url,
        test_url=test_url,
        filter_source="CETEMPúblico",
    )


def load_fodt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Faroese Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://github.com/UniversalDependencies/UD_Faroese-FarPaHC/raw/master/"
        "fo_farpahc-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_isdt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Icelandic Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://github.com/UniversalDependencies/UD_Icelandic-Modern/raw/master/"
        "is_modern-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_nodt_nb_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Norwegian Bokmål Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal/raw/master/"
        "no_bokmaal-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_nodt_nn_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Norwegian Nynorsk Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://github.com/UniversalDependencies/UD_Norwegian-Nynorsk/raw/master/"
        "no_nynorsk-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_svdt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Swedish Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://github.com/UniversalDependencies/UD_Swedish-Talbanken/raw/master/"
        "sv_talbanken-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    # Define document processing function
    def process_document(doc: str) -> str:
        doc = (
            doc.replace(" s k", " s.k.")
            .replace("S k", "S.k.")
            .replace(" bl a", " bl.a.")
            .replace("Bl a", "Bl.a.")
            .replace(" t o m", " t.o.m.")
            .replace("T o m", "T.o.m.")
            .replace(" fr o m", " fr.o.m.")
            .replace("Fr o m", "Fr.o.m.")
            .replace(" o s v", " o.s.v.")
            .replace("O s v", "O.s.v.")
            .replace(" d v s", " d.v.s.")
            .replace("D v s", "D.v.s.")
            .replace(" m fl", " m.fl.")
            .replace("M fl", "M.fl.")
            .replace(" t ex", " t.ex.")
            .replace("T ex", "T.ex.")
            .replace(" f n", " f.n.")
            .replace("F n", "F.n.")
        )
        return doc

    return load_ud_pos(
        train_url=train_url,
        val_url=val_url,
        test_url=test_url,
        doc_process_fn=process_document,
    )


def load_dedt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the German Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/"
        "de_gsd-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_nldt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Dutch Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Dutch-Alpino/"
        "master/nl_alpino-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_endt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the English Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/"
        "master/en_gum-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_frdt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the French Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/"
        "master/fr_gsd-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_itdt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Italian Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/"
        "master/it_isdt-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_esdt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Spanish Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora"
        "/refs/heads/master/es_ancora-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_fidt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Finnish Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Finnish-TDT/"
        "master/fi_tdt-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_lvdt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Latvian Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Latvian-LVTB/"
        "master/lv_lvtb-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_etdt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Estonian Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Estonian-EDT/"
        "refs/heads/master/et_edt-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_pldt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Polish Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Polish-PDB/"
        "refs/heads/master/pl_pdb-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_ltdt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Lithuanian Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Lithuanian-ALKSNIS/"
        "master/lt_alksnis-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_csdt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Czech Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Czech-CAC/refs/"
        "heads/master/cs_cac-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_skdt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Slovak Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val`, and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Slovak-SNK/refs/heads/master/"
        "sk_snk-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_ukdt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Ukrainian Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Ukrainian-ParlaMint/refs/heads/master/"
        "uk_parlamint-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_eldt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Greek Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val`, and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Greek-GUD/refs/heads/master/"
        "el_gud-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_bgdt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Bulgarian Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Bulgarian-BTB/refs/heads/master/"
        "bg_btb-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_srdt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Serbian Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val`, and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Serbian-SET/refs/heads/master/"
        "sr_set-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_sldt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Slovenian Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val`, and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Slovenian-SSJ/refs/heads/master/"
        "sl_ssj-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def load_hrdt_pos() -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of the Croatian Dependency Treebank.

    Returns:
        The dataframes, stored in the keys `train`, `val`, and `test`.
    """
    # Define download URLs
    base_url = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Croatian-SET/refs/heads/master/"
        "hr_set-ud-{}.conllu"
    )
    train_url = base_url.format("train")
    val_url = base_url.format("dev")
    test_url = base_url.format("test")

    return load_ud_pos(train_url=train_url, val_url=val_url, test_url=test_url)


def _load_file_or_url(url_or_path: str) -> list[str]:
    """Load a file from a URL or local path.

    Args:
        url_or_path: The URL or local path to load.

    Returns:
        The list of strings, one per line.
    """
    parsed = urlparse(url_or_path)
    if parsed.scheme.lower() in ("http", "https"):
        return requests.get(url_or_path).text.split("\n")
    else:
        with open(url_or_path, "r") as f:
            logger.warning(f"Loading data from local file: {url_or_path}")
            return [line.strip() for line in f.readlines()]


def _filter_token_range(data_dict: dict[str, list]) -> dict[str, list]:
    """Filter out tokens that belong to ranges in UD source files.

    Tokens that span more than one position are not supported by create_scala's
    prepare_df logic.

    Example files:

    - tests/test_scripts/test_create_scala/test_data/de_gsd-ud-train.conllu.adp_det
    - tests/test_scripts/test_create_scala/test_data/pl_pdb-ud-train.conllu.aux_clitic_*

    Args:
        data_dict: The input data dictionary.

    Returns:
        The filtered data dictionary. Its format is identical to the input.
    """
    output: DefaultDict[str, list] = defaultdict(list)

    range_start: int = 0
    range_end: int = 0

    # Pattern used to detect merged IDs
    merged_ids_pattern = re.compile(r"(\d+)-(\d+)", re.I)

    for i in range(len(data_dict["ids"])):
        match = merged_ids_pattern.match(data_dict["ids"][i])
        if match is not None:
            output["ids"].append(data_dict["ids"][i])
            output["tokens"].append(data_dict["tokens"][i])
            output["pos_tags"].append(data_dict["pos_tags"][i])
            range_start = int(match.group(1))
            range_end = int(match.group(2))
        else:
            token_id = int(data_dict["ids"][i].split(".")[0])
            if token_id >= range_start and token_id <= range_end:
                # Skip token if in range
                continue
            else:
                output["ids"].append(data_dict["ids"][i])
                output["tokens"].append(data_dict["tokens"][i])
                output["pos_tags"].append(data_dict["pos_tags"][i])

    return output


def _load_split(
    *,
    lines: list[str],
    filter_source: str | None = None,
    doc_process_fn: Callable[[str], str] = lambda x: x,
) -> pd.DataFrame:
    """Load single split of the POS part of a Universal Dependencies treebank.

    Args:
        lines: The lines of the file to process.
        filter_source:
            If not `None`, only include entries with this source in the dataset.
        doc_process_fn:
            A function to apply to each document before parsing it.

    Returns:
        The dataframe for the given split.
    """
    # Initialise the records, data dictionary and document
    records: list[dict] = []
    data_dict: dict[str, list[list[int | str] | str]] = defaultdict(list)

    # Iterate over the data for the given split
    doc = ""
    source = ""
    for line_idx, line in enumerate(lines):
        # If we are at the first line of an entry then extract the document
        if line.startswith("# text = "):
            doc = re.sub("# text = ", "", line)

            # Process the document if needed
            doc = doc_process_fn(doc)

        elif line.startswith("# source = "):
            source = line.removeprefix("# source = ").strip()

        # Otherwise, if the line is a comment then ignore it
        elif line.startswith("#"):
            continue

        # Otherwise, if we have reached the end of an entry then store it to the
        # list of records and reset the data dictionary and document
        elif line == "" or line_idx == len(lines) - 1:
            if len(data_dict["tokens"]) > 0:
                if filter_source is None or filter_source in source:
                    merged_data_dict: dict[str, str | list[int | str]]
                    merged_data_dict = {**_filter_token_range(data_dict), "doc": doc}
                    records.append(merged_data_dict)
            data_dict = defaultdict(list)
            doc = ""
            source = ""

        # Otherwise we are in the middle of an entry which is not a comment, so
        # we extract the data from the line and store it in the data dictionary
        else:
            data_tup = line.split("\t")
            data_dict["ids"].append(data_tup[0])
            data_dict["tokens"].append(data_tup[1])
            data_dict["pos_tags"].append(data_tup[3])

    # Convert the records to a dataframe
    return pd.DataFrame.from_records(records)


def load_ud_pos(
    train_url: str,
    val_url: str,
    test_url: str,
    doc_process_fn: Callable[[str], str] = lambda x: x,
    filter_source: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load the part-of-speech part of a Universal Dependencies treebank.

    Args:
        train_url:
            The URL of the training data.
        val_url:
            The URL of the validation data.
        test_url:
            The URL of the test data.
        doc_process_fn:
            A function to apply to each document before parsing it.
        filter_source:
            If not `None`, only include entries with this source in the dataset.

    Returns:
        The dataframes, stored in the keys `train`, `val` and `test`.
    """
    if filter_source is not None:
        logger.warning(
            f"Warning: Filtering dataset to include only entries with {filter_source=}"
        )

    # Load the lines of the splits
    train_lines = _load_file_or_url(url_or_path=train_url)
    val_lines = _load_file_or_url(url_or_path=val_url)
    test_lines = _load_file_or_url(url_or_path=test_url)

    # Load the splits
    split_loader = partial(
        _load_split, filter_source=filter_source, doc_process_fn=doc_process_fn
    )
    dfs = dict(
        train=split_loader(lines=train_lines),
        val=split_loader(lines=val_lines),
        test=split_loader(lines=test_lines),
    )

    # Return the dictionary of dataframes
    return dfs
