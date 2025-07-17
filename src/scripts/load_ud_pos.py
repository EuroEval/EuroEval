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
from typing import Callable, Dict, List, Union

import pandas as pd
import requests

logging.basicConfig(format="%(asctime)s ⋅ %(message)s", level=logging.INFO)
logger = logging.getLogger("load_ud_pos")


def load_dadt_pos() -> Dict[str, pd.DataFrame]:
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


def load_ptdt_pos() -> Dict[str, pd.DataFrame]:
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


def load_fodt_pos() -> Dict[str, pd.DataFrame]:
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


def load_isdt_pos() -> Dict[str, pd.DataFrame]:
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


def load_nodt_nb_pos() -> Dict[str, pd.DataFrame]:
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


def load_nodt_nn_pos() -> Dict[str, pd.DataFrame]:
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


def load_svdt_pos() -> Dict[str, pd.DataFrame]:
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


def load_dedt_pos() -> Dict[str, pd.DataFrame]:
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


def load_nldt_pos() -> Dict[str, pd.DataFrame]:
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


def load_endt_pos() -> Dict[str, pd.DataFrame]:
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


def load_frdt_pos() -> Dict[str, pd.DataFrame]:
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


def load_itdt_pos() -> Dict[str, pd.DataFrame]:
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


def load_esdt_pos() -> Dict[str, pd.DataFrame]:
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


def load_fidt_pos() -> Dict[str, pd.DataFrame]:
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


def load_ud_pos(
    train_url: str,
    val_url: str,
    test_url: str,
    doc_process_fn: Callable[[str], str] = lambda x: x,
    filter_source: str | None = None,
) -> Dict[str, pd.DataFrame]:
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
    # Download the data
    data = dict(
        train=requests.get(train_url).text.split("\n"),
        val=requests.get(val_url).text.split("\n"),
        test=requests.get(test_url).text.split("\n"),
    )

    if filter_source is not None:
        logger.warning(
            f"Warning: Filtering dataset to include only entries with {filter_source=}"
        )

    # Iterate over the data splits
    dfs = dict()
    for split, lines in data.items():
        # Initialise the records, data dictionary and document
        records = list()
        data_dict: Dict[str, List[Union[int, str]]] = defaultdict(list)
        doc = ""
        source = ""

        # Iterate over the data for the given split
        for line in lines:
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
            elif line == "":
                if len(data_dict["tokens"]) > 0:
                    if filter_source is None or filter_source in source:
                        merged_data_dict: Dict[str, Union[str, List[Union[int, str]]]]
                        merged_data_dict = {**data_dict, "doc": doc}
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

        # Convert the records to a dataframe and store it
        dfs[split] = pd.DataFrame.from_records(records)

    # Return the dictionary of dataframes
    return dfs
