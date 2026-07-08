"""All Luxembourgish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import LUXEMBOURGISH
from ..tasks import LA, NER, NLI, SENT, TEXT_CLASSIFICATION

# Official datasets ###

# No official datasets yet

# Unofficial datasets ###
# From the ltzGLUE benchmark (arxiv.org/abs/2604.17976)

LTZGLUE_HC_CONFIG = DatasetConfig(
    name="ltzglue-hc",
    pretty_name="ltzGLUE-HC",
    source="EuroEval/ltzglue-hc",
    task=TEXT_CLASSIFICATION,
    languages=[LUXEMBOURGISH],
    train_split=None,
    val_split=None,
    unofficial=True,
)

LTZGLUE_ID_CONFIG = DatasetConfig(
    name="ltzglue-id",
    pretty_name="ltzGLUE-ID",
    source="EuroEval/ltzglue-id",
    task=TEXT_CLASSIFICATION,
    languages=[LUXEMBOURGISH],
    train_split=None,
    val_split=None,
    unofficial=True,
)

LTZGLUE_LA_BINARY_CONFIG = DatasetConfig(
    name="ltzglue-la-binary",
    pretty_name="ltzGLUE-LA-binary",
    source="EuroEval/ltzglue-la-binary",
    task=LA,
    languages=[LUXEMBOURGISH],
    train_split=None,
    val_split=None,
    unofficial=True,
)

LTZGLUE_LA_MULTI_CONFIG = DatasetConfig(
    name="ltzglue-la-multi",
    pretty_name="ltzGLUE-LA-multi",
    source="EuroEval/ltzglue-la-multi",
    task=LA,
    languages=[LUXEMBOURGISH],
    train_split=None,
    val_split=None,
    unofficial=True,
)

LTZGLUE_NER_CONFIG = DatasetConfig(
    name="ltzglue-ner",
    pretty_name="ltzGLUE-NER",
    source="EuroEval/ltzglue-ner",
    task=NER,
    languages=[LUXEMBOURGISH],
    train_split=None,
    val_split=None,
    unofficial=True,
)

LTZGLUE_RTE_CONFIG = DatasetConfig(
    name="ltzglue-rte",
    pretty_name="ltzGLUE-RTE",
    source="EuroEval/ltzglue-rte",
    task=NLI,
    languages=[LUXEMBOURGISH],
    train_split=None,
    val_split=None,
    unofficial=True,
)

LTZGLUE_SA_CONFIG = DatasetConfig(
    name="ltzglue-sa",
    pretty_name="ltzGLUE-SA",
    source="EuroEval/ltzglue-sa",
    task=SENT,
    languages=[LUXEMBOURGISH],
    train_split=None,
    val_split=None,
    unofficial=True,
)

LTZGLUE_TC_CONFIG = DatasetConfig(
    name="ltzglue-tc",
    pretty_name="ltzGLUE-TC",
    source="EuroEval/ltzglue-tc",
    task=TEXT_CLASSIFICATION,
    languages=[LUXEMBOURGISH],
    train_split=None,
    val_split=None,
    unofficial=True,
)
