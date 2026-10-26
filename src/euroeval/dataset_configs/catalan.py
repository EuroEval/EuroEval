"""All CATALAN dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import CATALAN, ENGLISH
from ..tasks import (
    COMMON_SENSE,
    HALLU,
    INSTRUCTION_FOLLOWING,
    KNOW,
    LA,
    NER,
    RC,
    SENT,
    SUMM,
    TOOL_CALLING,
    TRANSLATION,
)

# Official datasets ###

WMT24PP_EN_CA_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-ca",
    pretty_name="WMT24++-en-ca",
    source="EuroEval/wmt24pp-en-ca",
    task=TRANSLATION,
    languages=[CATALAN],
    source_language=ENGLISH,
    target_language=CATALAN,
    revision="5d84d176c7300fa038648d4131ed1b66a84f95a2",
)

FLORES_EN_CA_CONFIG = TranslationDatasetConfig(
    name="flores-en-ca",
    pretty_name="FLORES-en-ca",
    source="EuroEval/flores-en-ca",
    task=TRANSLATION,
    languages=[CATALAN],
    source_language=ENGLISH,
    target_language=CATALAN,
    revision="a8ed9d89a22dd070f2c91f8ce4378c569a7d4688",
)

GUIA_CAT_CONFIG = DatasetConfig(
    name="guia-cat",
    pretty_name="GuiaCat",
    source="EuroEval/guia-cat-mini",
    task=SENT,
    languages=[CATALAN],
    revision="6b487cc03a7ef10c36db25070a11409ad98da0f8",
)

SCALA_CA_CONFIG = DatasetConfig(
    name="scala-ca",
    pretty_name="ScaLA-ca",
    source="EuroEval/scala-ca",
    task=LA,
    languages=[CATALAN],
    revision="18b2007f3f41702d0de09a2018abe8734967cd3a",
)

WIKIANN_CA_CONFIG = DatasetConfig(
    name="wikiann-ca",
    pretty_name="WikiANN-ca",
    source="EuroEval/wikiann-ca-mini",
    task=NER,
    languages=[CATALAN],
    revision="389808ae4a10efac694307bf0fdfe3d8ab786f0f",
)

MULTI_WIKI_QA_CA_CONFIG = DatasetConfig(
    name="multi-wiki-qa-ca",
    pretty_name="MultiWikiQA-ca",
    source="EuroEval/multi-wiki-qa-ca-mini",
    task=RC,
    languages=[CATALAN],
    revision="7f8c74dc9955a42515a003890edab052af581fee",
)

DACSA_CA_CONFIG = DatasetConfig(
    name="dacsa-ca",
    pretty_name="DACSA-ca",
    source="EuroEval/dacsa-ca-mini",
    task=SUMM,
    languages=[CATALAN],
    revision="547965939bad29cd982e38db102835f265d6028a",
)

MMLU_CA_CONFIG = DatasetConfig(
    name="mmlu-ca",
    pretty_name="MMLU-ca",
    source="EuroEval/mmlu-ca-mini",
    task=KNOW,
    languages=[CATALAN],
    revision="cdeac11b49eae0d3fa2e48d7c8e88d433bdad3ab",
)

WINOGRANDE_CA_CONFIG = DatasetConfig(
    name="winogrande-ca",
    pretty_name="Winogrande-ca",
    source="EuroEval/winogrande-ca",
    task=COMMON_SENSE,
    languages=[CATALAN],
    labels=["a", "b"],
    revision="f362e22dcf6eb1a54287c1709154003a7c3b6755",
)

IFEVAL_CA_CONFIG = DatasetConfig(
    name="ifeval-ca",
    pretty_name="IFEval-ca",
    source="EuroEval/ifeval-ca",
    task=INSTRUCTION_FOLLOWING,
    languages=[CATALAN],
    train_split=None,
    val_split=None,
    revision="09f65978dcdfbc67de5deee7df5a8a283ad7b4ef",
)

RAGTRUTH_CA_CONFIG = DatasetConfig(
    name="ragtruth-ca",
    pretty_name="RAGTruth-ca",
    source="EuroEval/ragtruth-translated-hallucinations-ca-mini",
    task=HALLU,
    languages=[CATALAN],
    train_split=None,
    revision="bdc1971f7ca1728d675b42c186302bc076e13593",
)


# Unofficial datasets ###

MULTI_BFCL_CA_CONFIG = DatasetConfig(
    name="multi-bfcl-ca",
    pretty_name="MultiBFCL-ca",
    source="EuroEval/multi-bfcl-ca-mini",
    task=TOOL_CALLING,
    languages=[CATALAN],
    train_split=None,
    revision="29ea89ee990d0934963f8ea592893ad657a8d532",
)


WMT24PP_CA_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-ca-en",
    pretty_name="WMT24++-ca-en",
    source="EuroEval/wmt24pp-ca-en",
    task=TRANSLATION,
    languages=[CATALAN],
    source_language=CATALAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="0a2b6df52032a94ce130cf14a4d3abe292b6e5f2",
)

FLORES_CA_EN_CONFIG = TranslationDatasetConfig(
    name="flores-ca-en",
    pretty_name="FLORES-ca-en",
    source="EuroEval/flores-ca-en",
    task=TRANSLATION,
    languages=[CATALAN],
    source_language=CATALAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="69148cf01f3a8a8f683866d4a143008d8414091e",
)

MULTI_IFEVAL_CA_CONFIG = DatasetConfig(
    name="multi-ifeval-ca",
    pretty_name="MultiIFEval-ca",
    source="EuroEval/multi-ifeval-ca",
    task=INSTRUCTION_FOLLOWING,
    languages=[CATALAN],
    train_split=None,
    val_split=None,
    unofficial=True,
    revision="9e4540d78fc4ff72f15c402d6992c22d6386c568",
)
