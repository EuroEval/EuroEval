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
    TRANSLATION,
)

# Official datasets ###

GUIA_CAT_CONFIG = DatasetConfig(
    name="guia-cat",
    pretty_name="GuiaCat",
    source="EuroEval/guia-cat-mini",
    task=SENT,
    languages=[CATALAN],
)

SCALA_CA_CONFIG = DatasetConfig(
    name="scala-ca",
    pretty_name="ScaLA-ca",
    source="EuroEval/scala-ca",
    task=LA,
    languages=[CATALAN],
)

WIKIANN_CA_CONFIG = DatasetConfig(
    name="wikiann-ca",
    pretty_name="WikiANN-ca",
    source="EuroEval/wikiann-ca-mini",
    task=NER,
    languages=[CATALAN],
)

MULTI_WIKI_QA_CA_CONFIG = DatasetConfig(
    name="multi-wiki-qa-ca",
    pretty_name="MultiWikiQA-ca",
    source="EuroEval/multi-wiki-qa-ca-mini",
    task=RC,
    languages=[CATALAN],
)

DACSA_CA_CONFIG = DatasetConfig(
    name="dacsa-ca",
    pretty_name="DACSA-ca",
    source="EuroEval/dacsa-ca-mini",
    task=SUMM,
    languages=[CATALAN],
)

MMLU_CA_CONFIG = DatasetConfig(
    name="mmlu-ca",
    pretty_name="MMLU-ca",
    source="EuroEval/mmlu-ca-mini",
    task=KNOW,
    languages=[CATALAN],
)

WINOGRANDE_CA_CONFIG = DatasetConfig(
    name="winogrande-ca",
    pretty_name="Winogrande-ca",
    source="EuroEval/winogrande-ca",
    task=COMMON_SENSE,
    languages=[CATALAN],
    labels=["a", "b"],
)

IFEVAL_CA_CONFIG = DatasetConfig(
    name="ifeval-ca",
    pretty_name="IFEval-ca",
    source="EuroEval/ifeval-ca",
    task=INSTRUCTION_FOLLOWING,
    languages=[CATALAN],
    train_split=None,
    val_split=None,
)

RAGTRUTH_CA_CONFIG = DatasetConfig(
    name="ragtruth-ca",
    pretty_name="RAGTruth-ca",
    source="EuroEval/ragtruth-translated-hallucinations-ca-mini",
    task=HALLU,
    languages=[CATALAN],
    train_split=None,
)

WMT24PP_EN_CA_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-ca",
    pretty_name="WMT24++-en-ca",
    source="EuroEval/wmt24pp-en-ca",
    task=TRANSLATION,
    languages=[CATALAN],
    source_language=ENGLISH,
    target_language=CATALAN,
)

WMT24PP_CA_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-ca-en",
    pretty_name="WMT24++-ca-en",
    source="EuroEval/wmt24pp-ca-en",
    task=TRANSLATION,
    languages=[CATALAN],
    source_language=CATALAN,
    target_language=ENGLISH,
)


# Unofficial datasets ###

MULTI_IFEVAL_CA_CONFIG = DatasetConfig(
    name="multi-ifeval-ca",
    pretty_name="MultiIFEval-ca",
    source="EuroEval/multi-ifeval-ca",
    task=INSTRUCTION_FOLLOWING,
    languages=[CATALAN],
    train_split=None,
    val_split=None,
    unofficial=True,
)


FLORES_EN_CA_CONFIG = TranslationDatasetConfig(
    name="flores-en-ca",
    pretty_name="FLORES-en-ca",
    source="EuroEval/flores-en-ca",
    task=TRANSLATION,
    languages=[CATALAN],
    source_language=ENGLISH,
    target_language=CATALAN,
    unofficial=True,
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
)
