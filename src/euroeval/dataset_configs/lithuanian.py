"""All Lithuanian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import ENGLISH, LITHUANIAN
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

# WMT24++ translation datasets ###

WMT24PP_EN_LT_CONFIG = DatasetConfig(
    name="wmt24pp-en-lt",
    pretty_name="WMT24++-lt",
    source="EuroEval/wmt24pp-en-lt",
    task=TRANSLATION,
    languages=LITHUANIAN,
    source_language=ENGLISH,
    target_language=LITHUANIAN,
)

WMT24PP_LT_EN_CONFIG = DatasetConfig(
    name="wmt24pp-lt-en",
    pretty_name="WMT24++-lt-en",
    source="EuroEval/wmt24pp-lt-en",
    task=TRANSLATION,
    languages=LITHUANIAN,
    source_language=LITHUANIAN,
    target_language=ENGLISH,
)

# Official datasets ###

ATSILIEPIMAI_CONFIG = DatasetConfig(
    name="atsiliepimai",
    pretty_name="Atsiliepimai",
    source="EuroEval/atsiliepimai",
    task=SENT,
    languages=[LITHUANIAN],
)

SCALA_LT_CONFIG = DatasetConfig(
    name="scala-lt",
    pretty_name="ScaLA-lt",
    source="EuroEval/scala-lt",
    task=LA,
    languages=[LITHUANIAN],
)

WIKIANN_LT_CONFIG = DatasetConfig(
    name="wikiann-lt",
    pretty_name="WikiANN-lt",
    source="EuroEval/wikiann-lt-mini",
    task=NER,
    languages=[LITHUANIAN],
)

MULTI_WIKI_QA_LT_CONFIG = DatasetConfig(
    name="multi-wiki-qa-lt",
    pretty_name="MultiWikiQA-lt",
    source="EuroEval/multi-wiki-qa-lt-mini",
    task=RC,
    languages=[LITHUANIAN],
)

LRYTAS_CONFIG = DatasetConfig(
    name="lrytas",
    pretty_name="Lrytas",
    source="EuroEval/lrytas-mini",
    task=SUMM,
    languages=[LITHUANIAN],
)

LT_HISTORY_CONFIG = DatasetConfig(
    name="lt-history",
    pretty_name="LT-History",
    source="EuroEval/lt-history",
    task=KNOW,
    languages=[LITHUANIAN],
)

WINOGRANDE_LT_CONFIG = DatasetConfig(
    name="winogrande-lt",
    pretty_name="Winogrande-lt",
    source="EuroEval/winogrande-lt",
    task=COMMON_SENSE,
    languages=[LITHUANIAN],
    labels=["a", "b"],
)

MULTI_IFEVAL_LT_CONFIG = DatasetConfig(
    name="multi-ifeval-lt",
    pretty_name="MultiIFEval-lt",
    source="EuroEval/multi-ifeval-lt",
    task=INSTRUCTION_FOLLOWING,
    languages=[LITHUANIAN],
    train_split=None,
    val_split=None,
)

RAGTRUTH_LT_CONFIG = DatasetConfig(
    name="ragtruth-lt",
    pretty_name="RAGTruth-lt",
    source="EuroEval/ragtruth-translated-hallucinations-lt-mini",
    task=HALLU,
    languages=[LITHUANIAN],
    train_split=None,
)

INCLUDE_LT_CONFIG = DatasetConfig(
    name="include-lt",
    pretty_name="INCLUDE-lt",
    source="EuroEval/include-lt-mini",
    task=KNOW,
    languages=[LITHUANIAN],
)

# Unofficial datasets ###

LITHUANIAN_EMOTIONS_CONFIG = DatasetConfig(
    name="lithuanian-emotions",
    pretty_name="Lithuanian Emotions",
    source="EuroEval/lithuanian-emotions-mini",
    task=SENT,
    languages=[LITHUANIAN],
    unofficial=True,
)

EU_MMLU_LT_CONFIG = DatasetConfig(
    name="eu-mmlu-lt",
    pretty_name="EU-MMLU-lt",
    source="EuroEval/eu-mmlu-lt",
    task=KNOW,
    languages=[LITHUANIAN],
    unofficial=True,
)
