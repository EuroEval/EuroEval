"""All Romanian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, ROMANIAN
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

ROSENT_CONFIG = DatasetConfig(
    name="ro-sent",
    pretty_name="RoSent",
    source="EuroEval/ro-sent-mini",
    task=SENT,
    languages=[ROMANIAN],
    labels=["positive", "negative"],
)

SCALA_RO_CONFIG = DatasetConfig(
    name="scala-ro",
    pretty_name="ScaLA-ro",
    source="EuroEval/scala-ro",
    task=LA,
    languages=[ROMANIAN],
)

RONEC_CONFIG = DatasetConfig(
    name="ronec",
    pretty_name="RoNEC",
    source="EuroEval/ronec-mini",
    task=NER,
    languages=[ROMANIAN],
)

MULTI_WIKI_QA_RO_CONFIG = DatasetConfig(
    name="multi-wiki-qa-ro",
    pretty_name="MultiWikiQA-ro",
    source="EuroEval/multi-wiki-qa-ro-mini",
    task=RC,
    languages=[ROMANIAN],
)

SUMO_RO_CONFIG = DatasetConfig(
    name="sumo-ro",
    pretty_name="SumO-Ro",
    source="EuroEval/sumo-ro-mini",
    task=SUMM,
    languages=[ROMANIAN],
)

GLOBAL_MMLU_RO_CONFIG = DatasetConfig(
    name="global-mmlu-ro",
    pretty_name="GlobalMMLU-ro",
    source="EuroEval/global-mmlu-ro-mini",
    task=KNOW,
    languages=[ROMANIAN],
)

WINOGRANDE_RO_CONFIG = DatasetConfig(
    name="winogrande-ro",
    pretty_name="Winogrande-ro",
    source="EuroEval/winogrande-ro",
    task=COMMON_SENSE,
    languages=[ROMANIAN],
    labels=["a", "b"],
)

MULTI_IFEVAL_RO_CONFIG = DatasetConfig(
    name="multi-ifeval-ro",
    pretty_name="MultiIFEval-ro",
    source="EuroEval/multi-ifeval-ro",
    task=INSTRUCTION_FOLLOWING,
    languages=[ROMANIAN],
    train_split=None,
    val_split=None,
)

RAGTRUTH_RO_CONFIG = DatasetConfig(
    name="ragtruth-ro",
    pretty_name="RAGTruth-ro",
    source="EuroEval/ragtruth-translated-hallucinations-ro-mini",
    task=HALLU,
    languages=[ROMANIAN],
    train_split=None,
)


WMT24PP_EN_RO_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-ro",
    pretty_name="WMT24++-en-ro",
    source="EuroEval/wmt24pp-en-ro",
    task=TRANSLATION,
    languages=[ROMANIAN],
    source_language=ENGLISH,
    target_language=ROMANIAN,
)

WMT24PP_RO_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-ro-en",
    pretty_name="WMT24++-ro-en",
    source="EuroEval/wmt24pp-ro-en",
    task=TRANSLATION,
    languages=[ROMANIAN],
    source_language=ROMANIAN,
    target_language=ENGLISH,
    unofficial=True,
)


# Unofficial datasets ###

EU_MMLU_RO_CONFIG = DatasetConfig(
    name="eu-mmlu-ro",
    pretty_name="EU-MMLU-ro",
    source="EuroEval/eu-mmlu-ro",
    task=KNOW,
    languages=[ROMANIAN],
    unofficial=True,
)


FLORES_EN_RO_CONFIG = TranslationDatasetConfig(
    name="flores-en-ro",
    pretty_name="FLORES-en-ro",
    source="EuroEval/flores-en-ro",
    task=TRANSLATION,
    languages=[ROMANIAN],
    source_language=ENGLISH,
    target_language=ROMANIAN,
)

FLORES_RO_EN_CONFIG = TranslationDatasetConfig(
    name="flores-ro-en",
    pretty_name="FLORES-ro-en",
    source="EuroEval/flores-ro-en",
    task=TRANSLATION,
    languages=[ROMANIAN],
    source_language=ROMANIAN,
    target_language=ENGLISH,
    unofficial=True,
)
