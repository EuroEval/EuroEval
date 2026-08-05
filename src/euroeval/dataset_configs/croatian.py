"""All Croatian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import CROATIAN, ENGLISH
from ..tasks import (
    COMMON_SENSE,
    HALLU,
    INSTRUCTION_FOLLOWING,
    KNOW,
    LA,
    NER,
    RC,
    SENT,
    TRANSLATION,
)

# Official datasets ###

MMS_HR_CONFIG = DatasetConfig(
    name="mms-hr",
    pretty_name="MMS-hr",
    source="EuroEval/mms-hr-mini",
    task=SENT,
    languages=[CROATIAN],
)

SCALA_HR_CONFIG = DatasetConfig(
    name="scala-hr",
    pretty_name="ScaLA-hr",
    source="EuroEval/scala-hr",
    task=LA,
    languages=[CROATIAN],
)

WIKIANN_HR_CONFIG = DatasetConfig(
    name="wikiann-hr",
    pretty_name="WikiANN-hr",
    source="EuroEval/wikiann-hr-mini",
    task=NER,
    languages=[CROATIAN],
)

MULTI_WIKI_QA_HR_CONFIG = DatasetConfig(
    name="multi-wiki-qa-hr",
    pretty_name="MultiWikiQA-hr",
    source="EuroEval/multi-wiki-qa-hr-mini",
    task=RC,
    languages=[CROATIAN],
)

WINOGRANDE_HR_CONFIG = DatasetConfig(
    name="winogrande-hr",
    pretty_name="Winogrande-hr",
    source="EuroEval/winogrande-hr",
    task=COMMON_SENSE,
    languages=[CROATIAN],
    labels=["a", "b"],
)

MULTI_IFEVAL_HR_CONFIG = DatasetConfig(
    name="multi-ifeval-hr",
    pretty_name="MultiIFEval-hr",
    source="EuroEval/multi-ifeval-hr",
    task=INSTRUCTION_FOLLOWING,
    languages=[CROATIAN],
    train_split=None,
    val_split=None,
)

RAGTRUTH_HR_CONFIG = DatasetConfig(
    name="ragtruth-hr",
    pretty_name="RAGTruth-hr",
    source="EuroEval/ragtruth-translated-hallucinations-hr-mini",
    task=HALLU,
    languages=[CROATIAN],
    train_split=None,
)


INCLUDE_HR_CONFIG = DatasetConfig(
    name="include-hr",
    pretty_name="INCLUDE-hr",
    source="EuroEval/include-hr-mini",
    task=KNOW,
    languages=[CROATIAN],
)

WMT24PP_EN_HR_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-hr",
    pretty_name="WMT24++-en-hr",
    source="EuroEval/wmt24pp-en-hr",
    task=TRANSLATION,
    languages=[CROATIAN],
    source_language=ENGLISH,
    target_language=CROATIAN,
)

WMT24PP_HR_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-hr-en",
    pretty_name="WMT24++-hr-en",
    source="EuroEval/wmt24pp-hr-en",
    task=TRANSLATION,
    languages=[CROATIAN],
    source_language=CROATIAN,
    target_language=ENGLISH,
)


# Unofficial datasets ###

MMLU_HR_CONFIG = DatasetConfig(
    name="mmlu-hr",
    pretty_name="MMLU-hr",
    source="EuroEval/mmlu-hr-mini",
    task=KNOW,
    languages=[CROATIAN],
    unofficial=True,
)

EU_MMLU_HR_CONFIG = DatasetConfig(
    name="eu-mmlu-hr",
    pretty_name="EU-MMLU-hr",
    source="EuroEval/eu-mmlu-hr",
    task=KNOW,
    languages=[CROATIAN],
    unofficial=True,
)
