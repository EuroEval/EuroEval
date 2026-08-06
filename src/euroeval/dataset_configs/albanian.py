"""All Albanian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ALBANIAN, ENGLISH
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

MMS_SQ_CONFIG = DatasetConfig(
    name="mms-sq",
    pretty_name="MMS-sq",
    source="EuroEval/mms-sq-mini",
    task=SENT,
    languages=[ALBANIAN],
)

SCALA_SQ_CONFIG = DatasetConfig(
    name="scala-sq",
    pretty_name="ScaLA-sq",
    source="EuroEval/scala-sq",
    task=LA,
    languages=[ALBANIAN],
)

WIKIANN_SQ_CONFIG = DatasetConfig(
    name="wikiann-sq",
    pretty_name="WikiANN-sq",
    source="EuroEval/wikiann-sq-mini",
    task=NER,
    languages=[ALBANIAN],
)

MULTI_WIKI_QA_SQ_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sq",
    pretty_name="MultiWikiQA-sq",
    source="EuroEval/multi-wiki-qa-sq-mini",
    task=RC,
    languages=[ALBANIAN],
)

LR_SUM_SQ_CONFIG = DatasetConfig(
    name="lr-sum-sq",
    pretty_name="LRSum-sq",
    source="EuroEval/lr-sum-sq-mini",
    task=SUMM,
    languages=[ALBANIAN],
)

WINOGRANDE_SQ_CONFIG = DatasetConfig(
    name="winogrande-sq",
    pretty_name="Winogrande-sq",
    source="EuroEval/winogrande-sq",
    task=COMMON_SENSE,
    languages=[ALBANIAN],
    labels=["a", "b"],
)

MULTI_IFEVAL_SQ_CONFIG = DatasetConfig(
    name="multi-ifeval-sq",
    pretty_name="MultiIFEval-sq",
    source="EuroEval/multi-ifeval-sq",
    task=INSTRUCTION_FOLLOWING,
    languages=[ALBANIAN],
    train_split=None,
    val_split=None,
)


INCLUDE_SQ_CONFIG = DatasetConfig(
    name="include-sq",
    pretty_name="INCLUDE-sq",
    source="EuroEval/include-sq-mini",
    task=KNOW,
    languages=[ALBANIAN],
)

FLORES_EN_SQ_CONFIG = TranslationDatasetConfig(
    name="flores-en-sq",
    pretty_name="FLORES-en-sq",
    source="EuroEval/flores-en-sq",
    task=TRANSLATION,
    languages=[ALBANIAN],
    source_language=ENGLISH,
    target_language=ALBANIAN,
)

FLORES_SQ_EN_CONFIG = TranslationDatasetConfig(
    name="flores-sq-en",
    pretty_name="FLORES-sq-en",
    source="EuroEval/flores-sq-en",
    task=TRANSLATION,
    languages=[ALBANIAN],
    source_language=ALBANIAN,
    target_language=ENGLISH,
)


# Unofficial datasets ###

GLOBAL_MMLU_LITE_SQ_CONFIG = DatasetConfig(
    name="global-mmlu-lite-sq",
    pretty_name="GlobalMMLULite-sq",
    source="EuroEval/global-mmlu-lite-sq",
    task=KNOW,
    languages=[ALBANIAN],
    unofficial=True,
)

RAGTRUTH_SQ_CONFIG = DatasetConfig(
    name="ragtruth-sq",
    pretty_name="RAGTruth-sq",
    source="EuroEval/ragtruth-translated-hallucinations-sq-mini",
    task=HALLU,
    languages=[ALBANIAN],
    train_split=None,
    unofficial=True,
)
