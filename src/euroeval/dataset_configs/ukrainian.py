"""All Ukrainian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, UKRAINIAN
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

CROSS_DOMAIN_UK_REVIEWS_CONFIG = DatasetConfig(
    name="cross-domain-uk-reviews",
    pretty_name="Cross Domain Ukrainian Reviews",
    source="EuroEval/cross-domain-uk-reviews-mini",
    task=SENT,
    languages=[UKRAINIAN],
)

SCALA_UK_CONFIG = DatasetConfig(
    name="scala-uk",
    pretty_name="ScaLA-uk",
    source="EuroEval/scala-uk",
    task=LA,
    languages=[UKRAINIAN],
)

NER_UK_CONFIG = DatasetConfig(
    name="ner-uk",
    pretty_name="NER-uk",
    source="EuroEval/ner-uk-mini",
    task=NER,
    languages=[UKRAINIAN],
)

MULTI_WIKI_QA_UK_CONFIG = DatasetConfig(
    name="multi-wiki-qa-uk",
    pretty_name="MultiWikiQA-uk",
    source="EuroEval/multi-wiki-qa-uk-mini",
    task=RC,
    languages=[UKRAINIAN],
)

LR_SUM_UK_CONFIG = DatasetConfig(
    name="lr-sum-uk",
    pretty_name="LRSum-uk",
    source="EuroEval/lr-sum-uk-mini",
    task=SUMM,
    languages=[UKRAINIAN],
)

WINOGRANDE_UK_CONFIG = DatasetConfig(
    name="winogrande-uk",
    pretty_name="Winogrande-uk",
    source="EuroEval/winogrande-uk",
    task=COMMON_SENSE,
    languages=[UKRAINIAN],
    labels=["a", "b"],
)

MULTI_IFEVAL_UK_CONFIG = DatasetConfig(
    name="multi-ifeval-uk",
    pretty_name="MultiIFEval-uk",
    source="EuroEval/multi-ifeval-uk",
    task=INSTRUCTION_FOLLOWING,
    languages=[UKRAINIAN],
    train_split=None,
    val_split=None,
)

RAGTRUTH_UK_CONFIG = DatasetConfig(
    name="ragtruth-uk",
    pretty_name="RAGTruth-uk",
    source="EuroEval/ragtruth-translated-hallucinations-uk-mini",
    task=HALLU,
    languages=[UKRAINIAN],
    train_split=None,
)


INCLUDE_UK_CONFIG = DatasetConfig(
    name="include-uk",
    pretty_name="INCLUDE-uk",
    source="EuroEval/include-uk-mini",
    task=KNOW,
    languages=[UKRAINIAN],
)


WMT24PP_EN_UK_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-uk",
    pretty_name="WMT24++-en-uk",
    source="EuroEval/wmt24pp-en-uk",
    task=TRANSLATION,
    languages=[UKRAINIAN],
    source_language=ENGLISH,
    target_language=UKRAINIAN,
)

WMT24PP_UK_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-uk-en",
    pretty_name="WMT24++-uk-en",
    source="EuroEval/wmt24pp-uk-en",
    task=TRANSLATION,
    languages=[UKRAINIAN],
    source_language=UKRAINIAN,
    target_language=ENGLISH,
    unofficial=True,
)


# Unofficial datasets ###

GLOBAL_MMLU_UK_CONFIG = DatasetConfig(
    name="global-mmlu-uk",
    pretty_name="GlobalMMLU-uk",
    source="EuroEval/global-mmlu-uk-mini",
    task=KNOW,
    languages=[UKRAINIAN],
    unofficial=True,
)

IFEVAL_UK_CONFIG = DatasetConfig(
    name="ifeval-uk",
    pretty_name="IFEval-uk",
    source="EuroEval/ifeval-uk",
    task=INSTRUCTION_FOLLOWING,
    languages=[UKRAINIAN],
    train_split=None,
    val_split=None,
    unofficial=True,
)

FLORES_EN_UK_CONFIG = TranslationDatasetConfig(
    name="flores-en-uk",
    pretty_name="FLORES-en-uk",
    source="EuroEval/flores-en-uk",
    task=TRANSLATION,
    languages=[UKRAINIAN],
    source_language=ENGLISH,
    target_language=UKRAINIAN,
    unofficial=True,
)

FLORES_UK_EN_CONFIG = TranslationDatasetConfig(
    name="flores-uk-en",
    pretty_name="FLORES-uk-en",
    source="EuroEval/flores-uk-en",
    task=TRANSLATION,
    languages=[UKRAINIAN],
    source_language=UKRAINIAN,
    target_language=ENGLISH,
    unofficial=True,
)
