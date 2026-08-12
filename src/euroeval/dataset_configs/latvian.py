"""All Latvian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, LATVIAN
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

LATVIAN_TWITTER_SENTIMENT_CONFIG = DatasetConfig(
    name="latvian-twitter-sentiment",
    pretty_name="Latvian Twitter Sentiment",
    source="EuroEval/latvian-twitter-sentiment-mini",
    task=SENT,
    languages=[LATVIAN],
)

SCALA_LV_CONFIG = DatasetConfig(
    name="scala-lv",
    pretty_name="ScaLA-lv",
    source="EuroEval/scala-lv",
    task=LA,
    languages=[LATVIAN],
)

FULLSTACK_NER_LV_CONFIG = DatasetConfig(
    name="fullstack-ner-lv",
    pretty_name="FullStack NER-lv",
    source="EuroEval/fullstack-ner-lv-mini",
    task=NER,
    languages=[LATVIAN],
)

MULTI_WIKI_QA_LV_CONFIG = DatasetConfig(
    name="multi-wiki-qa-lv",
    pretty_name="MultiWikiQA-lv",
    source="EuroEval/multi-wiki-qa-lv-mini",
    task=RC,
    languages=[LATVIAN],
)

LSM_CONFIG = DatasetConfig(
    name="lsm",
    pretty_name="LSM",
    source="EuroEval/lsm-mini",
    task=SUMM,
    languages=[LATVIAN],
)


MMLU_LV_CONFIG = DatasetConfig(
    name="mmlu-lv",
    pretty_name="MMLU-lv",
    source="EuroEval/mmlu-lv-mini",
    task=KNOW,
    languages=[LATVIAN],
)

COPA_LV_CONFIG = DatasetConfig(
    name="copa-lv",
    pretty_name="COPA-lv",
    source="EuroEval/copa-lv",
    task=COMMON_SENSE,
    languages=[LATVIAN],
    labels=["a", "b"],
)

MULTI_IFEVAL_LV_CONFIG = DatasetConfig(
    name="multi-ifeval-lv",
    pretty_name="MultiIFEval-lv",
    source="EuroEval/multi-ifeval-lv",
    task=INSTRUCTION_FOLLOWING,
    languages=[LATVIAN],
    train_split=None,
    val_split=None,
)

RAGTRUTH_LV_CONFIG = DatasetConfig(
    name="ragtruth-lv",
    pretty_name="RAGTruth-lv",
    source="EuroEval/ragtruth-translated-hallucinations-lv-mini",
    task=HALLU,
    languages=[LATVIAN],
    train_split=None,
)


WMT24PP_EN_LV_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-lv",
    pretty_name="WMT24++-en-lv",
    source="EuroEval/wmt24pp-en-lv",
    task=TRANSLATION,
    languages=[LATVIAN],
    source_language=ENGLISH,
    target_language=LATVIAN,
)

WMT24PP_LV_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-lv-en",
    pretty_name="WMT24++-lv-en",
    source="EuroEval/wmt24pp-lv-en",
    task=TRANSLATION,
    languages=[LATVIAN],
    source_language=LATVIAN,
    target_language=ENGLISH,
    unofficial=True,
)


# Unofficial datasets ###

WIKIANN_LV_CONFIG = DatasetConfig(
    name="wikiann-lv",
    pretty_name="WikiANN-lv",
    source="EuroEval/wikiann-lv-mini",
    task=NER,
    languages=[LATVIAN],
    unofficial=True,
)

WINOGRANDE_LV_CONFIG = DatasetConfig(
    name="winogrande-lv",
    pretty_name="Winogrande-lv",
    source="EuroEval/winogrande-lv",
    task=COMMON_SENSE,
    languages=[LATVIAN],
    labels=["a", "b"],
    unofficial=True,
)


FLORES_EN_LV_CONFIG = TranslationDatasetConfig(
    name="flores-en-lv",
    pretty_name="FLORES-en-lv",
    source="EuroEval/flores-en-lv",
    task=TRANSLATION,
    languages=[LATVIAN],
    source_language=ENGLISH,
    target_language=LATVIAN,
    unofficial=True,
)

FLORES_LV_EN_CONFIG = TranslationDatasetConfig(
    name="flores-lv-en",
    pretty_name="FLORES-lv-en",
    source="EuroEval/flores-lv-en",
    task=TRANSLATION,
    languages=[LATVIAN],
    source_language=LATVIAN,
    target_language=ENGLISH,
    unofficial=True,
)
