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
    TOOL_CALLING,
    TRANSLATION,
)

# Official datasets ###

WMT24PP_EN_LV_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-lv",
    pretty_name="WMT24++-en-lv",
    source="EuroEval/wmt24pp-en-lv",
    task=TRANSLATION,
    languages=[LATVIAN],
    source_language=ENGLISH,
    target_language=LATVIAN,
    revision="6cc01893e7f5677bba34bbbdf416f404226a3793",
)

FLORES_EN_LV_CONFIG = TranslationDatasetConfig(
    name="flores-en-lv",
    pretty_name="FLORES-en-lv",
    source="EuroEval/flores-en-lv",
    task=TRANSLATION,
    languages=[LATVIAN],
    source_language=ENGLISH,
    target_language=LATVIAN,
    revision="e5aa5533e28f34f1d7b05098ea3ddd16fef0f788",
)

LATVIAN_TWITTER_SENTIMENT_CONFIG = DatasetConfig(
    name="latvian-twitter-sentiment",
    pretty_name="Latvian Twitter Sentiment",
    source="EuroEval/latvian-twitter-sentiment-mini",
    task=SENT,
    languages=[LATVIAN],
    revision="a2047746b7847777fdd50f7c9a08048dd0ed8237",
)

SCALA_LV_CONFIG = DatasetConfig(
    name="scala-lv",
    pretty_name="ScaLA-lv",
    source="EuroEval/scala-lv",
    task=LA,
    languages=[LATVIAN],
    revision="f7bb2acedd39515de9cb1686f5d1fca381893d3f",
)

FULLSTACK_NER_LV_CONFIG = DatasetConfig(
    name="fullstack-ner-lv",
    pretty_name="FullStack NER-lv",
    source="EuroEval/fullstack-ner-lv-mini",
    task=NER,
    languages=[LATVIAN],
    revision="210b1342d6f9a733d284325f7c28ba54bd07ac56",
)

MULTI_WIKI_QA_LV_CONFIG = DatasetConfig(
    name="multi-wiki-qa-lv",
    pretty_name="MultiWikiQA-lv",
    source="EuroEval/multi-wiki-qa-lv-mini",
    task=RC,
    languages=[LATVIAN],
    revision="3c8c53a1294e4d15a2857b724c751c9956b34901",
)

LSM_CONFIG = DatasetConfig(
    name="lsm",
    pretty_name="LSM",
    source="EuroEval/lsm-mini",
    task=SUMM,
    languages=[LATVIAN],
    revision="3dcb4f438003e41a2837edaa541de0422f2267f3",
)


MMLU_LV_CONFIG = DatasetConfig(
    name="mmlu-lv",
    pretty_name="MMLU-lv",
    source="EuroEval/mmlu-lv-mini",
    task=KNOW,
    languages=[LATVIAN],
    revision="90c12b0f8efffd8439effa504ae54c5ac96f0fa4",
)

COPA_LV_CONFIG = DatasetConfig(
    name="copa-lv",
    pretty_name="COPA-lv",
    source="EuroEval/copa-lv",
    task=COMMON_SENSE,
    languages=[LATVIAN],
    labels=["a", "b"],
    revision="42ad1433a7b060eb534ebb31900df4b37fd6360d",
)

MULTI_IFEVAL_LV_CONFIG = DatasetConfig(
    name="multi-ifeval-lv",
    pretty_name="MultiIFEval-lv",
    source="EuroEval/multi-ifeval-lv",
    task=INSTRUCTION_FOLLOWING,
    languages=[LATVIAN],
    train_split=None,
    val_split=None,
    revision="dc8471b265b09d7146e068d069b66f0b00003b6e",
)

RAGTRUTH_LV_CONFIG = DatasetConfig(
    name="ragtruth-lv",
    pretty_name="RAGTruth-lv",
    source="EuroEval/ragtruth-translated-hallucinations-lv-mini",
    task=HALLU,
    languages=[LATVIAN],
    train_split=None,
    revision="1f007e343bb9cd28835f570a0c322380f71cb0d7",
)


# Unofficial datasets ###

MULTI_BFCL_LV_CONFIG = DatasetConfig(
    name="multi-bfcl-lv",
    pretty_name="MultiBFCL-lv",
    source="EuroEval/multi-bfcl-lv-mini",
    task=TOOL_CALLING,
    languages=[LATVIAN],
    train_split=None,
    revision="75f155c898090fc7383d4e16af49229bc1d18b49",
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
    revision="040cb7f88c83db065bc82b2fca114031b8f14a6f",
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
    revision="65c69e31c56192ce2cc3d4d9c94ef7a9152b4a23",
)

WIKIANN_LV_CONFIG = DatasetConfig(
    name="wikiann-lv",
    pretty_name="WikiANN-lv",
    source="EuroEval/wikiann-lv-mini",
    task=NER,
    languages=[LATVIAN],
    unofficial=True,
    revision="b44948498e9a5345fe4bd2cc21506fc8884c40bb",
)

WINOGRANDE_LV_CONFIG = DatasetConfig(
    name="winogrande-lv",
    pretty_name="Winogrande-lv",
    source="EuroEval/winogrande-lv",
    task=COMMON_SENSE,
    languages=[LATVIAN],
    labels=["a", "b"],
    unofficial=True,
    revision="438591e619d06209ccc52d6cf5054bf5c0c2d81a",
)
