"""All Slovene dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, SLOVENE
from ..tasks import (
    COMMON_SENSE,
    HALLU,
    INSTRUCTION_FOLLOWING,
    KNOW,
    LA,
    NER,
    RC,
    SENT,
    TOOL_CALLING,
    TRANSLATION,
)

# Official datasets ###

WMT24PP_EN_SL_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-sl",
    pretty_name="WMT24++-en-sl",
    source="EuroEval/wmt24pp-en-sl",
    task=TRANSLATION,
    languages=[SLOVENE],
    source_language=ENGLISH,
    target_language=SLOVENE,
    revision="a50f4bf53f1183dd84634644d92d77960d6e6212",
)

FLORES_EN_SL_CONFIG = TranslationDatasetConfig(
    name="flores-en-sl",
    pretty_name="FLORES-en-sl",
    source="EuroEval/flores-en-sl",
    task=TRANSLATION,
    languages=[SLOVENE],
    source_language=ENGLISH,
    target_language=SLOVENE,
    revision="528c44bbdba7128ebe74b8607986772037e65891",
)

SENTINEWS_CONFIG = DatasetConfig(
    name="sentinews",
    pretty_name="Sentinews-sl",
    source="EuroEval/sentinews-mini",
    task=SENT,
    languages=[SLOVENE],
    revision="2520b444bfc498f000d85a3467a5c419cfb9937a",
)

SCALA_SL_CONFIG = DatasetConfig(
    name="scala-sl",
    pretty_name="ScaLA-sl",
    source="EuroEval/scala-sl",
    task=LA,
    languages=[SLOVENE],
    revision="d39c89f635a17267e552f94390c3a7eaf4616e46",
)

SSJ500K_NER_CONFIG = DatasetConfig(
    name="ssj500k-ner",
    pretty_name="ssj500k-NER",
    source="EuroEval/ssj500k-ner-mini",
    task=NER,
    languages=[SLOVENE],
    revision="2b6a32dd2c80e4739307c47f91e65e933958b8a6",
)

MULTI_WIKI_QA_SL_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sl",
    pretty_name="MultiWikiQA-sl",
    source="EuroEval/multi-wiki-qa-sl-mini",
    task=RC,
    languages=[SLOVENE],
    revision="62a1049b0c5f7bb9de046f93f4af5e8c00f739fe",
)

MMLU_SL_CONFIG = DatasetConfig(
    name="mmlu-sl",
    pretty_name="MMLU-sl",
    source="EuroEval/mmlu-sl-mini",
    task=KNOW,
    languages=[SLOVENE],
    revision="8f1f202e8f129c3198836c469d19d5e1ed03c904",
)

WINOGRANDE_SL_CONFIG = DatasetConfig(
    name="winogrande-sl",
    pretty_name="Winogrande-sl",
    source="EuroEval/winogrande-sl",
    task=COMMON_SENSE,
    languages=[SLOVENE],
    labels=["a", "b"],
    revision="d3c85421c7b10cc5cd94f11a23cde33166e1cb4d",
)

MULTI_IFEVAL_SL_CONFIG = DatasetConfig(
    name="multi-ifeval-sl",
    pretty_name="MultiIFEval-sl",
    source="EuroEval/multi-ifeval-sl",
    task=INSTRUCTION_FOLLOWING,
    languages=[SLOVENE],
    train_split=None,
    val_split=None,
    revision="b86dbe7ac71a1608efdb0ff485718ce8ee29f97e",
)

RAGTRUTH_SL_CONFIG = DatasetConfig(
    name="ragtruth-sl",
    pretty_name="RAGTruth-sl",
    source="EuroEval/ragtruth-translated-hallucinations-sl-mini",
    task=HALLU,
    languages=[SLOVENE],
    train_split=None,
    revision="6acf6b9c9e2bad8c53447e8ffd6767f25ecb9d5c",
)


# Unofficial datasets ###

MULTI_BFCL_SL_CONFIG = DatasetConfig(
    name="multi-bfcl-sl",
    pretty_name="MultiBFCL-sl",
    source="EuroEval/multi-bfcl-sl-mini",
    task=TOOL_CALLING,
    languages=[SLOVENE],
    train_split=None,
    revision="56892022be4f95344a22b16ebc96e61810fcc708",
)


WMT24PP_SL_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-sl-en",
    pretty_name="WMT24++-sl-en",
    source="EuroEval/wmt24pp-sl-en",
    task=TRANSLATION,
    languages=[SLOVENE],
    source_language=SLOVENE,
    target_language=ENGLISH,
    unofficial=True,
    revision="0364682560bd30624c7f7b940b10e9a76d0ec477",
)

FLORES_SL_EN_CONFIG = TranslationDatasetConfig(
    name="flores-sl-en",
    pretty_name="FLORES-sl-en",
    source="EuroEval/flores-sl-en",
    task=TRANSLATION,
    languages=[SLOVENE],
    source_language=SLOVENE,
    target_language=ENGLISH,
    unofficial=True,
    revision="42bade6137785d9ead965da110c3f0d20694cf0c",
)

EU_MMLU_SL_CONFIG = DatasetConfig(
    name="eu-mmlu-sl",
    pretty_name="EU-MMLU-sl",
    source="EuroEval/eu-mmlu-sl",
    task=KNOW,
    languages=[SLOVENE],
    unofficial=True,
    revision="c02ac75840430e11ce947044869804fb712d982b",
)
