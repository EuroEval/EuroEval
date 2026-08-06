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
    TRANSLATION,
)

# Official datasets ###

SENTINEWS_CONFIG = DatasetConfig(
    name="sentinews",
    pretty_name="Sentinews-sl",
    source="EuroEval/sentinews-mini",
    task=SENT,
    languages=[SLOVENE],
)

SCALA_SL_CONFIG = DatasetConfig(
    name="scala-sl",
    pretty_name="ScaLA-sl",
    source="EuroEval/scala-sl",
    task=LA,
    languages=[SLOVENE],
)

SSJ500K_NER_CONFIG = DatasetConfig(
    name="ssj500k-ner",
    pretty_name="ssj500k-NER",
    source="EuroEval/ssj500k-ner-mini",
    task=NER,
    languages=[SLOVENE],
)

MULTI_WIKI_QA_SL_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sl",
    pretty_name="MultiWikiQA-sl",
    source="EuroEval/multi-wiki-qa-sl-mini",
    task=RC,
    languages=[SLOVENE],
)

MMLU_SL_CONFIG = DatasetConfig(
    name="mmlu-sl",
    pretty_name="MMLU-sl",
    source="EuroEval/mmlu-sl-mini",
    task=KNOW,
    languages=[SLOVENE],
)

WINOGRANDE_SL_CONFIG = DatasetConfig(
    name="winogrande-sl",
    pretty_name="Winogrande-sl",
    source="EuroEval/winogrande-sl",
    task=COMMON_SENSE,
    languages=[SLOVENE],
    labels=["a", "b"],
)

MULTI_IFEVAL_SL_CONFIG = DatasetConfig(
    name="multi-ifeval-sl",
    pretty_name="MultiIFEval-sl",
    source="EuroEval/multi-ifeval-sl",
    task=INSTRUCTION_FOLLOWING,
    languages=[SLOVENE],
    train_split=None,
    val_split=None,
)

RAGTRUTH_SL_CONFIG = DatasetConfig(
    name="ragtruth-sl",
    pretty_name="RAGTruth-sl",
    source="EuroEval/ragtruth-translated-hallucinations-sl-mini",
    task=HALLU,
    languages=[SLOVENE],
    train_split=None,
)


WMT24PP_EN_SL_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-sl",
    pretty_name="WMT24++-en-sl",
    source="EuroEval/wmt24pp-en-sl",
    task=TRANSLATION,
    languages=[SLOVENE],
    source_language=ENGLISH,
    target_language=SLOVENE,
)

WMT24PP_SL_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-sl-en",
    pretty_name="WMT24++-sl-en",
    source="EuroEval/wmt24pp-sl-en",
    task=TRANSLATION,
    languages=[SLOVENE],
    source_language=SLOVENE,
    target_language=ENGLISH,
)


# Unofficial datasets ###

EU_MMLU_SL_CONFIG = DatasetConfig(
    name="eu-mmlu-sl",
    pretty_name="EU-MMLU-sl",
    source="EuroEval/eu-mmlu-sl",
    task=KNOW,
    languages=[SLOVENE],
    unofficial=True,
)


FLORES_EN_SL_CONFIG = TranslationDatasetConfig(
    name="flores-en-sl",
    pretty_name="FLORES-en-sl",
    source="EuroEval/flores-en-sl",
    task=TRANSLATION,
    languages=[SLOVENE],
    source_language=ENGLISH,
    target_language=SLOVENE,
    unofficial=True,
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
)
