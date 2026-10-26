"""All Hungarian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, HUNGARIAN
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

WMT24PP_EN_HU_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-hu",
    pretty_name="WMT24++-en-hu",
    source="EuroEval/wmt24pp-en-hu",
    task=TRANSLATION,
    languages=[HUNGARIAN],
    source_language=ENGLISH,
    target_language=HUNGARIAN,
    revision="07776e0d07f11ab0c8d22e4f3da82deeef2b1feb",
)

FLORES_EN_HU_CONFIG = TranslationDatasetConfig(
    name="flores-en-hu",
    pretty_name="FLORES-en-hu",
    source="EuroEval/flores-en-hu",
    task=TRANSLATION,
    languages=[HUNGARIAN],
    source_language=ENGLISH,
    target_language=HUNGARIAN,
    revision="3ddcbf6222574fba116a52f0c3974ecb9c52d7b0",
)

HUSST_CONFIG = DatasetConfig(
    name="husst",
    pretty_name="HuSST",
    source="EuroEval/husst-mini",
    task=SENT,
    languages=[HUNGARIAN],
    revision="42ea10979c3e99497a358fc0a3ced9d53635e052",
)

SCALA_HU_CONFIG = DatasetConfig(
    name="scala-hu",
    pretty_name="ScaLA-hu",
    source="EuroEval/scala-hu",
    task=LA,
    languages=[HUNGARIAN],
    revision="9a0a7532b2786e87c6f91fd639f6d4aabc360a08",
)

SZEGED_NER_CONFIG = DatasetConfig(
    name="szeged-ner",
    pretty_name="SzegedNER",
    source="EuroEval/szeged-ner",
    task=NER,
    languages=[HUNGARIAN],
    revision="6c81e41b0aeec5b07071707b1b9bc78916d1da3d",
)

MULTI_WIKI_QA_HU_CONFIG = DatasetConfig(
    name="multi-wiki-qa-hu",
    pretty_name="MultiWikiQA-hu",
    source="EuroEval/multi-wiki-qa-hu-mini",
    task=RC,
    languages=[HUNGARIAN],
    revision="5b268f6d5a32851d0b702898c1e87cda9a823682",
)

HUNSUM_CONFIG = DatasetConfig(
    name="hunsum",
    pretty_name="HunSum",
    source="EuroEval/hun-sum-mini",
    task=SUMM,
    languages=[HUNGARIAN],
    revision="4eee98a3e7c140d6015eef930b9879be5436b311",
)

WINOGRANDE_HU_CONFIG = DatasetConfig(
    name="winogrande-hu",
    pretty_name="Winogrande-hu",
    source="EuroEval/winogrande-hu",
    task=COMMON_SENSE,
    languages=[HUNGARIAN],
    labels=["a", "b"],
    revision="333f75dcc917b8c76c98418d8c594181b89bb040",
)

MULTI_IFEVAL_HU_CONFIG = DatasetConfig(
    name="multi-ifeval-hu",
    pretty_name="MultiIFEval-hu",
    source="EuroEval/multi-ifeval-hu",
    task=INSTRUCTION_FOLLOWING,
    languages=[HUNGARIAN],
    train_split=None,
    val_split=None,
    revision="1e8328a4a27b274e6949151271aba4e0ed44196f",
)

RAGTRUTH_HU_CONFIG = DatasetConfig(
    name="ragtruth-hu",
    pretty_name="RAGTruth-hu",
    source="EuroEval/ragtruth-translated-hallucinations-hu-mini",
    task=HALLU,
    languages=[HUNGARIAN],
    train_split=None,
    revision="301f03ac4a4ec1163d9eb04d8c5df54a6712a630",
)


INCLUDE_HU_CONFIG = DatasetConfig(
    name="include-hu",
    pretty_name="INCLUDE-hu",
    source="EuroEval/include-hu-mini",
    task=KNOW,
    languages=[HUNGARIAN],
    revision="1fd12a4b7b8dfda17fba9cdb38ab10824703b28c",
)


# Unofficial datasets ###

MULTI_BFCL_HU_CONFIG = DatasetConfig(
    name="multi-bfcl-hu",
    pretty_name="MultiBFCL-hu",
    source="EuroEval/multi-bfcl-hu-mini",
    task=TOOL_CALLING,
    languages=[HUNGARIAN],
    train_split=None,
    revision="709f59877930d14f2d651abc88b834fabdee2e7c",
)


WMT24PP_HU_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-hu-en",
    pretty_name="WMT24++-hu-en",
    source="EuroEval/wmt24pp-hu-en",
    task=TRANSLATION,
    languages=[HUNGARIAN],
    source_language=HUNGARIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="763be675c78730f5bf725c3e482c88df80e67f66",
)

FLORES_HU_EN_CONFIG = TranslationDatasetConfig(
    name="flores-hu-en",
    pretty_name="FLORES-hu-en",
    source="EuroEval/flores-hu-en",
    task=TRANSLATION,
    languages=[HUNGARIAN],
    source_language=HUNGARIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="a40b16c40aab180834833c6802117165cc3cb412",
)

MMLU_HU_CONFIG = DatasetConfig(
    name="mmlu-hu",
    pretty_name="MMLU-hu",
    source="EuroEval/mmlu-hu-mini",
    task=KNOW,
    languages=[HUNGARIAN],
    unofficial=True,
    revision="e1a40897ee80172bc0f10f9a3dfe9eb13ac8133a",
)

EU_MMLU_HU_CONFIG = DatasetConfig(
    name="eu-mmlu-hu",
    pretty_name="EU-MMLU-hu",
    source="EuroEval/eu-mmlu-hu",
    task=KNOW,
    languages=[HUNGARIAN],
    unofficial=True,
    revision="2209e2ae9ac1add5efec7c000d0a95ada18c295c",
)
