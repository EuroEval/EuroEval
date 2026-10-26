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
    TOOL_CALLING,
    TRANSLATION,
)

# Official datasets ###

WMT24PP_EN_UK_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-uk",
    pretty_name="WMT24++-en-uk",
    source="EuroEval/wmt24pp-en-uk",
    task=TRANSLATION,
    languages=[UKRAINIAN],
    source_language=ENGLISH,
    target_language=UKRAINIAN,
    revision="eb89bf96875f8ab140692f5a891146b8d6bde649",
)

FLORES_EN_UK_CONFIG = TranslationDatasetConfig(
    name="flores-en-uk",
    pretty_name="FLORES-en-uk",
    source="EuroEval/flores-en-uk",
    task=TRANSLATION,
    languages=[UKRAINIAN],
    source_language=ENGLISH,
    target_language=UKRAINIAN,
    revision="af6bcd3a384b7a4b8a19d7a40c5d38980bd85851",
)

CROSS_DOMAIN_UK_REVIEWS_CONFIG = DatasetConfig(
    name="cross-domain-uk-reviews",
    pretty_name="Cross Domain Ukrainian Reviews",
    source="EuroEval/cross-domain-uk-reviews-mini",
    task=SENT,
    languages=[UKRAINIAN],
    revision="f2e9af65f9fb79a5f7422903610e88ef9cf395ad",
)

SCALA_UK_CONFIG = DatasetConfig(
    name="scala-uk",
    pretty_name="ScaLA-uk",
    source="EuroEval/scala-uk",
    task=LA,
    languages=[UKRAINIAN],
    revision="b1b4b3fbdc324c1df09a952d9f3897accd7a0750",
)

NER_UK_CONFIG = DatasetConfig(
    name="ner-uk",
    pretty_name="NER-uk",
    source="EuroEval/ner-uk-mini",
    task=NER,
    languages=[UKRAINIAN],
    revision="c748fc324e19a49d8e9d74413f68b7b44acb340a",
)

MULTI_WIKI_QA_UK_CONFIG = DatasetConfig(
    name="multi-wiki-qa-uk",
    pretty_name="MultiWikiQA-uk",
    source="EuroEval/multi-wiki-qa-uk-mini",
    task=RC,
    languages=[UKRAINIAN],
    revision="c2e4878def737ca442706eece9b29d9fc73ee9f2",
)

LR_SUM_UK_CONFIG = DatasetConfig(
    name="lr-sum-uk",
    pretty_name="LRSum-uk",
    source="EuroEval/lr-sum-uk-mini",
    task=SUMM,
    languages=[UKRAINIAN],
    revision="033c9e302a524f11d42d9264366ddb0f610fd9de",
)

WINOGRANDE_UK_CONFIG = DatasetConfig(
    name="winogrande-uk",
    pretty_name="Winogrande-uk",
    source="EuroEval/winogrande-uk",
    task=COMMON_SENSE,
    languages=[UKRAINIAN],
    labels=["a", "b"],
    revision="4a94d19924a8d1e1d1285e09d881b8c8413dd2ec",
)

MULTI_IFEVAL_UK_CONFIG = DatasetConfig(
    name="multi-ifeval-uk",
    pretty_name="MultiIFEval-uk",
    source="EuroEval/multi-ifeval-uk",
    task=INSTRUCTION_FOLLOWING,
    languages=[UKRAINIAN],
    train_split=None,
    val_split=None,
    revision="7c8f6918f1c82f772652dd51d88f95e2152beb8c",
)

RAGTRUTH_UK_CONFIG = DatasetConfig(
    name="ragtruth-uk",
    pretty_name="RAGTruth-uk",
    source="EuroEval/ragtruth-translated-hallucinations-uk-mini",
    task=HALLU,
    languages=[UKRAINIAN],
    train_split=None,
    revision="5d2aa7ccb63dad4d9b8268e8ff169cde1d47a6f9",
)


INCLUDE_UK_CONFIG = DatasetConfig(
    name="include-uk",
    pretty_name="INCLUDE-uk",
    source="EuroEval/include-uk-mini",
    task=KNOW,
    languages=[UKRAINIAN],
    revision="f5db25dc73fb69902ef99cffda7ae98dc48f2b6d",
)


# Unofficial datasets ###

MULTI_BFCL_UK_CONFIG = DatasetConfig(
    name="multi-bfcl-uk",
    pretty_name="MultiBFCL-uk",
    source="EuroEval/multi-bfcl-uk-mini",
    task=TOOL_CALLING,
    languages=[UKRAINIAN],
    train_split=None,
    revision="d99a99f51a356eff1caa80dc29a14f78878a1c08",
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
    revision="ff651ad28a024d49c508fc09131f1168df3070d4",
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
    revision="ca89619a966ca80d2c20c26fbc254ff7765043d3",
)

GLOBAL_MMLU_UK_CONFIG = DatasetConfig(
    name="global-mmlu-uk",
    pretty_name="GlobalMMLU-uk",
    source="EuroEval/global-mmlu-uk-mini",
    task=KNOW,
    languages=[UKRAINIAN],
    unofficial=True,
    revision="81fc650f4612f9151eb38d515a70339cbf27d98f",
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
    revision="52af20cafef686b45dc303e1e72f77b9eccc296f",
)
