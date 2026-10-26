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
    TOOL_CALLING,
    TRANSLATION,
)

# Official datasets ###

WMT24PP_EN_HR_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-hr",
    pretty_name="WMT24++-en-hr",
    source="EuroEval/wmt24pp-en-hr",
    task=TRANSLATION,
    languages=[CROATIAN],
    source_language=ENGLISH,
    target_language=CROATIAN,
    revision="7b479937045cfb9d04c5d5f08abc54e42583426e",
)

FLORES_EN_HR_CONFIG = TranslationDatasetConfig(
    name="flores-en-hr",
    pretty_name="FLORES-en-hr",
    source="EuroEval/flores-en-hr",
    task=TRANSLATION,
    languages=[CROATIAN],
    source_language=ENGLISH,
    target_language=CROATIAN,
    revision="87461bea7b54485e1d7c4cb0dc93d73e664d3c6d",
)

MMS_HR_CONFIG = DatasetConfig(
    name="mms-hr",
    pretty_name="MMS-hr",
    source="EuroEval/mms-hr-mini",
    task=SENT,
    languages=[CROATIAN],
    revision="b5062b95128ec6d0a2aa132287142806e6356b68",
)

SCALA_HR_CONFIG = DatasetConfig(
    name="scala-hr",
    pretty_name="ScaLA-hr",
    source="EuroEval/scala-hr",
    task=LA,
    languages=[CROATIAN],
    revision="f4253598b4b559de42dc589d7a52b3d17b69bc93",
)

WIKIANN_HR_CONFIG = DatasetConfig(
    name="wikiann-hr",
    pretty_name="WikiANN-hr",
    source="EuroEval/wikiann-hr-mini",
    task=NER,
    languages=[CROATIAN],
    revision="141c07309ac7f4305d8513d866ef4aa2c1560abf",
)

MULTI_WIKI_QA_HR_CONFIG = DatasetConfig(
    name="multi-wiki-qa-hr",
    pretty_name="MultiWikiQA-hr",
    source="EuroEval/multi-wiki-qa-hr-mini",
    task=RC,
    languages=[CROATIAN],
    revision="05ddffe357f7daa7b1955f4dc27c871f6c064f94",
)

WINOGRANDE_HR_CONFIG = DatasetConfig(
    name="winogrande-hr",
    pretty_name="Winogrande-hr",
    source="EuroEval/winogrande-hr",
    task=COMMON_SENSE,
    languages=[CROATIAN],
    labels=["a", "b"],
    revision="c0715ac7ea6abff641b2283a2650ca2a417c95e5",
)

MULTI_IFEVAL_HR_CONFIG = DatasetConfig(
    name="multi-ifeval-hr",
    pretty_name="MultiIFEval-hr",
    source="EuroEval/multi-ifeval-hr",
    task=INSTRUCTION_FOLLOWING,
    languages=[CROATIAN],
    train_split=None,
    val_split=None,
    revision="45db5844205e03c683c82e4a4f76b9cd1ba067ef",
)

RAGTRUTH_HR_CONFIG = DatasetConfig(
    name="ragtruth-hr",
    pretty_name="RAGTruth-hr",
    source="EuroEval/ragtruth-translated-hallucinations-hr-mini",
    task=HALLU,
    languages=[CROATIAN],
    train_split=None,
    revision="9876cc127057efcacf2f45c72171e9446c958e7e",
)


INCLUDE_HR_CONFIG = DatasetConfig(
    name="include-hr",
    pretty_name="INCLUDE-hr",
    source="EuroEval/include-hr-mini",
    task=KNOW,
    languages=[CROATIAN],
    revision="3adafb91867d14369dea6ba9f71f799d2f86107f",
)


# Unofficial datasets ###

MULTI_BFCL_HR_CONFIG = DatasetConfig(
    name="multi-bfcl-hr",
    pretty_name="MultiBFCL-hr",
    source="EuroEval/multi-bfcl-hr-mini",
    task=TOOL_CALLING,
    languages=[CROATIAN],
    train_split=None,
    revision="ee220830dc8141a3c70c177d48308b7e1f2cb36f",
)


WMT24PP_HR_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-hr-en",
    pretty_name="WMT24++-hr-en",
    source="EuroEval/wmt24pp-hr-en",
    task=TRANSLATION,
    languages=[CROATIAN],
    source_language=CROATIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="009791163b5852a1030407160ae6d1072602e2bc",
)

FLORES_HR_EN_CONFIG = TranslationDatasetConfig(
    name="flores-hr-en",
    pretty_name="FLORES-hr-en",
    source="EuroEval/flores-hr-en",
    task=TRANSLATION,
    languages=[CROATIAN],
    source_language=CROATIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="bb95f29a599a3ef7ee084fe736bdf065b1001cbd",
)

MMLU_HR_CONFIG = DatasetConfig(
    name="mmlu-hr",
    pretty_name="MMLU-hr",
    source="EuroEval/mmlu-hr-mini",
    task=KNOW,
    languages=[CROATIAN],
    unofficial=True,
    revision="cf0fdc0259ab483d7e4ed25c0ed37da250d0a0f2",
)

EU_MMLU_HR_CONFIG = DatasetConfig(
    name="eu-mmlu-hr",
    pretty_name="EU-MMLU-hr",
    source="EuroEval/eu-mmlu-hr",
    task=KNOW,
    languages=[CROATIAN],
    unofficial=True,
    revision="1f76090f5bc859e3bd7bc98b24ee4c174658db63",
)
