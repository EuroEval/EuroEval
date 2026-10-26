"""All Bulgarian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import BULGARIAN, ENGLISH
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

WMT24PP_EN_BG_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-bg",
    pretty_name="WMT24++-en-bg",
    source="EuroEval/wmt24pp-en-bg",
    task=TRANSLATION,
    languages=[BULGARIAN],
    source_language=ENGLISH,
    target_language=BULGARIAN,
    revision="8f9ee2cbedfcf0c7b197d3c64bd272843848df6e",
)

FLORES_EN_BG_CONFIG = TranslationDatasetConfig(
    name="flores-en-bg",
    pretty_name="FLORES-en-bg",
    source="EuroEval/flores-en-bg",
    task=TRANSLATION,
    languages=[BULGARIAN],
    source_language=ENGLISH,
    target_language=BULGARIAN,
    revision="10f0a618a4ece754e26e34f3e9adebca4e65a64a",
)

CINEXIO_CONFIG = DatasetConfig(
    name="cinexio",
    pretty_name="Cinexio",
    source="EuroEval/cinexio-mini",
    task=SENT,
    languages=[BULGARIAN],
    revision="bf886245c3eacf081563cc29da9b2d3cf602b6ec",
)

SCALA_BG_CONFIG = DatasetConfig(
    name="scala-bg",
    pretty_name="ScaLA-bg",
    source="EuroEval/scala-bg",
    task=LA,
    languages=[BULGARIAN],
    revision="bb4bc39ebb68e11da1efdc1936c941ff7ec7d53c",
)

BG_NER_BSNLP_CONFIG = DatasetConfig(
    name="bg-ner-bsnlp",
    pretty_name="BG-NER-BSNLp",
    source="EuroEval/bg-ner-bsnlp-mini",
    task=NER,
    languages=[BULGARIAN],
    revision="4d59400e8422ef75d376aae425fe6106d3895633",
)

MULTI_WIKI_QA_BG_CONFIG = DatasetConfig(
    name="multi-wiki-qa-bg",
    pretty_name="MultiWikiQA-bg",
    source="EuroEval/multi-wiki-qa-bg-mini",
    task=RC,
    languages=[BULGARIAN],
    revision="12868c70f50620b81bfc6df2337bfb8041cf7bf5",
)

EXAMS_BG_CONFIG = DatasetConfig(
    name="exams-bg",
    pretty_name="Exams-bg",
    source="EuroEval/exams-bg-mini",
    task=KNOW,
    languages=[BULGARIAN],
    revision="b9c08d316777f273321c2bee756aca2c5cf1a516",
)

WINOGRANDE_BG_CONFIG = DatasetConfig(
    name="winogrande-bg",
    pretty_name="Winogrande-bg",
    source="EuroEval/winogrande-bg",
    task=COMMON_SENSE,
    languages=[BULGARIAN],
    labels=["a", "b"],
    revision="efb6d62210e58294ccb49bbd88c61f3c42b6c487",
)

MULTI_IFEVAL_BG_CONFIG = DatasetConfig(
    name="multi-ifeval-bg",
    pretty_name="MultiIFEval-bg",
    source="EuroEval/multi-ifeval-bg",
    task=INSTRUCTION_FOLLOWING,
    languages=[BULGARIAN],
    train_split=None,
    val_split=None,
    revision="bbca89003160727dcd925542ce5049909966c35e",
)

RAGTRUTH_BG_CONFIG = DatasetConfig(
    name="ragtruth-bg",
    pretty_name="RAGTruth-bg",
    source="EuroEval/ragtruth-translated-hallucinations-bg-mini",
    task=HALLU,
    languages=[BULGARIAN],
    train_split=None,
    revision="75b15c73a28a4b9eb8ed53d61faab052a66a52f6",
)


# Unofficial datasets ###

MULTI_BFCL_BG_CONFIG = DatasetConfig(
    name="multi-bfcl-bg",
    pretty_name="MultiBFCL-bg",
    source="EuroEval/multi-bfcl-bg-mini",
    task=TOOL_CALLING,
    languages=[BULGARIAN],
    train_split=None,
    revision="b545ee875a9b3907a40de063b52470d9138472a8",
)


WMT24PP_BG_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-bg-en",
    pretty_name="WMT24++-bg-en",
    source="EuroEval/wmt24pp-bg-en",
    task=TRANSLATION,
    languages=[BULGARIAN],
    source_language=BULGARIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="2cef823d27d844c5970bc5bc245256318b85de83",
)

FLORES_BG_EN_CONFIG = TranslationDatasetConfig(
    name="flores-bg-en",
    pretty_name="FLORES-bg-en",
    source="EuroEval/flores-bg-en",
    task=TRANSLATION,
    languages=[BULGARIAN],
    source_language=BULGARIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="9548c62d8241a6a5af3c84a7cbaa7368547adaa8",
)

INCLUDE_BG_CONFIG = DatasetConfig(
    name="include-bg",
    pretty_name="INCLUDE-bg",
    source="EuroEval/include-bg-mini",
    task=KNOW,
    languages=[BULGARIAN],
    unofficial=True,
    revision="9348f952d29e0327b1c5c58bcdde7ce5ba63bb0a",
)
