"""All Lithuanian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import ENGLISH, LITHUANIAN
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

WMT24PP_EN_LT_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-en-lt",
    pretty_name="WMT24++-en-lt",
    source="EuroEval/wmt24pp-en-lt",
    task=TRANSLATION,
    languages=[LITHUANIAN],
    source_language=ENGLISH,
    target_language=LITHUANIAN,
    revision="0b21c8bf74beff8aee3022887c78772801b8340b",
)

FLORES_EN_LT_CONFIG = TranslationDatasetConfig(
    name="flores-en-lt",
    pretty_name="FLORES-en-lt",
    source="EuroEval/flores-en-lt",
    task=TRANSLATION,
    languages=[LITHUANIAN],
    source_language=ENGLISH,
    target_language=LITHUANIAN,
    revision="cb2fdb73d5df436ed52032d255dc9dfd76adbca2",
)

ATSILIEPIMAI_CONFIG = DatasetConfig(
    name="atsiliepimai",
    pretty_name="Atsiliepimai",
    source="EuroEval/atsiliepimai",
    task=SENT,
    languages=[LITHUANIAN],
    revision="d19902d48f102db6a7219dd1b5fcd8b00d358a53",
)

SCALA_LT_CONFIG = DatasetConfig(
    name="scala-lt",
    pretty_name="ScaLA-lt",
    source="EuroEval/scala-lt",
    task=LA,
    languages=[LITHUANIAN],
    revision="580103f697144f3d95aa7a934e6ff2ae63a40be0",
)

WIKIANN_LT_CONFIG = DatasetConfig(
    name="wikiann-lt",
    pretty_name="WikiANN-lt",
    source="EuroEval/wikiann-lt-mini",
    task=NER,
    languages=[LITHUANIAN],
    revision="4a4a69de20d4bd7671f62a1eafb23aa4648af9c3",
)

MULTI_WIKI_QA_LT_CONFIG = DatasetConfig(
    name="multi-wiki-qa-lt",
    pretty_name="MultiWikiQA-lt",
    source="EuroEval/multi-wiki-qa-lt-mini",
    task=RC,
    languages=[LITHUANIAN],
    revision="f957457c6bd91e38bf9036a13401cdd92c699635",
)

LRYTAS_CONFIG = DatasetConfig(
    name="lrytas",
    pretty_name="Lrytas",
    source="EuroEval/lrytas-mini",
    task=SUMM,
    languages=[LITHUANIAN],
    revision="27d113ec030fdcf8da56fc4fcdb473204bc5f48b",
)

LT_HISTORY_CONFIG = DatasetConfig(
    name="lt-history",
    pretty_name="LT-History",
    source="EuroEval/lt-history",
    task=KNOW,
    languages=[LITHUANIAN],
    revision="f8f4d4fd38f872d4beb3d9082e696130bf3a219d",
)

WINOGRANDE_LT_CONFIG = DatasetConfig(
    name="winogrande-lt",
    pretty_name="Winogrande-lt",
    source="EuroEval/winogrande-lt",
    task=COMMON_SENSE,
    languages=[LITHUANIAN],
    labels=["a", "b"],
    revision="197e366ff2968d4faff214c713fef8457c7efcac",
)

MULTI_IFEVAL_LT_CONFIG = DatasetConfig(
    name="multi-ifeval-lt",
    pretty_name="MultiIFEval-lt",
    source="EuroEval/multi-ifeval-lt",
    task=INSTRUCTION_FOLLOWING,
    languages=[LITHUANIAN],
    train_split=None,
    val_split=None,
    revision="cb31305b8a19f2b7df844917246d03ae23d4ebb3",
)

RAGTRUTH_LT_CONFIG = DatasetConfig(
    name="ragtruth-lt",
    pretty_name="RAGTruth-lt",
    source="EuroEval/ragtruth-translated-hallucinations-lt-mini",
    task=HALLU,
    languages=[LITHUANIAN],
    train_split=None,
    revision="fac924fcd68db74671062147ac5d964d945a082e",
)

INCLUDE_LT_CONFIG = DatasetConfig(
    name="include-lt",
    pretty_name="INCLUDE-lt",
    source="EuroEval/include-lt-mini",
    task=KNOW,
    languages=[LITHUANIAN],
    revision="ff2aa9e2bca7d1c755f238386e2c55a497ed71bb",
)


# Unofficial datasets ###

MULTI_BFCL_LT_CONFIG = DatasetConfig(
    name="multi-bfcl-lt",
    pretty_name="MultiBFCL-lt",
    source="EuroEval/multi-bfcl-lt-mini",
    task=TOOL_CALLING,
    languages=[LITHUANIAN],
    train_split=None,
    revision="19c5f41a808e2d0a64e4a5cf68ee0cb278c96348",
)


WMT24PP_LT_EN_CONFIG = TranslationDatasetConfig(
    name="wmt24pp-lt-en",
    pretty_name="WMT24++-lt-en",
    source="EuroEval/wmt24pp-lt-en",
    task=TRANSLATION,
    languages=[LITHUANIAN],
    source_language=LITHUANIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="2a9237cf5e014ca4f1096d9a20eadbdf09ab8d82",
)

FLORES_LT_EN_CONFIG = TranslationDatasetConfig(
    name="flores-lt-en",
    pretty_name="FLORES-lt-en",
    source="EuroEval/flores-lt-en",
    task=TRANSLATION,
    languages=[LITHUANIAN],
    source_language=LITHUANIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="052ae9d1c13c8571b1edad54ce4fa0b5684b6222",
)

LITHUANIAN_EMOTIONS_CONFIG = DatasetConfig(
    name="lithuanian-emotions",
    pretty_name="Lithuanian Emotions",
    source="EuroEval/lithuanian-emotions-mini",
    task=SENT,
    languages=[LITHUANIAN],
    unofficial=True,
    revision="31bbc85ef232cf285c26e403dfc30f7b2f960f96",
)

EU_MMLU_LT_CONFIG = DatasetConfig(
    name="eu-mmlu-lt",
    pretty_name="EU-MMLU-lt",
    source="EuroEval/eu-mmlu-lt",
    task=KNOW,
    languages=[LITHUANIAN],
    unofficial=True,
    revision="cad4ade3f456d8a0c363126897d70fd818e02acc",
)
