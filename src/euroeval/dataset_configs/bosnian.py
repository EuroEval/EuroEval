"""All Bosnian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, TranslationDatasetConfig
from ..languages import BOSNIAN, ENGLISH
from ..tasks import (
    HALLU,
    INSTRUCTION_FOLLOWING,
    NER,
    RC,
    SENT,
    SUMM,
    TOOL_CALLING,
    TRANSLATION,
)

# Official datasets ###

FLORES_EN_BS_CONFIG = TranslationDatasetConfig(
    name="flores-en-bs",
    pretty_name="FLORES-en-bs",
    source="EuroEval/flores-en-bs",
    task=TRANSLATION,
    languages=[BOSNIAN],
    source_language=ENGLISH,
    target_language=BOSNIAN,
    revision="f758c855df0e8051a437c871491e2e6ee8e004e0",
)

MMS_BS_CONFIG = DatasetConfig(
    name="mms-bs",
    pretty_name="MMS-bs",
    source="EuroEval/mms-bs-mini",
    task=SENT,
    languages=[BOSNIAN],
    revision="e6c181c64c7f51e69f2e7b1768a235a11eb35d5b",
)

WIKIANN_BS_CONFIG = DatasetConfig(
    name="wikiann-bs",
    pretty_name="WikiANN-bs",
    source="EuroEval/wikiann-bs-mini",
    task=NER,
    languages=[BOSNIAN],
    revision="d7b87068fdaabe5945db97314610f15d674f53d3",
)

MULTI_WIKI_QA_BS_CONFIG = DatasetConfig(
    name="multi-wiki-qa-bs",
    pretty_name="MultiWikiQA-bs",
    source="EuroEval/multi-wiki-qa-bs-mini",
    task=RC,
    languages=[BOSNIAN],
    revision="c24bc03a9d5a3a543469ec425f748bdc966b5da5",
)

LR_SUM_BS_CONFIG = DatasetConfig(
    name="lr-sum-bs",
    pretty_name="LRSum-bs",
    source="EuroEval/lr-sum-bs-mini",
    task=SUMM,
    languages=[BOSNIAN],
    revision="4d58181f3b6b249b6943fb1044480a7a9c7703d2",
)

MULTI_IFEVAL_BS_CONFIG = DatasetConfig(
    name="multi-ifeval-bs",
    pretty_name="MultiIFEval-bs",
    source="EuroEval/multi-ifeval-bs",
    task=INSTRUCTION_FOLLOWING,
    languages=[BOSNIAN],
    train_split=None,
    val_split=None,
    revision="835cedc296b844c2632c63753ce80c0bb6547518",
)

RAGTRUTH_BS_CONFIG = DatasetConfig(
    name="ragtruth-bs",
    pretty_name="RAGTruth-bs",
    source="EuroEval/ragtruth-translated-hallucinations-bs-mini",
    task=HALLU,
    languages=[BOSNIAN],
    train_split=None,
    revision="86368bbbf54bf8530528758a0b6526ab4ef9a989",
)


# Unofficial datasets ###

MULTI_BFCL_BS_CONFIG = DatasetConfig(
    name="multi-bfcl-bs",
    pretty_name="MultiBFCL-bs",
    source="EuroEval/multi-bfcl-bs-mini",
    task=TOOL_CALLING,
    languages=[BOSNIAN],
    train_split=None,
    revision="bf46dea9bc614d9eb7cf8b3245c8267f35e6decd",
)


FLORES_BS_EN_CONFIG = TranslationDatasetConfig(
    name="flores-bs-en",
    pretty_name="FLORES-bs-en",
    source="EuroEval/flores-bs-en",
    task=TRANSLATION,
    languages=[BOSNIAN],
    source_language=BOSNIAN,
    target_language=ENGLISH,
    unofficial=True,
    revision="29159100ff5ee1d1eed6dbc3571996e97eb0bd23",
)
