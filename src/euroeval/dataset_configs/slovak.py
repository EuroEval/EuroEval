"""All Slovak dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import SLOVAK
from ..tasks import (
    COMMON_SENSE,
    HALLU,
    INSTRUCTION_FOLLOWING,
    KNOW,
    LA,
    LOGIC,
    NER,
    RC,
    SENT,
)

# Official datasets ###

CSFD_SENTIMENT_SK_CONFIG = DatasetConfig(
    name="csfd-sentiment-sk",
    pretty_name="CSFD Sentiment SK",
    source="EuroEval/csfd-sentiment-sk-mini",
    task=SENT,
    languages=[SLOVAK],
)

SCALA_SK_CONFIG = DatasetConfig(
    name="scala-sk",
    pretty_name="ScaLA-sk",
    source="EuroEval/scala-sk",
    task=LA,
    languages=[SLOVAK],
)

UNER_SK_CONFIG = DatasetConfig(
    name="uner-sk",
    pretty_name="UNER-sk",
    source="EuroEval/uner-sk-mini",
    task=NER,
    languages=[SLOVAK],
)

MULTI_WIKI_QA_SK_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sk",
    pretty_name="MultiWikiQA-sk",
    source="EuroEval/multi-wiki-qa-sk-mini",
    task=RC,
    languages=[SLOVAK],
)

MMLU_SK_CONFIG = DatasetConfig(
    name="mmlu-sk",
    pretty_name="MMLU-sk",
    source="EuroEval/mmlu-sk-mini",
    task=KNOW,
    languages=[SLOVAK],
)

WINOGRANDE_SK_CONFIG = DatasetConfig(
    name="winogrande-sk",
    pretty_name="Winogrande-sk",
    source="EuroEval/winogrande-sk",
    task=COMMON_SENSE,
    languages=[SLOVAK],
)

MULTI_IFEVAL_SK_CONFIG = DatasetConfig(
    name="multi-ifeval-sk",
    pretty_name="MultiIFEval-sk",
    source="EuroEval/multi-ifeval-sk",
    task=INSTRUCTION_FOLLOWING,
    languages=[SLOVAK],
    train_split=None,
    val_split=None,
)

RAGTRUTH_SK_CONFIG = DatasetConfig(
    name="ragtruth-sk",
    pretty_name="RAGTruth-sk",
    source="EuroEval/ragtruth-translated-hallucinations-sk-mini",
    task=HALLU,
    languages=[SLOVAK],
    train_split=None,
)


ZEBRA_PUZZLE_EASY_SK_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-sk",
    pretty_name="ZebraPuzzlesEasy-sk",
    source="EuroEval/zebra-puzzles-easy-sk",
    task=LOGIC,
    languages=[SLOVAK],
)

# Unofficial datasets ###

EU_MMLU_SK_CONFIG = DatasetConfig(
    name="eu-mmlu-sk",
    pretty_name="EU-MMLU-sk",
    source="EuroEval/eu-mmlu-sk",
    task=KNOW,
    languages=[SLOVAK],
    unofficial=True,
)

ZEBRA_PUZZLE_HARD_SK_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-sk",
    pretty_name="ZebraPuzzlesHard-sk",
    source="EuroEval/zebra-puzzles-hard-sk",
    task=LOGIC,
    languages=[SLOVAK],
    unofficial=True,
)
