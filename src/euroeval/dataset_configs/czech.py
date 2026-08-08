"""All Czech dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import CZECH
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
    SUMM,
)

# Official datasets ###

CSFD_SENTIMENT_CONFIG = DatasetConfig(
    name="csfd-sentiment",
    pretty_name="CSFD Sentiment",
    source="EuroEval/csfd-sentiment-mini",
    task=SENT,
    languages=[CZECH],
)

CS_GEC_CONFIG = DatasetConfig(
    name="cs-gec",
    pretty_name="CS-GEC",
    source="EuroEval/cs-gec-mini",
    task=LA,
    languages=[CZECH],
)

PONER_CONFIG = DatasetConfig(
    name="poner",
    pretty_name="PoNER",
    source="EuroEval/poner-mini",
    task=NER,
    languages=[CZECH],
)

SQAD_CONFIG = DatasetConfig(
    name="sqad",
    pretty_name="SQAD",
    source="EuroEval/sqad-mini",
    task=RC,
    languages=[CZECH],
)

CZECH_NEWS_CONFIG = DatasetConfig(
    name="czech-news",
    pretty_name="Czech News",
    source="EuroEval/czech-news-mini",
    task=SUMM,
    languages=[CZECH],
)

UMIMETO_QA_CONFIG = DatasetConfig(
    name="umimeto-qa",
    pretty_name="Umimeto QA",
    source="EuroEval/umimeto-qa",
    task=KNOW,
    languages=[CZECH],
)

HELLASWAG_CS_CONFIG = DatasetConfig(
    name="hellaswag-cs",
    pretty_name="HellaSwag-cs",
    source="EuroEval/hellaswag-cs-mini",
    task=COMMON_SENSE,
    languages=[CZECH],
)

MULTI_IFEVAL_CS_CONFIG = DatasetConfig(
    name="multi-ifeval-cs",
    pretty_name="MultiIFEval-cs",
    source="EuroEval/multi-ifeval-cs",
    task=INSTRUCTION_FOLLOWING,
    languages=[CZECH],
    train_split=None,
    val_split=None,
)

RAGTRUTH_CS_CONFIG = DatasetConfig(
    name="ragtruth-cs",
    pretty_name="RAGTruth-cs",
    source="EuroEval/ragtruth-translated-hallucinations-cs-mini",
    task=HALLU,
    languages=[CZECH],
    train_split=None,
)


ZEBRA_PUZZLE_EASY_CS_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-cs",
    pretty_name="ZebraPuzzlesEasy-cs",
    source="EuroEval/zebra-puzzles-easy-cs",
    task=LOGIC,
    languages=[CZECH],
)

# Unofficial datasets ###

SCALA_CS_CONFIG = DatasetConfig(
    name="scala-cs",
    pretty_name="ScaLA-cs",
    source="EuroEval/scala-cs",
    task=LA,
    languages=[CZECH],
    unofficial=True,
)

EU_MMLU_CS_CONFIG = DatasetConfig(
    name="eu-mmlu-cs",
    pretty_name="EU-MMLU-cs",
    source="EuroEval/eu-mmlu-cs",
    task=KNOW,
    languages=[CZECH],
    unofficial=True,
)

ZEBRA_PUZZLE_HARD_CS_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-cs",
    pretty_name="ZebraPuzzlesHard-cs",
    source="EuroEval/zebra-puzzles-hard-cs",
    task=LOGIC,
    languages=[CZECH],
    unofficial=True,
)
