"""All Bosnian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import BOSNIAN
from ..tasks import HALLU, INSTRUCTION_FOLLOWING, LOGIC, NER, RC, SENT, SUMM

# Official datasets ###

MMS_BS_CONFIG = DatasetConfig(
    name="mms-bs",
    pretty_name="MMS-bs",
    source="EuroEval/mms-bs-mini",
    task=SENT,
    languages=[BOSNIAN],
)

WIKIANN_BS_CONFIG = DatasetConfig(
    name="wikiann-bs",
    pretty_name="WikiANN-bs",
    source="EuroEval/wikiann-bs-mini",
    task=NER,
    languages=[BOSNIAN],
)

MULTI_WIKI_QA_BS_CONFIG = DatasetConfig(
    name="multi-wiki-qa-bs",
    pretty_name="MultiWikiQA-bs",
    source="EuroEval/multi-wiki-qa-bs-mini",
    task=RC,
    languages=[BOSNIAN],
)

LR_SUM_BS_CONFIG = DatasetConfig(
    name="lr-sum-bs",
    pretty_name="LRSum-bs",
    source="EuroEval/lr-sum-bs-mini",
    task=SUMM,
    languages=[BOSNIAN],
)

MULTI_IFEVAL_BS_CONFIG = DatasetConfig(
    name="multi-ifeval-bs",
    pretty_name="MultiIFEval-bs",
    source="EuroEval/multi-ifeval-bs",
    task=INSTRUCTION_FOLLOWING,
    languages=[BOSNIAN],
    train_split=None,
    val_split=None,
)

RAGTRUTH_BS_CONFIG = DatasetConfig(
    name="ragtruth-bs",
    pretty_name="RAGTruth-bs",
    source="EuroEval/ragtruth-translated-hallucinations-bs-mini",
    task=HALLU,
    languages=[BOSNIAN],
    train_split=None,
)

ZEBRA_PUZZLE_EASY_BS_CONFIG = DatasetConfig(
    name="zebra-puzzles-easy-bs",
    pretty_name="ZebraPuzzlesEasy-bs",
    source="EuroEval/zebra-puzzles-easy-bs",
    task=LOGIC,
    languages=[BOSNIAN],
)

# Unofficial datasets ###

ZEBRA_PUZZLE_HARD_BS_CONFIG = DatasetConfig(
    name="zebra-puzzles-hard-bs",
    pretty_name="ZebraPuzzlesHard-bs",
    source="EuroEval/zebra-puzzles-hard-bs",
    task=LOGIC,
    languages=[BOSNIAN],
    unofficial=True,
)
