"""All Serbian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import SERBIAN
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

MMS_SR_CONFIG = DatasetConfig(
    name="mms-sr",
    pretty_name="MMS-sr",
    source="EuroEval/mms-sr-mini",
    task=SENT,
    languages=[SERBIAN],
)

SCALA_SR_CONFIG = DatasetConfig(
    name="scala-sr",
    pretty_name="ScaLA-sr",
    source="EuroEval/scala-sr",
    task=LA,
    languages=[SERBIAN],
)

UNER_SR_CONFIG = DatasetConfig(
    name="uner-sr",
    pretty_name="UNER-sr",
    source="EuroEval/uner-sr-mini",
    task=NER,
    languages=[SERBIAN],
)

MULTI_WIKI_QA_SR_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sr",
    pretty_name="MultiWikiQA-sr",
    source="EuroEval/multi-wiki-qa-sr-mini",
    task=RC,
    languages=[SERBIAN],
)

LR_SUM_SR_CONFIG = DatasetConfig(
    name="lr-sum-sr",
    pretty_name="LRSum-sr",
    source="EuroEval/lr-sum-sr-mini",
    task=SUMM,
    languages=[SERBIAN],
)

MMLU_SR_CONFIG = DatasetConfig(
    name="mmlu-sr",
    pretty_name="MMLU-sr",
    source="EuroEval/mmlu-sr-mini",
    task=KNOW,
    languages=[SERBIAN],
)

WINOGRANDE_SR_CONFIG = DatasetConfig(
    name="winogrande-sr",
    pretty_name="Winogrande-sr",
    source="EuroEval/winogrande-sr",
    task=COMMON_SENSE,
    languages=[SERBIAN],
    _labels=["a", "b"],
)
