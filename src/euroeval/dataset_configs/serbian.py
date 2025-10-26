"""All Serbian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import SERBIAN
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

MMS_SR_CONFIG = DatasetConfig(
    name="mms-sr", source="EuroEval/mms-sr-mini", task=SENT, languages=[SERBIAN]
)

SCALA_SR_CONFIG = DatasetConfig(
    name="scala-sr", source="EuroEval/scala-sr", task=LA, languages=[SERBIAN]
)

UNER_SR_CONFIG = DatasetConfig(
    name="uner-sr", source="EuroEval/uner-sr-mini", task=NER, languages=[SERBIAN]
)

MULTI_WIKI_QA_SR_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sr",
    source="EuroEval/multi-wiki-qa-sr-mini",
    task=RC,
    languages=[SERBIAN],
)

LR_SUM_SR_CONFIG = DatasetConfig(
    name="lr-sum-sr", source="EuroEval/lr-sum-sr-mini", task=SUMM, languages=[SERBIAN]
)

MMLU_SR_CONFIG = DatasetConfig(
    name="mmlu-sr", source="EuroEval/mmlu-sr-mini", task=KNOW, languages=[SERBIAN]
)

WINOGRANDE_SR_CONFIG = DatasetConfig(
    name="winogrande-sr",
    source="EuroEval/winogrande-sr",
    task=COMMON_SENSE,
    languages=[SERBIAN],
    _labels=["a", "b"],
)
