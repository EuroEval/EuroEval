"""All Serbian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import SR
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT, SUMM

### Official datasets ###

MMS_SR_CONFIG = DatasetConfig(
    name="mms-sr",
    pretty_name="the truncated version of the Serbian part of the MMS sentiment classification "
    "dataset MMS-sr",
    huggingface_id="EuroEval/mms-sr-mini",
    task=SENT,
    languages=[SR],
)

SCALA_SR_CONFIG = DatasetConfig(
    name="scala-sr",
    pretty_name="the Serbian part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-sr",
    task=LA,
    languages=[SR],
)

UNER_SR_CONFIG = DatasetConfig(
    name="uner-sr",
    pretty_name="the truncated version of the Serbian named entity recognition dataset "
    "UNER-sr",
    huggingface_id="EuroEval/uner-sr-mini",
    task=NER,
    languages=[SR],
)

MULTI_WIKI_QA_SR_CONFIG = DatasetConfig(
    name="multi-wiki-qa-sr",
    pretty_name="the truncated version of the Serbian part of the reading "
    "comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-sr-mini",
    task=RC,
    languages=[SR],
)

LR_SUM_SR_CONFIG = DatasetConfig(
    name="lr-sum-sr",
    pretty_name="the truncated version of the Serbian part of the "
    "summarisation dataset LR-Sum",
    huggingface_id="EuroEval/lr-sum-sr-mini",
    task=SUMM,
    languages=[SR],
)

MMLU_SR_CONFIG = DatasetConfig(
    name="mmlu-sr",
    pretty_name="the truncated version of the Serbian knowledge dataset MMLU-sr, "
    "translated from the English MMLU dataset",
    huggingface_id="EuroEval/mmlu-sr-mini",
    task=KNOW,
    languages=[SR],
)

WINOGRANDE_SR_CONFIG = DatasetConfig(
    name="winogrande-sr",
    pretty_name="the Serbian common-sense reasoning dataset Winogrande-sr, translated "
    "from the English Winogrande dataset",
    huggingface_id="EuroEval/winogrande-sr",
    task=COMMON_SENSE,
    languages=[SR],
    _labels=["a", "b"],
)
