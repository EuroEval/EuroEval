"""All Lithuanian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import LT
from ..tasks import COMMON_SENSE, KNOW, LA, NER, RC, SENT

### Official datasets ###

LITHUANIAN_EMOTIONS_CONFIG = DatasetConfig(
    name="lithuanian-emotions",
    pretty_name="the truncated version of the Lithuanian sentiment "
    "classification dataset Lithuanian Emotions",
    huggingface_id="EuroEval/lithuanian-emotions-mini",
    task=SENT,
    languages=[LT],
)

SCALA_LT_CONFIG = DatasetConfig(
    name="scala-lt",
    pretty_name="the Lithuanian part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-lt",
    task=LA,
    languages=[LT],
)

WIKIANN_LT_CONFIG = DatasetConfig(
    name="wikiann-lt",
    pretty_name="the truncated version of the Lithuanian part of the named entity "
    "recognition dataset WikiANN",
    huggingface_id="EuroEval/wikiann-lt-mini",
    task=NER,
    languages=[LT],
)

MULTI_WIKI_QA_LT_CONFIG = DatasetConfig(
    name="multi-wiki-qa-lt",
    pretty_name="the truncated version of the Lithuanian part of the reading "
    "comprehension dataset MultiWikiQA",
    huggingface_id="EuroEval/multi-wiki-qa-lt-mini",
    task=RC,
    languages=[LT],
)

LT_HISTORY_CONFIG = DatasetConfig(
    name="lt-history",
    pretty_name="the Lithuanian knowledge dataset LT-History",
    huggingface_id="EuroEval/lt-history",
    task=KNOW,
    languages=[LT],
    splits=["train", "test"],
)

WINOGRANDE_LT_CONFIG = DatasetConfig(
    name="winogrande-lt",
    pretty_name="the Lithuanian common-sense reasoning dataset Winogrande-lt, "
    "translated from the English Winogrande dataset",
    huggingface_id="EuroEval/winogrande-lt",
    task=COMMON_SENSE,
    languages=[LT],
    splits=["train", "test"],
    _labels=["a", "b"],
)
