"""All Lithuanian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig, ModelType
from ..languages import LT
from ..tasks import COMMON_SENSE, KNOW, LA, NER, SENT

### Official datasets ###

SCALA_LT_CONFIG = DatasetConfig(
    name="scala-lt",
    pretty_name="the Lithuanian part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-lt",
    task=LA,
    languages=[LT],
)

LITHUANIAN_EMOTIONS_CONFIG = DatasetConfig(
    name="lithuanian-emotions",
    pretty_name="the truncated version of the Lithuanian sentiment "
    "classification dataset Lithuanian Emotions",
    huggingface_id="EuroEval/lithuanian-emotions-mini",
    task=SENT,
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

LT_HISTORY_CONFIG = DatasetConfig(
    name="lt-history",
    pretty_name="the Lithuanian history knowledge dataset LT-History",
    huggingface_id="EuroEval/lt-history",
    task=KNOW,
    languages=[LT],
)

WINOGRANDE_LV_CONFIG = DatasetConfig(
    name="winogrande-lv",
    pretty_name="the Lithuanian common-sense reasoning dataset Winogrande-lv, "
    "translated from the English Winogrande dataset",
    huggingface_id="EuroEval/winogrande-lv",
    task=COMMON_SENSE,
    languages=[LT],
    splits=["train", "test"],
    _labels=["a", "b"],
    _allowed_model_types=[ModelType.GENERATIVE],
)
