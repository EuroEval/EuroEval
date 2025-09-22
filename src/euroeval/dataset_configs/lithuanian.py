"""All Lithuanian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import LT
from ..tasks import LA, NER, SENT

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
