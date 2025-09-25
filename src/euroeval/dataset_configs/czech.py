"""All Czech dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import CS
from ..tasks import LA, NER, RC, SENT

### Official datasets ###

SCALA_CS_CONFIG = DatasetConfig(
    name="scala-cs",
    pretty_name="the Czech part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-cs",
    task=LA,
    languages=[CS],
)

CSFD_SENTIMENT_CONFIG = DatasetConfig(
    name="csfd-sentiment",
    pretty_name="the Czech Movie Database sentiment dataset",
    huggingface_id="EuroEval/csfd-sentiment-mini",
    task=SENT,
    languages=[CS],
)

CS_GEC_CONFIG = DatasetConfig(
    name="cs-gec",
    pretty_name="the Czech linguistic acceptability dataset CS-GEC",
    huggingface_id="EuroEval/cs-gec",
    task=LA,
    languages=[CS],
)

PONER_CONFIG = DatasetConfig(
    name="poner",
    pretty_name="the Czech named entity recognition dataset PONER",
    huggingface_id="EuroEval/poner-mini",
    task=NER,
    languages=[CS],
)

SQAD_CONFIG = DatasetConfig(
    name="sqad",
    pretty_name="the truncated version of the Czech reading comprehension dataset SQAD",
    huggingface_id="EuroEval/sqad-mini",
    task=RC,
    languages=[CS],
)
