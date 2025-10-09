"""All Slovak dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import SK
from ..tasks import LA, SENT

### Official datasets ###

CSFD_SENTIMENT_CONFIG = DatasetConfig(
    name="csfd-sentiment-sk",
    pretty_name="the Slovak Movie Database sentiment dataset",
    huggingface_id="EuroEval/csfd-sentiment-sk-mini",
    task=SENT,
    languages=[SK],
)

SCALA_SK_CONFIG = DatasetConfig(
    name="scala-sk",
    pretty_name="the Slovak part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-cs",
    task=LA,
    languages=[SK],
)
