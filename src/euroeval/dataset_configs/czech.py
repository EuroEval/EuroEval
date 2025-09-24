"""All Czech dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import CS
from ..tasks import LA, SENT

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
