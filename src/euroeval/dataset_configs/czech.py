"""All Czech dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import CS
from ..tasks import SENT

### Official datasets ###

CSFD_SENTIMENT_CONFIG = DatasetConfig(
    name="csfd-sentiment",
    pretty_name="the Czech Movie Database sentiment dataset",
    huggingface_id="EuroEval/csfd-sentiment-mini",
    task=SENT,
    languages=[CS],
)
