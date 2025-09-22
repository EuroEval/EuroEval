"""All Lithuanian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import LT
from ..tasks import SENT

### Official datasets ###

LITHUANIAN_EMOTIONS_CONFIG = DatasetConfig(
    name="lithuanian-emotions",
    pretty_name="the truncated version of the Lithuanian sentiment "
    "classification dataset Lithuanian Emotions",
    huggingface_id="EuroEval/lithuanian-emotions-mini",
    task=SENT,
    languages=[LT],
)
