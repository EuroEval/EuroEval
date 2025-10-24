"""All Serbian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import SR
from ..tasks import SENT

### Official datasets ###

MMS_SR_CONFIG = DatasetConfig(
    name="mms-sr",
    pretty_name="the truncated version of the Serbian part of the MMS sentiment "
    "dataset MMS-sr",
    huggingface_id="EuroEval/mms-sr-mini",
    task=SENT,
    languages=[SR],
)
