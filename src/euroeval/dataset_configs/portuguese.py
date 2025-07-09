"""All Portuguese dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import PT
from ..tasks import MCRC, SENT

### Unofficial datasets ###

SST2_PT_CONFIG = DatasetConfig(
    name="sst2-pt",
    pretty_name="extraglue-sst2-pt dataset for portuguese (PT)",
    huggingface_id="EuroEval/extraglue-sst2-pt",
    task=SENT,
    languages=[PT],
    unofficial=True,
    _labels=["positive", "negative"],
)

BOOLQ_PT_CONFIG = DatasetConfig(
    name="boolq-pt",
    pretty_name="extraglue-boolq-pt dataset for portuguese (PT)",
    huggingface_id="EuroEval/extraglue-boolq-pt",
    task=MCRC,
    languages=[PT],
    unofficial=True,
)
