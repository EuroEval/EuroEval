"""All Portuguese dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import PT
from ..tasks import COMMON_SENSE, KNOW, MCRC, SENT

### Unofficial datasets ###

PREFIX = "EuroEval"

SST2_PT_CONFIG = DatasetConfig(
    name="sst2-pt",
    pretty_name="extraglue-sst2-pt dataset for portuguese (PT)",
    huggingface_id=f"{PREFIX}/extraglue-sst2-pt",
    task=SENT,
    languages=[PT],
    unofficial=True,
    _labels=["positive", "negative"],
)

BOOLQ_PT_CONFIG = DatasetConfig(
    name="boolq-pt",
    pretty_name="extraglue-boolq-pt dataset for portuguese (PT)",
    huggingface_id=f"{PREFIX}/extraglue-boolq-pt",
    task=MCRC,
    languages=[PT],
    unofficial=True,
)


MMLU_PT_CONFIG = DatasetConfig(
    name="mmlu-pt",
    pretty_name="Portuguese version of mmlu mini",
    huggingface_id=f"{PREFIX}/mmlu-pt-mini",
    task=KNOW,
    languages=[PT],
    unofficial=True,
)


HELLASWAG_PT_CONFIG = DatasetConfig(
    name="hellaswag-pt",
    pretty_name="Portuguese version of HellaSwag mini",
    huggingface_id=f"{PREFIX}/hellaswag-pt-mini",
    task=COMMON_SENSE,
    languages=[PT],
    unofficial=True,
)
