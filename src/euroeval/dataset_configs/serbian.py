"""All Serbian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import SR
from ..tasks import LA, NER, SENT

### Official datasets ###

MMS_SR_CONFIG = DatasetConfig(
    name="mms-sr",
    pretty_name="the truncated version of the Serbian part of the MMS sentiment "
    "dataset MMS-sr",
    huggingface_id="EuroEval/mms-sr-mini",
    task=SENT,
    languages=[SR],
)

SCALA_SR_CONFIG = DatasetConfig(
    name="scala-sr",
    pretty_name="the Serbian part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-sr",
    task=LA,
    languages=[SR],
)

UNER_SR_CONFIG = DatasetConfig(
    name="uner-sr",
    pretty_name="the truncated version of the Serbian named entity recognition dataset "
    "UNER-sr",
    huggingface_id="EuroEval/uner-sr-mini",
    task=NER,
    languages=[SR],
)
