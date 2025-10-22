"""All Slovak dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import EL
from ..tasks import LA, SENT

### Official datasets ###

GREEK_SA_CONFIG = DatasetConfig(
    name="greek-sa",
    pretty_name="the truncated version of the Greek sentiment "
    "classification dataset Greek-SA",
    huggingface_id="EuroEval/greek-sa-mini",
    task=SENT,
    languages=[EL],
    _labels=["negative", "positive"],
)

SCALA_EL_CONFIG = DatasetConfig(
    name="scala-el",
    pretty_name="the Greek part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-el",
    task=LA,
    languages=[EL],
)
