"""All Bulgarian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import BG
from ..tasks import LA, SENT

### Official datasets ###

CINEXIO_CONFIG = DatasetConfig(
    name="cinexio",
    pretty_name="the truncated version of the Bulgarian sentiment "
    "classification dataset Cinexio",
    huggingface_id="EuroEval/cinexio-mini",
    task=SENT,
    languages=[BG],
)

SCALA_BG_CONFIG = DatasetConfig(
    name="scala-bg",
    pretty_name="the Bulgarian part of the linguistic acceptability dataset ScaLA",
    huggingface_id="EuroEval/scala-bg",
    task=LA,
    languages=[BG],
)
