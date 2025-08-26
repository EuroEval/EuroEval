"""All Estonian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import ET
from ..tasks import SENT

### Official datasets ###

ESTONIAN_VALENCE_CONFIG = DatasetConfig(
    name="estonian-valence",
    pretty_name="the Estonian sentiment classification dataset Estonian Valence",
    huggingface_id="EuroEval/estonian-valence",
    task=SENT,
    languages=[ET],
)
