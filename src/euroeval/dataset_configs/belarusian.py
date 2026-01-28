"""All Belarusian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import BELARUSIAN
from ..tasks import SENT

### Official datasets ###

BESLS_CONFIG = DatasetConfig(
    name="besls",
    pretty_name="BeSLS",
    source="EuroEval/besls",
    task=SENT,
    languages=[BELARUSIAN],
)
