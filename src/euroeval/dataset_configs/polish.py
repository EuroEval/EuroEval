"""All Polish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import PL
from ..tasks import RC

### Official datasets ###

POQUAD_CONFIG = DatasetConfig(
    name="poquad",
    pretty_name="the Polish question answering dataset PoQuAD",
    huggingface_id="EuroEval/poquad",
    task=RC,
    languages=[PL],
)
