"""All Polish dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import PL
from ..tasks import RC, SENT

### Official datasets ###

POQUAD_CONFIG = DatasetConfig(
    name="poquad",
    pretty_name="the Polish question answering dataset PoQuAD",
    huggingface_id="EuroEval/poquad",
    task=RC,
    languages=[PL],
)

POLEMO2_CONFIG = DatasetConfig(
    name="polemo2",
    pretty_name="the Polish sentiment analysis dataset PolEmo 2.0",
    huggingface_id="EuroEval/polemo2",
    task=SENT,
    languages=[PL],
)
