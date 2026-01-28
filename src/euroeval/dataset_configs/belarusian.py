"""All Belarusian dataset configurations used in EuroEval."""

from ..data_models import DatasetConfig
from ..languages import BELARUSIAN
from ..tasks import LA, NER, SENT

### Official datasets ###

BESLS_CONFIG = DatasetConfig(
    name="besls",
    pretty_name="BeSLS",
    source="EuroEval/besls",
    task=SENT,
    languages=[BELARUSIAN],
)

SCALA_BE_CONFIG = DatasetConfig(
    name="scala-be",
    pretty_name="ScaLA-be",
    source="EuroEval/scala-be",
    task=LA,
    languages=[BELARUSIAN],
)

WIKIANN_BE_CONFIG = DatasetConfig(
    name="wikiann-be",
    pretty_name="WikiANN-be",
    source="EuroEval/wikiann-be-mini",
    task=NER,
    languages=[BELARUSIAN],
)
